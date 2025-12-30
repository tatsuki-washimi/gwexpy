import sys
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
from gwpy.timeseries import TimeSeries

import loaders
import products
from normalize import normalize_series
from engine import Engine
from tabs import create_input_tab, create_measurement_tab, create_excitation_tab, create_result_tab
from graph_panel import GraphPanel
from nds.cache import NDSDataCache
# Excitation Module Imports
from excitation.generator import SignalGenerator
from excitation.params import GeneratorParams
from channel_browser import ChannelBrowserDialog

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyaggui : a diaggui-like gwexpy GUI-tool")
        self.resize(1100, 850)

        mb = self.menuBar()
        for m in ["File", "Edit", "Measurement", "Plot", "Window", "Help"]:
            menu = mb.addMenu(m)
            if m == "File":
                op = QtWidgets.QAction("Open...", self); op.triggered.connect(self.open_file_dialog); menu.addAction(op)
                menu.addAction("Exit").triggered.connect(self.close)

        self.tabs = QtWidgets.QTabWidget()
        input_tab, self.input_controls = create_input_tab()
        self.tabs.addTab(input_tab, "Input")
        meas_tab, self.meas_controls = create_measurement_tab()
        self.tabs.addTab(meas_tab, "Measurement")
        
        # Create Excitation Tab and capture controls
        exc_tab, self.exc_controls = create_excitation_tab()
        self.tabs.addTab(exc_tab, "Excitation (Simulation)") 
        # Enable it now that we have implementation
        self.tabs.setTabEnabled(2, True)

        res_tab, self.graph_info1, self.graph_info2, self.traces1, self.traces2 = create_result_tab(on_import=self.open_file_dialog)
        self.tabs.addTab(res_tab, "Result")

        self.engine = Engine()
        self.loaded_products = {}; self.is_file_mode = False; self.is_loading_file = False
        
        # NDS Integration
        self.nds_cache = NDSDataCache()
        self.nds_cache.signal_data.connect(self.on_nds_data)
        self.nds_latest_raw = None # Stores latest DataBufferDict
        self.data_source = 'SIM' # SIM, FILE, NDS
        
        # Signal Generator
        self.sig_gen = SignalGenerator()
        
        self.input_controls['ds_combo'].currentTextChanged.connect(self.on_source_changed)

        def connect_trace_combos(graph_info):
            for ctrl in graph_info['traces']:
                if 'chan_a' in ctrl: ctrl['chan_a'].currentTextChanged.connect(self.on_trace_channel_changed)
                if 'chan_b' in ctrl: ctrl['chan_b'].currentTextChanged.connect(self.on_trace_channel_changed)
            # Also connect graph type combo
            if 'graph_combo' in graph_info:
                 graph_info['graph_combo'].currentTextChanged.connect(self.on_trace_channel_changed)

        connect_trace_combos(self.graph_info1); connect_trace_combos(self.graph_info2)

        central = QtWidgets.QWidget(); self.setCentralWidget(central); cv = QtWidgets.QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 0, 0); cv.addWidget(self.tabs)
        
        bot = QtWidgets.QHBoxLayout(); central.layout().addLayout(bot)
        self.btn_start = QtWidgets.QPushButton("Start"); self.btn_start.setMinimumWidth(120); self.btn_start.setStyleSheet("font-weight: bold;")
        self.btn_pause = QtWidgets.QPushButton("Pause"); self.btn_pause.setEnabled(False); self.btn_pause.setMinimumWidth(120)
        self.btn_resume = QtWidgets.QPushButton("Resume"); self.btn_resume.setEnabled(False); self.btn_resume.setMinimumWidth(120)
        self.btn_abort = QtWidgets.QPushButton("Abort"); self.btn_abort.setMinimumWidth(120)
        [bot.addWidget(b) or (bot.addStretch(1) if b != self.btn_abort else None) for b in [self.btn_start, self.btn_pause, self.btn_resume, self.btn_abort]]
        self.btn_start.clicked.connect(self.start_animation); self.btn_pause.clicked.connect(self.pause_animation)
        self.btn_resume.clicked.connect(self.resume_animation); self.btn_abort.clicked.connect(self.stop_animation)

        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.update_graphs); self.time_counter = 0.0
        
        # Connect Measurement Channel Updates
        self.meas_controls['set_change_callback'](self.on_measurement_channel_changed)
        self.meas_controls['btn_browse'].clicked.connect(lambda: self.show_channel_browser())
        
        # Connect Individual Channel Browse buttons
        for i, row in enumerate(self.meas_controls['grid_refs']):
            # Each row is (label, checkbox, combobox, browse_button)
            btn = row[3]
            btn.clicked.connect(lambda checked, idx=i: self.show_channel_browser(start_slot=idx))
        
        # Connect Excitation Channel Updates to trigger channel list refresh
        if self.exc_controls and 'panels' in self.exc_controls:
            for p in self.exc_controls['panels']:
                # When Active toggled or Channel name changed
                p['active'].toggled.connect(self.on_measurement_channel_changed)
                p['ex_chan'].editTextChanged.connect(self.on_measurement_channel_changed)
                p['ex_chan'].currentIndexChanged.connect(self.on_measurement_channel_changed)

        # Initial sync
        self.on_measurement_channel_changed()

    def on_measurement_channel_changed(self):
        # Update Result tab comboboxes based on active Measurement channels
        active_channels = []
        states = self.meas_controls['channel_states']
        for s in states:
            if s['active'] and s['name']:
                active_channels.append(s['name'])
        
        # Add generated channels if known? 
        # Or should they always be available?
        # Let's add standard "Excitation" if enabled?
        # Better: Do it dynamically or just rely on them being added when active.
        # Ideally, user should see "Excitation" in the list even if stopped?
        # For now, let's keep it simple: Measurement channels define the majority.
        # We can append "Exictation" and "Sum:..." implicitly if we want persistent selection.
        # But if we clear the combo, we lose the selection if it's not in the list.
        # Let's add 'Excitation' to the list if the module is theoretically available.
        # if "Excitation" not in active_channels:
        #      active_channels.append("Excitation")
             
        # Feature: Add channels defined in Excitation Tab (active ones) to the list
        if hasattr(self, 'exc_controls') and 'panels' in self.exc_controls:
            for p in self.exc_controls['panels']:
                if p['active'].isChecked():
                     # Channel used as injection TARGET
                     tgt = p['ex_chan'].currentText()
                     if tgt and tgt not in active_channels:
                         active_channels.append(tgt)
                     # Also Add Sum:{tgt} if it's injection? 
                     # For now, let's blindly add the Target name.
                     # If user typed a NEW name (not in measurement), it's a pure sim channel.
                     # If it's existing, it's already in active_channels (measurement).

             
        # Also, update the Target Channel list in Excitation Tab (New Multi-panel support)
        if hasattr(self, 'exc_controls') and 'target_combos' in self.exc_controls:
             for cb_target in self.exc_controls['target_combos']:
                 curr = cb_target.currentText()
                 cb_target.blockSignals(True)
                 cb_target.clear()
                 cb_target.addItems(active_channels) 
                 # Restore
                 idx = cb_target.findText(curr)
                 if idx != -1: 
                     cb_target.setCurrentIndex(idx)
                 else:
                     # If custom name (e.g. default Excitation-0) is not in list, preserve it visually
                     cb_target.setEditText(curr)
                     
                 cb_target.blockSignals(False)
        # Fallback for old key if mixed versions (optional, can remove)
        elif hasattr(self, 'exc_controls') and 'ex_target' in self.exc_controls:
             cb_target = self.exc_controls['ex_target']
             curr = cb_target.currentText()
             cb_target.blockSignals(True)
             cb_target.clear()
             cb_target.addItems(active_channels)
             idx = cb_target.findText(curr)
             if idx != -1: cb_target.setCurrentIndex(idx)
             cb_target.blockSignals(False)

        # Update comboboxes in Result tab
        # Note: We should preserve current selection if it is still valid
        for info in [self.graph_info1, self.graph_info2]:
            for ctrl in info['traces']:
                for c_key in ['chan_a', 'chan_b']:
                    if c_key in ctrl:
                        combo = ctrl[c_key]
                        current = combo.currentText()
                        combo.blockSignals(True)
                        combo.clear()
                        combo.addItems(active_channels)
                        
                        # Restore previous selection if possible, otherwise default or empty
                        idx = combo.findText(current)
                        if idx != -1:
                            combo.setCurrentIndex(idx)
                        elif active_channels:
                            combo.setCurrentIndex(0)
                        
                        combo.blockSignals(False)

    def show_channel_browser(self, start_slot=None):
        # Determine which server/port to use based on current selection in Input tab
        ds_mode = self.input_controls['ds_combo'].currentText()
        if ds_mode == "NDS2":
            server = self.input_controls['nds2_server'].currentText()
            port = self.input_controls['nds2_port'].value()
        else:
            server = self.input_controls['nds_server'].currentText()
            port = self.input_controls['nds_port'].value()
        
        dlg = ChannelBrowserDialog(server, port, self)
        if dlg.exec_():
            chans = dlg.selected_channels
            if not chans:
                return

            # Get current display offset (0, 16, 32, ...)
            offset = self.meas_controls['get_bank_offset']()
            states = self.meas_controls['channel_states']
            new_states = [dict(s) for s in states] 

            if start_slot is not None:
                # 1. Individual row triggered (starts at start_slot + offset)
                idx = offset + start_slot
                
                # Replace the current one with the first selected channel
                new_states[idx]['name'] = chans[0]
                new_states[idx]['active'] = True
                
                # Fill subsequent selected channels into empty slots after idx
                chan_idx = 1
                if len(chans) > 1:
                    for j in range(idx + 1, len(new_states)):
                        if chan_idx >= len(chans): break
                        if not new_states[j]['name']:
                            new_states[j]['name'] = chans[chan_idx]
                            new_states[j]['active'] = True
                            chan_idx += 1
                count = chan_idx
            else:
                # 2. Master "Channel Browser..." button (find any empty slots from start)
                count = 0
                for i in range(len(new_states)):
                    if not new_states[i]['name']:
                        new_states[i]['name'] = chans[count]
                        new_states[i]['active'] = True
                        count += 1
                        if count >= len(chans):
                            break
            
            # Update UI to reflect model
            self.meas_controls['set_all_channels'](new_states)
            print(f"Added {count} channels from {server}:{port}")

    def on_source_changed(self, text):
        self.data_source = text
        if text == "NDS":
             self.nds_latest_raw = None

    def on_nds_data(self, buffers):
        self.nds_latest_raw = buffers

    def start_animation(self):
        # Validate Excitation Channels (uniqueness)
        if self.exc_controls and 'panels' in self.exc_controls:
            active_names = []
            for i, p in enumerate(self.exc_controls['panels']):
                if p['active'].isChecked():
                    name = p['ex_chan'].currentText()
                    if name in active_names:
                        QtWidgets.QMessageBox.critical(self, "Configuration Error", f"Duplicate Excitation Channel name detected: '{name}'.\nPlease assign unique names for active excitation channels.")
                        return
                    active_names.append(name)
    
        self.tabs.setCurrentIndex(3)
        self.is_file_mode = False 
        
        # Read Data Source directly from UI to avoid signal issues
        self.data_source = self.input_controls['ds_combo'].currentText()
        print(f"DEBUG: start_animation called. DataSource: {self.data_source}")
        
        self.btn_start.setEnabled(False); self.btn_pause.setEnabled(True); self.btn_resume.setEnabled(False); self.btn_abort.setEnabled(True)
        # self.tabs.setTabEnabled(0, False) # Allow switching to Input tab even during operation
        
        use_pc_audio = self.input_controls['pcaudio'].isChecked()
        
        if self.data_source == 'NDS' or use_pc_audio:
            # Collect channels
            channels = []
            sim_channels = ["HF_sine", "LF_sine", "beating_sine", "white_noise", "sine_plus_noise", "square_wave", "sawtooth_wave", "random_walk"]
            states = self.meas_controls['channel_states']
            for s in states:
                if s['active'] and s['name']:
                    if s['name'] in sim_channels: 
                        continue
                    
                    if s['name'].startswith("PC:"):
                        if use_pc_audio:
                            channels.append(s['name'])
                    elif self.data_source == 'NDS':
                        channels.append(s['name'])
            
            if channels:
                # Determine which NDS server/port to use (ignored by PC Audio channels anyway)
                ds_mode = self.input_controls['ds_combo'].currentText()
                if ds_mode == "NDS2":
                    server = self.input_controls['nds2_server'].currentText()
                    port = self.input_controls['nds2_port'].value()
                else:
                    server = self.input_controls['nds_server'].currentText()
                    port = self.input_controls['nds_port'].value()
                
                print(f"DEBUG: Starting DataCache for channels: {channels}")
                self.nds_cache.set_server(f"{server}:{port}")
                self.nds_cache.set_channels(channels)
                
                win_sec = self.input_controls['nds_win'].value()
                self.nds_cache.online_start(lookback=win_sec)
        
        self.timer.start(50)

    def pause_animation(self):
        self.timer.stop(); self.btn_start.setEnabled(False); self.btn_pause.setEnabled(False); self.btn_resume.setEnabled(True); self.btn_abort.setEnabled(True)
        self.nds_cache.online_stop()

    def resume_animation(self):
        self.timer.start(50); self.btn_start.setEnabled(False); self.btn_pause.setEnabled(True); self.btn_resume.setEnabled(False); self.btn_abort.setEnabled(True)
        win = self.input_controls['nds_win'].value()
        self.nds_cache.online_start(lookback=win)

    def stop_animation(self):
        self.timer.stop(); self.btn_start.setEnabled(True); self.btn_pause.setEnabled(False); self.btn_resume.setEnabled(False); self.btn_abort.setEnabled(True)
        self.nds_cache.reset(); self.nds_latest_raw = None
        for t_list in [self.traces1, self.traces2]:
            for t in t_list: t['curve'].setData([], []); t['bar'].setOpts(x=[0], height=[0]); t['img'].clear()
        self.time_counter = 0.0

    def get_ui_params(self):
        p = {}; c = self.meas_controls
        p.update({'start_freq': c['start_freq'].value(), 'stop_freq': c['stop_freq'].value(), 'bw': c['bw'].value(), 'averages': c['averages'].value(), 'overlap': c['overlap'].value()/100.0, 'window': c['window'].currentText().lower()})
        p['avg_type'] = 'fixed' if c['avg_type_fixed'].isChecked() else ('exponential' if c['avg_type_exp'].isChecked() else 'accumulative')
        return p

    def update_graphs(self):
        if self.is_file_mode: return
        
        data_map = {}
        
        # NDS Mode Logic: Build data_map from buffers
        current_times = None
        current_fs = 16384 # Fallback
        
        # NDS Mode Logic: Build data_map from buffers
        current_times = None
        current_fs = 16384 # Fallback
        
        # Check if we have Active Excitation to decide if we need a fallback timebase
        has_active_excitation = False
        if self.exc_controls and 'panels' in self.exc_controls:
            for p in self.exc_controls['panels']:
                if p['active'].isChecked():
                    has_active_excitation = True; break

        if self.data_source == 'NDS':
             # Try to get data
             if self.nds_latest_raw:
                 for ch_name, buf in self.nds_latest_raw.items():
                     y_vals = buf.data_map.get('raw')
                     if y_vals is not None and len(y_vals) > 0 and len(buf.tarray) == len(y_vals):
                         ts = TimeSeries(y_vals, t0=buf.tarray[0], dt=buf.step, name=ch_name)
                         data_map[ch_name] = ts
                         if current_times is None:
                             current_times = buf.tarray
                             current_fs = 1.0/buf.step
             
             # Fallback if no NDS data but we want to see Excitation
             if not data_map and has_active_excitation:
                 self.time_counter += 0.05
                 params = self.get_ui_params()
                 duration = max(10.0, 2.0/params.get('bw', 1.0))
                 fs = 16384
                 current_fs = fs
                 t0 = self.time_counter
                 n = int(fs * duration)
                 current_times = np.linspace(t0, t0 + duration, n, endpoint=False)
             
             if not data_map and current_times is None: return # No data and no fallback

        # SIM Logic (Legacy/Fallback, though UI removed)
        elif self.data_source == 'SIM':
            self.time_counter += 0.05; params = self.get_ui_params(); self.engine.configure(params)
            duration = max(10.0, 2.0/params.get('bw', 1.0))
            fs = 16384
            current_fs = fs
            t0 = self.time_counter
            n = int(fs * duration)
            current_times = np.linspace(t0, t0 + duration, n, endpoint=False)
            
            # ... (Sim generators removed from here as per request) ...
                     
        # --- EXCITATION / SIGNAL GENERATION ---
        # Initialize global Excitation readback
        # This represents the sum of all generated signals, useful for TF calculation
        total_excitation = np.zeros(len(current_times)) if current_times is not None else None
        
        if self.exc_controls and 'panels' in self.exc_controls and current_times is not None:
             panels = self.exc_controls['panels']
             
             for p in panels:
                 # Check if panel is active
                 if not p['active'].isChecked(): continue
                 
                 # Waveform Parameters
                 gen_params = GeneratorParams(
                    enabled=True,
                    waveform_type=p['waveform'].currentText(),
                    amplitude=p['amp'].value(),
                    frequency=p['freq'].value(),
                    offset=p['offset'].value(),
                    phase=p['phase'].value(),
                    start_freq=p['freq'].value(), # 'Frequency' box = Start
                    stop_freq=p['fstart'].value(), # 'Freq. Range' box = Stop
                    output_mode="Sum", # Always Sum/Inject behavior
                    target_channel=p['ex_chan'].currentText() # User input
                )
                 
                 # Generate
                 sig = self.sig_gen.generate(current_times, gen_params)
                 
                 # Accumulate to global readback
                 if total_excitation is not None:
                     total_excitation += sig
                     
                 # Injection / Assignment
                 tgt = gen_params.target_channel
                 if not tgt: tgt = "Excitation" # Default if empty
                 
                 # If target exists in data (Measurement Channel), INJECT (Sum)
                 if tgt in data_map:
                     try:
                         # In-place addition to simulate injection
                         data_map[tgt] = data_map[tgt] + sig
                     except Exception as e:
                         print(f"Injection Error for {tgt}: {e}")
                 else:
                     # If target does NOT exist (e.g. Pure Simulation or 'Excitation' placeholder),
                     # Create it.
                     # Note: If user typed 'MySig', it creates 'MySig'.
                     # But user can't select 'MySig' in Result tab unless it's in the list.
                     # So usually they should use 'Excitation' or an existing channel.
                     ts_sig = TimeSeries(sig, t0=current_times[0], sample_rate=current_fs, name=tgt)
                     data_map[tgt] = ts_sig

        # Always publish the global 'Excitation' channel if we have any signal
        if total_excitation is not None and np.any(total_excitation):
            data_map['Excitation'] = TimeSeries(total_excitation, t0=current_times[0], sample_rate=current_fs, name='Excitation')
        elif 'Excitation' not in data_map and current_times is not None:
             # Ensure zero-signal Excitation exists if enabled, or just leave it?
             # Better to have it if we want to allow selecting it without crash.
             pass

        for plot_idx, info_root in enumerate([self.graph_info1, self.graph_info2]):
            # Update meta info in Param tab (Start, Avgs, BW)
            ui_p = self.get_ui_params()
            if data_map:
                first_ts = next(iter(data_map.values()))
                if hasattr(first_ts, 't0'):
                    info_root['panel'].meta_info['start_time'] = first_ts.t0.value
                    info_root['panel'].meta_info['avgs'] = ui_p.get('averages', 1)
                    info_root['panel'].meta_info['bw'] = ui_p.get('bw', 0)
                    info_root['panel'].update_params_display()

            try:
                traces_items = [self.traces1, self.traces2][plot_idx]; g_type = info_root['graph_combo'].currentText()
                results = self.engine.compute(data_map, g_type, [{'active': c['active'].isChecked(), 'ch_a': c['chan_a'].currentText(), 'ch_b': c['chan_b'].currentText()} for c in info_root['traces']])
                for t_idx, result in enumerate(results):
                    try:
                        tr = traces_items[t_idx]; curve, bar, img = tr['curve'], tr['bar'], tr['img']
                        if result is None: curve.setData([], []); (bar.setOpts(height=[]) if bar.isVisible() else None); img.clear(); continue
                        if isinstance(result, dict) and result.get('type') == 'spectrogram':
                            data = result['value']; disp = info_root.get('units', {}).get('display_y').currentText()
                            if disp == "dB": data = 10 * np.log10(np.abs(data)+1e-20)
                            elif disp == "Phase": data = np.angle(data, deg=True) if np.iscomplexobj(data) else np.zeros_like(data)
                            elif disp == "Magnitude": data = np.abs(data)
                            img.setImage(data, autoLevels=False); img.setLevels([np.min(data), np.max(data)])
                            if len(result['times'])>1 and len(result['freqs'])>1:
                                img.setRect(QtCore.QRectF(result['times'][0], result['freqs'][0], (result['times'][1]-result['times'][0])*len(result['times']), (result['freqs'][1]-result['freqs'][0])*len(result['freqs']))); img.setVisible(True); curve.setData([], []); (bar.setOpts(height=[]) if bar.isVisible() else None)
                            else: img.clear()
                            continue
                        img.setVisible(False); x_vals, y_vals = result; disp = info_root.get('units', {}).get('display_y').currentText()
                        if disp == "dB": y_vals = (10 if "Power" in g_type or "Squared" in g_type else 20) * np.log10(np.abs(y_vals)+1e-20)
                        elif disp == "Phase": y_vals = np.angle(y_vals, deg=True) if np.iscomplexobj(y_vals) else np.zeros_like(y_vals)
                        elif disp == "Magnitude": y_vals = np.abs(y_vals)
                        # "None" case does nothing, keeping y_vals as is
                        curve.setData(x_vals, y_vals)
                        if bar.isVisible(): bar.setOpts(x=x_vals, height=y_vals, width=(x_vals[1]-x_vals[0] if len(x_vals)>1 else 1))
                    except Exception as e:
                        print(f"Error updating Graph {plot_idx+1} Trace {t_idx}: {e}")
                if 'range_updater' in info_root: info_root['range_updater']()
            except Exception as e:
                print(f"Error in update_graphs for Graph {plot_idx+1}: {e}")

    def on_trace_channel_changed(self):
        if not self.is_loading_file and self.is_file_mode: self.update_file_plot()

    def update_file_plot(self):
        if not self.loaded_products: return
        for graph_idx in [0, 1]:
            info, traces = (self.graph_info1, self.traces1) if graph_idx==0 else (self.graph_info2, self.traces2)
            # Update t0 from loaded products
            for p_type in ["TS", "ASD", "PSD", "CSD", "Spectrogram"]:
                items = self.loaded_products.get(p_type)
                if items:
                    first_val = next(iter(items.values()))
                    if hasattr(first_val, 't0'):
                        info['panel'].meta_info['start_time'] = first_val.t0.value
                        info['panel'].update_params_display()
                        break
            try:
                g_type = info['graph_combo'].currentText(); p_name = "TS"
                if g_type == "Amplitude Spectral Density": p_name = "ASD"
                elif g_type == "Cross Spectral Density": p_name = "CSD"
                elif g_type == "Coherence": p_name = "COH"
                elif g_type == "Transfer Function": p_name = "TF"
                elif g_type == "Spectrogram": p_name = "Spectrogram"
                
                items = self.loaded_products.get(p_name) or (self.loaded_products.get("PSD") if p_name=="ASD" else None) or (self.loaded_products.get("Spectrogram") if p_name=="Spectrogram" else None)
                if not items: continue
                
                for t_idx, ctrl in enumerate(info['traces']):
                    try:
                        tr = traces[t_idx]; curve, bar, img = tr['curve'], tr['bar'], tr['img']
                        
                        if not ctrl['active'].isChecked(): 
                             curve.setData([], []); curve.setVisible(False); img.clear(); img.setVisible(False); continue
                        
                        ch_a, ch_b = ctrl['chan_a'].currentText(), ctrl['chan_b'].currentText()
                        key = ch_a if p_name in ["TS", "ASD", "PSD", "Spectrogram"] else (ch_b, ch_a)
                        val = items.get(key)
                        
                        # Handle Spectrogram (2D)
                        is_spectrogram = (hasattr(val, 'ndim') and val.ndim == 2 and hasattr(val, 'value') and hasattr(val, 'times') and hasattr(val, 'frequencies'))
                        
                        if is_spectrogram:
                            data = val.value
                            times = val.times.value
                            freqs = val.frequencies.value
                            disp = info.get('units', {}).get('display_y').currentText()
                            
                            if disp == "dB": data = 10 * np.log10(np.abs(data)+1e-20)
                            elif disp == "Phase": data = np.angle(data, deg=True) if np.iscomplexobj(data) else np.zeros_like(data)
                            elif disp == "Magnitude": data = np.abs(data)
                            
                            img.setImage(data, autoLevels=False)
                            img.setLevels([np.min(data), np.max(data)])
                            
                            if len(times)>1 and len(freqs)>1:
                                dt = times[1]-times[0]
                                df = freqs[1]-freqs[0]
                                img.setRect(QtCore.QRectF(times[0], freqs[0], dt*len(times), df*len(freqs)))
                                img.setVisible(True)
                                curve.setVisible(False)
                            else: 
                                img.clear()
                            continue
                        
                        # Handle 1D Series
                        res = normalize_series(val) if val is not None else None
                        if res:
                            x, d = res; disp = info.get('units', {}).get('display_y').currentText()
                            if disp == "dB": d = (10 if "Power" in g_type or "Squared" in g_type else 20) * np.log10(np.abs(d)+1e-20)
                            elif disp == "Phase": d = np.angle(d, deg=True) if np.iscomplexobj(d) else np.zeros_like(d)
                            elif disp == "Magnitude": d = np.abs(d)
                            elif disp == "Deg Unwrapped": d = np.degrees(np.unwrap(np.angle(d))) if np.iscomplexobj(d) else np.zeros_like(d)
                            
                            curve.setData(x, d); curve.setVisible(True)
                            img.setVisible(False)
                        else: 
                            curve.setVisible(False)
                            img.setVisible(False)
                    except Exception as e:
                        print(f"Error updating File Plot Graph {graph_idx+1} Trace {t_idx}: {e}")
            except Exception as e:
                print(f"Error in update_file_plot for Graph {graph_idx+1}: {e}")

    def open_file_dialog(self):
        filters = [
            "Data Files (*.xml *.gwf *.h5 *.hdf5 *.csv *.txt)",
            "All Files (*)"
        ]
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Data File", "", ";;".join(filters))
        if f: self.open_file(f)

    def reset_state(self):
        """Reset the entire GUI and engine state to defaults."""
        self.is_loading_file = True
        
        # 1. Reset Engine
        self.engine.reset()
        self.loaded_products = {}
        self.is_file_mode = False
        self.time_counter = 0.0
        
        # 2. Reset Graphs
        # We need to find the GraphPanel instances. 
        # In create_result_tab, they were not explicitly stored in self but are children of result_tab.
        # However, we can find them or use the info objects if they have refs.
        # Let's find them from the tab widget if possible, or assume they are accessible.
        # Actually, in MainWindow.__init__, we have self.graph_info1/2 which are from create_result_tab.
        # We should probably store the GraphPanel instances themselves.
        # For now, let's look for them in the Result tab.
        res_tab = self.tabs.widget(3) # Result tab is index 3
        if res_tab:
            panels = res_tab.findChildren(GraphPanel)
            for p in panels:
                p.reset()
        
        # 3. Reset Measurement Tab
        self.meas_controls['set_all_channels']([])
        
        # 4. Clear NDS status
        self.nds_latest_raw = None
        
        self.is_loading_file = False

    def open_file(self, filename):
        try:
            # Prepare fresh state
            self.reset_state()
            self.is_loading_file = True
            
            products_dict = loaders.load_products(filename)
            if not products_dict: 
                QtWidgets.QMessageBox.warning(self, "Error", "No products loaded")
                self.is_loading_file = False
                return
            
            self.loaded_products = products_dict
            self.is_file_mode = True
            self.timer.stop()
            # Extract common channels
            cols = products.extract_channels(products_dict)
            
            # --- Channel Population Logic ---
            new_states = []
            if filename.lower().endswith('.xml'):
                # Use local XML parser to get Active states (preserves DTT enabled/disabled state)
                xml_channels = loaders.extract_xml_channels(filename)
                if xml_channels:
                     new_states = xml_channels
            
            # Fallback / Non-XML: Use all loaded columns as active channels
            if not new_states:
                 new_states = [{'name': c, 'active': True} for c in cols]
                 
            self.meas_controls['set_all_channels'](new_states)
            # --- End Channel Population ---
            
            # --- Update Time / Duration Info ---
            try:
                # Find the first available product to get time info
                first_prod_dict = next(iter(products_dict.values()))
                first_series = next(iter(first_prod_dict.values()))
                
                t0 = None
                duration = None
                
                # Check for TimeSeries
                if hasattr(first_series, 't0'):
                    t0 = first_series.t0.value if hasattr(first_series.t0, 'value') else first_series.t0
                    if hasattr(first_series, 'duration'):
                        duration = first_series.duration.value if hasattr(first_series.duration, 'value') else first_series.duration
                    elif hasattr(first_series, 'dt') and hasattr(first_series, 'size'):
                        dt = first_series.dt.value if hasattr(first_series.dt, 'value') else first_series.dt
                        duration = dt * first_series.size
                
                # Check for FrequencySeries (epoch)
                elif hasattr(first_series, 'epoch'):
                     t0 = first_series.epoch.value if hasattr(first_series.epoch, 'value') else first_series.epoch
                     # Duration for ASD? Maybe unavailable.
                     
                if t0 is not None:
                     # Update Start Time GPS
                     if 'start_gps' in self.meas_controls:
                         self.meas_controls['start_gps'].setValue(int(t0))
                     if 'rb_gps' in self.meas_controls:
                         self.meas_controls['rb_gps'].setChecked(True)
                     
                     # Update Measurement Time String
                     # Format: MM/DD/YYYY HH:MM:SS UTC
                     import astropy.time
                     t_obj = astropy.time.Time(t0, format='gps', scale='utc')
                     # DTT format example: 06/01/1980 00:00:00 UTC
                     # strftime format: %m/%d/%Y %H:%M:%S UTC
                     t_str = t_obj.strftime('%m/%d/%Y %H:%M:%S UTC')
                     if 'meas_time_str' in self.meas_controls:
                         self.meas_controls['meas_time_str'].setText(t_str)
                         
                if duration is not None:
                     if 'time_span' in self.meas_controls:
                         self.meas_controls['time_span'].setValue(float(duration))
                         
            except Exception as e:
                print(f"Time info extraction failed: {e}")
            # --- End Time Info Update ---
            
            for g_idx in [0,1]:
                info = self.graph_info1 if g_idx==0 else self.graph_info2; p_name = list(products_dict.keys())[min(g_idx, len(products_dict)-1)]
                ctype = "Time Series"
                if p_name in ["ASD", "PSD"]: ctype = "Amplitude Spectral Density"
                elif p_name == "CSD": ctype = "Cross Spectral Density"
                elif p_name == "COH": ctype = "Coherence"
                elif p_name in ["TF", "STF"]: ctype = "Transfer Function"
                elif p_name == "Spectrogram": ctype = "Spectrogram"
                info['graph_combo'].blockSignals(True); info['graph_combo'].setCurrentText(ctype); info['graph_combo'].blockSignals(False)
                
                # For XML, we rely on the combo boxes being already populated by on_measurement_channel_changed.
                # For non-XML (File mode legacy), we populate them directly here.
                is_xml = filename.lower().endswith('.xml')
                
                # Auto-assign the first few channels to traces for visualization
                available_keys = list(products_dict[p_name].keys())[:8]
                for t_idx, item_key in enumerate(available_keys):
                    ctrl = info['traces'][t_idx]
                    
                    if not is_xml:
                        # Legacy behavior for non-XML: Populate combos directly
                        for c in ['chan_a', 'chan_b']:
                            if c in ctrl: ctrl[c].clear(); ctrl[c].addItems(cols)
                    
                    # Set selection
                    ca, cb = (str(item_key), "") if not isinstance(item_key, tuple) else (str(item_key[1]), str(item_key[0]))
                    
                    # For XML, check if these items exist in the combo (they should if populated from Meas tab)
                    if 'chan_a' in ctrl: 
                        if ctrl['chan_a'].findText(ca) == -1: ctrl['chan_a'].addItem(ca) # Fallback
                        ctrl['chan_a'].setCurrentText(ca)
                    if 'chan_b' in ctrl: 
                        if ctrl['chan_b'].findText(cb) == -1: ctrl['chan_b'].addItem(cb) # Fallback
                        ctrl['chan_b'].setCurrentText(cb)
                        
                    ctrl['active'].setChecked(True)
                
                # Trigger axis scale update based on graph type (since blockSignals was used)
                if 'range_updater' in info:
                    # Set appropriate axis scales based on graph type
                    res_tab = self.tabs.widget(3)
                    if res_tab:
                        from graph_panel import GraphPanel
                        panels = res_tab.findChildren(GraphPanel)
                        for p in panels:
                            # Trigger the graph type change handler to set correct axis scales
                            p.graph_combo.currentIndexChanged.emit(p.graph_combo.currentIndex())
                        
            self.is_loading_file = False; self.update_file_plot()
        except Exception as e:
            self.is_loading_file = False; import traceback; traceback.print_exc(); QtWidgets.QMessageBox.critical(self, "Error", f"Failed: {e}")

