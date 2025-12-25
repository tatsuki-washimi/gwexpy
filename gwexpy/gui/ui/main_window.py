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
from nds.cache import NDSDataCache

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
        self.tabs.addTab(create_excitation_tab(), "Excitation"); self.tabs.setTabEnabled(2, False)

        res_tab, self.graph_info1, self.graph_info2, self.traces1, self.traces2 = create_result_tab(on_import=self.open_file_dialog)
        self.tabs.addTab(res_tab, "Result")

        self.engine = Engine()
        self.loaded_products = {}; self.is_file_mode = False; self.is_loading_file = False
        
        # NDS Integration
        self.nds_cache = NDSDataCache()
        self.nds_cache.signal_data.connect(self.on_nds_data)
        self.nds_latest_raw = None # Stores latest DataBufferDict
        self.data_source = 'SIM' # SIM, FILE, NDS
        
        self.input_controls['ds_combo'].currentTextChanged.connect(self.on_source_changed)

        def connect_trace_combos(graph_info):
            for ctrl in graph_info['traces']:
                if 'chan_a' in ctrl: ctrl['chan_a'].currentTextChanged.connect(self.on_trace_channel_changed)
                if 'chan_b' in ctrl: ctrl['chan_b'].currentTextChanged.connect(self.on_trace_channel_changed)

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

    def on_source_changed(self, text):
        self.data_source = text
        if text == "NDS":
             self.nds_latest_raw = None

    def on_nds_data(self, buffers):
        self.nds_latest_raw = buffers

    def start_animation(self):
        self.tabs.setCurrentIndex(3); self.btn_start.setEnabled(False); self.btn_pause.setEnabled(True); self.btn_resume.setEnabled(False); self.btn_abort.setEnabled(True)
        self.is_file_mode = False 
        
        # Read Data Source directly from UI to avoid signal issues
        self.data_source = self.input_controls['ds_combo'].currentText()
        print(f"DEBUG: start_animation called. DataSource: {self.data_source}")
        
        if self.data_source == 'NDS':
            # Collect channels
            channels = []
            for info in [self.graph_info1, self.graph_info2]:
                for ctrl in info['traces']:
                    if ctrl['active'].isChecked():
                         # Force current text update if needed
                         ca = ctrl['chan_a'].currentText().strip()
                         cb = ctrl['chan_b'].currentText().strip()
                         print(f"DEBUG: Found channel input A: '{ca}', B: '{cb}'")
                         
                         # Filter out default SIM channels to avoid NDS errors
                         sim_channels = ["HF_sine", "LF_sine", "beating_sine", "white_noise", "sine_plus_noise", "square_wave", "sawtooth_wave", "random_walk"]
                         if ca and ca not in sim_channels: channels.append(ca)
                         if cb and cb not in sim_channels: channels.append(cb)
            
            # Start NDS
            print(f"DEBUG: Collected channels: {channels}")
            self.nds_cache.set_channels(channels)
            win_sec = self.input_controls['nds_win'].value()
            self.nds_cache.online_start(lookback=win_sec)
        
        self.timer.start(50)

    def pause_animation(self):
        self.timer.stop(); self.btn_start.setEnabled(False); self.btn_pause.setEnabled(False); self.btn_resume.setEnabled(True); self.btn_abort.setEnabled(True)
        if self.data_source == 'NDS': self.nds_cache.online_stop()

    def resume_animation(self):
        self.timer.start(50); self.btn_start.setEnabled(False); self.btn_pause.setEnabled(True); self.btn_resume.setEnabled(False); self.btn_abort.setEnabled(True)
        if self.data_source == 'NDS': 
             win = self.input_controls['nds_win'].value()
             self.nds_cache.online_start(lookback=win)

    def stop_animation(self):
        self.timer.stop(); self.btn_start.setEnabled(True); self.btn_pause.setEnabled(False); self.btn_resume.setEnabled(False); self.btn_abort.setEnabled(True)
        if self.data_source == 'NDS': self.nds_cache.reset(); self.nds_latest_raw = None
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
        if self.data_source == 'NDS':
             if not self.nds_latest_raw: return
             
             for ch_name, buf in self.nds_latest_raw.items():
                 y_vals = buf.data_map.get('raw')
                 if y_vals is not None and len(y_vals) > 0 and len(buf.tarray) == len(y_vals):
                     # Construct gwpy TimeSeries
                     # Use buffer's gps_start and step
                     ts = TimeSeries(y_vals, t0=buf.tarray[0], dt=buf.step, name=ch_name)
                     data_map[ch_name] = ts
             
             if not data_map: return # No valid data yet

        # SIM Logic: Generate fake data
        else:
            self.time_counter += 0.05; params = self.get_ui_params(); self.engine.configure(params)
            duration = max(10.0, 2.0/params.get('bw', 1.0))
            fs = 16384
            t0 = self.time_counter
            n = int(fs * duration)
            times = np.linspace(t0, t0 + duration, n, endpoint=False)
            
            data_arrays = [np.sin(2*np.pi*500*times), np.sin(2*np.pi*50*times), np.sin(2*np.pi*100*times)+0.5*np.sin(2*np.pi*105*times), np.random.normal(0,1,n), np.sin(2*np.pi*200*times)+np.random.normal(0,0.5,n), np.sign(np.sin(2*np.pi*20*times)), 2*(times*5%1)-1, np.cumsum(np.random.normal(0,0.1,n))]
            channel_names = ["HF_sine", "LF_sine", "beating_sine", "white_noise", "sine_plus_noise", "square_wave", "sawtooth_wave", "random_walk"]
            data_map = {name: TimeSeries(arr, t0=t0, sample_rate=fs, name=name) for name, arr in zip(channel_names, data_arrays)}
 
        for plot_idx, info_root in enumerate([self.graph_info1, self.graph_info2]):
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
            try:
                info, traces = (self.graph_info1, self.traces1) if graph_idx==0 else (self.graph_info2, self.traces2)
                g_type = info['graph_combo'].currentText(); p_name = "TS"
                if g_type == "Amplitude Spectral Density": p_name = "ASD"
                elif g_type == "Cross Spectral Density": p_name = "CSD"
                elif g_type == "Coherence": p_name = "COH"
                elif g_type == "Transfer Function": p_name = "TF"
                items = self.loaded_products.get(p_name) or (self.loaded_products.get("PSD") if p_name=="ASD" else None)
                if not items: continue
                for t_idx, ctrl in enumerate(info['traces']):
                    try:
                        if not ctrl['active'].isChecked(): traces[t_idx]['curve'].setData([], []); traces[t_idx]['curve'].setVisible(False); continue
                        ch_a, ch_b = ctrl['chan_a'].currentText(), ctrl['chan_b'].currentText()
                        key = ch_a if p_name in ["TS", "ASD", "PSD"] else (ch_b, ch_a)
                        val = items.get(key); res = normalize_series(val) if val is not None else None
                        if res:
                            x, d = res; disp = info.get('units', {}).get('display_y').currentText()
                            if disp == "dB": d = (10 if "Power" in g_type or "Squared" in g_type else 20) * np.log10(np.abs(d)+1e-20)
                            elif disp == "Phase": d = np.angle(d, deg=True) if np.iscomplexobj(d) else np.zeros_like(d)
                            elif disp == "Magnitude": d = np.abs(d)
                            traces[t_idx]['curve'].setData(x, d); traces[t_idx]['curve'].setVisible(True)
                        else: traces[t_idx]['curve'].setVisible(False)
                    except Exception as e:
                        print(f"Error updating File Plot Graph {graph_idx+1} Trace {t_idx}: {e}")
            except Exception as e:
                print(f"Error in update_file_plot for Graph {graph_idx+1}: {e}")

    def open_file_dialog(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Data File", "", "Data Files (*.xml *.gwf *.h5 *.hdf5 *.csv *.txt *.dat *.mseed *.wav);;All Files (*)")
        if f: self.open_file(f)

    def open_file(self, filename):
        try:
            self.is_loading_file = True; products_dict = loaders.load_products(filename)
            if not products_dict: QtWidgets.QMessageBox.warning(self, "Error", "No products loaded"); self.is_loading_file = False; return
            self.loaded_products = products_dict; self.is_file_mode = True; self.timer.stop()
            for tr_list in [self.traces1, self.traces2]:
                for tr in tr_list: tr['curve'].setData([], []); tr['bar'].setOpts(height=[]); tr['img'].clear()
            cols = products.extract_channels(products_dict)
            for g_idx in [0,1]:
                info = self.graph_info1 if g_idx==0 else self.graph_info2; p_name = list(products_dict.keys())[min(g_idx, len(products_dict)-1)]
                ctype = "Time Series"
                if p_name in ["ASD", "PSD"]: ctype = "Amplitude Spectral Density"
                elif p_name == "CSD": ctype = "Cross Spectral Density"
                elif p_name == "COH": ctype = "Coherence"
                elif p_name in ["TF", "STF"]: ctype = "Transfer Function"
                info['graph_combo'].blockSignals(True); info['graph_combo'].setCurrentText(ctype); info['graph_combo'].blockSignals(False)
                for t_idx, item_key in enumerate(list(products_dict[p_name].keys())[:8]):
                    ctrl = info['traces'][t_idx]
                    for c in ['chan_a', 'chan_b']:
                        if c in ctrl: ctrl[c].clear(); ctrl[c].addItems(cols)
                    ca, cb = (str(item_key), "") if not isinstance(item_key, tuple) else (str(item_key[1]), str(item_key[0]))
                    if 'chan_a' in ctrl: ctrl['chan_a'].setCurrentText(ca)
                    if 'chan_b' in ctrl: ctrl['chan_b'].setCurrentText(cb)
                    ctrl['active'].setChecked(True)
            self.is_loading_file = False; self.update_file_plot()
        except Exception as e:
            self.is_loading_file = False; import traceback; traceback.print_exc(); QtWidgets.QMessageBox.critical(self, "Error", f"Failed: {e}")
