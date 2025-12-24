"""
pyaggui - A diaggui-like GUI tool for gwexpy
============================================

Refactored version with better structure, threading, and type safety.
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, dict, list, tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
import logging

logger = logging.getLogger(__name__)

# Relative imports
from .engine import Engine
from ..timeseries import TimeSeries
from ..io.dttxml_common import load_dttxml_products

# Set white background for pyqtgraph
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

def _h_spacer():
    return QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

def _small_spin_int(min_val=-1000000000, max_val=1000000000, width=None):
    w = QtWidgets.QSpinBox()
    w.setRange(min_val, max_val)
    if width:
        w.setFixedWidth(width)
    return w

def _small_spin_dbl(decimals=1, width=None, min_val=-1e12, max_val=1e12, step=0.1):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(min_val, max_val)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    if width:
        w.setFixedWidth(width)
    w.setKeyboardTracking(False)
    return w

def _create_group(title: str, layout_type='grid'):
    gb = QtWidgets.QGroupBox(title)
    if layout_type == 'grid':
        layout = QtWidgets.QGridLayout(gb)
    elif layout_type == 'v':
        layout = QtWidgets.QVBoxLayout(gb)
    else:
        layout = QtWidgets.QHBoxLayout(gb)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(6)
    return gb, layout

class ComputeWorker(QtCore.QObject):
    """
    Worker for performing intensive spectral computations in a separate thread.
    """
    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, engine: Engine, data_map: dict[str, TimeSeries], 
                 graph_type: str, active_traces: list[dict[str, Any]]):
        super().__init__()
        self.engine = engine
        self.data_map = data_map
        self.graph_type = graph_type
        self.active_traces = active_traces

    def run(self):
        try:
            results = self.engine.compute(self.data_map, self.graph_type, self.active_traces)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class InputTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setSpacing(10)
        vbox.setContentsMargins(10, 10, 10, 10)

        # -- Data Source Selection --
        gb_ds, l_ds = _create_group("Data Source Selection", 'h')
        self.rb_online = QtWidgets.QRadioButton("Online system"); self.rb_online.setChecked(True)
        l_ds.addWidget(self.rb_online)
        l_ds.addWidget(QtWidgets.QRadioButton("User NDS"))
        l_ds.addWidget(QtWidgets.QRadioButton("NDS2"))
        l_ds.addWidget(QtWidgets.QRadioButton("LiDaX"))
        l_ds.addSpacing(20)
        l_ds.addWidget(QtWidgets.QCheckBox("Reconnect"))
        l_ds.addItem(_h_spacer())
        l_ds.addWidget(QtWidgets.QPushButton("Clear cache"))
        vbox.addWidget(gb_ds)

        # -- NDS Selection --
        gb_nds, l_nds = _create_group("NDS Selection", 'h')
        l_nds.addWidget(QtWidgets.QLabel("Server:"))
        self.cb_serv = QtWidgets.QComboBox(); self.cb_serv.addItems(["k1nds1"])
        self.cb_serv.setMinimumWidth(200)
        l_nds.addWidget(self.cb_serv)
        l_nds.addWidget(QtWidgets.QLabel("Port:"))
        l_nds.addWidget(_small_spin_int(0, 65535, width=80))
        l_nds.addItem(_h_spacer())
        vbox.addWidget(gb_nds)

        # -- NDS2 Selection (Summary/Simplified) --
        gb_nds2, l_nds2 = _create_group("NDS2 Selection", 'grid')
        l_nds2.addWidget(QtWidgets.QLabel("Server:"), 0, 0)
        l_nds2.addWidget(QtWidgets.QComboBox(), 0, 1)
        l_nds2.addWidget(QtWidgets.QLabel("Port:"), 0, 2)
        l_nds2.addWidget(_small_spin_int(0, 65535, width=80), 0, 3)
        vbox.addWidget(gb_nds2)

        vbox.addStretch(1)

class MeasurementTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controls = {}
        self.setup_ui()

    def setup_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # Measurement Type
        gb_meas, vb = _create_group("Measurement", 'v')
        hbox_m = QtWidgets.QHBoxLayout()
        self.rb_fourier = QtWidgets.QRadioButton("Fourier Tools", checked=True)
        hbox_m.addWidget(self.rb_fourier)
        hbox_m.addWidget(QtWidgets.QRadioButton("Swept Sine Response"))
        hbox_m.addStretch(1)
        vb.addLayout(hbox_m)
        outer.addWidget(gb_meas)

        # Channels Bank Selection
        gb_chan, v_chan = _create_group("Measurement Channels", 'v')
        hbox_banks = QtWidgets.QHBoxLayout()
        self.rb_list = []
        for i in range(6):
            rb = QtWidgets.QRadioButton(f"Channels {i*16} to {i*16+15}")
            if i == 0: rb.setChecked(True)
            hbox_banks.addWidget(rb)
            self.rb_list.append(rb)
        v_chan.addLayout(hbox_banks)

        # Channel Grid
        self.chan_grid_refs = []
        c_grid = QtWidgets.QGridLayout()
        for i in range(8):
            l_lbl = QtWidgets.QLabel(str(i))
            l_chk = QtWidgets.QCheckBox()
            l_cmb = QtWidgets.QComboBox()
            l_cmb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            c_grid.addWidget(l_lbl, i, 0); c_grid.addWidget(l_chk, i, 1); c_grid.addWidget(l_cmb, i, 2)

            r_lbl = QtWidgets.QLabel(str(i+8))
            r_chk = QtWidgets.QCheckBox()
            r_cmb = QtWidgets.QComboBox()
            r_cmb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            c_grid.addWidget(r_lbl, i, 3); c_grid.addWidget(r_chk, i, 4); c_grid.addWidget(r_cmb, i, 5)
            
            self.chan_grid_refs.append({'l_lbl': l_lbl, 'l_chk': l_chk, 'l_cmb': l_cmb,
                                       'r_lbl': r_lbl, 'r_chk': r_chk, 'r_cmb': r_cmb})
        v_chan.addLayout(c_grid)
        outer.addWidget(gb_chan)

        # Fourier Tools Controls
        gb_fft, g_fft = _create_group("Fourier Tools", 'grid')
        self.controls['start_freq'] = _small_spin_dbl(width=80)
        self.controls['stop_freq'] = _small_spin_dbl(width=80, max_val=1e5); self.controls['stop_freq'].setValue(1000)
        self.controls['bw'] = _small_spin_dbl(width=80); self.controls['bw'].setValue(1)
        
        g_fft.addWidget(QtWidgets.QLabel("Start:"), 0, 0); g_fft.addWidget(self.controls['start_freq'], 0, 1)
        g_fft.addWidget(QtWidgets.QLabel("Stop:"), 0, 2); g_fft.addWidget(self.controls['stop_freq'], 0, 3)
        g_fft.addWidget(QtWidgets.QLabel("BW:"), 0, 4); g_fft.addWidget(self.controls['bw'], 0, 5)

        self.controls['window'] = QtWidgets.QComboBox(); self.controls['window'].addItems(["Hanning", "Flattop", "Uniform"])
        g_fft.addWidget(QtWidgets.QLabel("Window:"), 1, 0); g_fft.addWidget(self.controls['window'], 1, 1, 1, 2)
        
        self.controls['overlap'] = _small_spin_dbl(width=80); self.controls['overlap'].setValue(50)
        g_fft.addWidget(QtWidgets.QLabel("Overlap %:"), 1, 3); g_fft.addWidget(self.controls['overlap'], 1, 4)

        self.controls['averages'] = _small_spin_int(width=80); self.controls['averages'].setValue(10)
        g_fft.addWidget(QtWidgets.QLabel("Averages:"), 2, 0); g_fft.addWidget(self.controls['averages'], 2, 1)

        self.controls['avg_type_fixed'] = QtWidgets.QRadioButton("Fixed", checked=True)
        self.controls['avg_type_exp'] = QtWidgets.QRadioButton("Exponential")
        self.controls['avg_type_accum'] = QtWidgets.QRadioButton("Accumulative")
        g_fft.addWidget(self.controls['avg_type_fixed'], 2, 3); g_fft.addWidget(self.controls['avg_type_exp'], 2, 4); g_fft.addWidget(self.controls['avg_type_accum'], 2, 5)

        outer.addWidget(gb_fft)
        outer.addStretch(1)

    def get_params(self) -> dict[str, Any]:
        p = {
            'start_freq': self.controls['start_freq'].value(),
            'stop_freq': self.controls['stop_freq'].value(),
            'bw': self.controls['bw'].value(),
            'averages': self.controls['averages'].value(),
            'overlap': self.controls['overlap'].value() / 100.0,
            'window': self.controls['window'].currentText().lower(),
        }
        if self.controls['avg_type_fixed'].isChecked(): p['avg_type'] = 'fixed'
        elif self.controls['avg_type_exp'].isChecked(): p['avg_type'] = 'exponential'
        else: p['avg_type'] = 'accumulative'
        return p

class ExcitationTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(QtWidgets.QLabel("Excitation configuration (Not implemented in this demo)"))
        vbox.addStretch(1)

class ResultTab(QtWidgets.QWidget):
    def __init__(self, parent=None, on_import=None):
        super().__init__(parent)
        self.on_import_callback = on_import
        self.graph_info = [] # List of dicts for each plot
        self.plots = [] # List of pg.PlotItem
        self.traces_data = [] # List of lists of trace display objects
        self.setup_ui()

    def setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # Left Panel (Controls) Scrollable
        left_panel = QtWidgets.QWidget()
        left_vbox = QtWidgets.QVBoxLayout(left_panel); left_vbox.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget(); scroll.setWidget(scroll_content)
        lv = QtWidgets.QVBoxLayout(scroll_content)

        # Right Panel (Plots)
        right_panel = QtWidgets.QWidget()
        self.rv = QtWidgets.QVBoxLayout(right_panel); self.rv.setContentsMargins(0, 0, 0, 0)

        for i in range(2):
            plot_item = pg.PlotWidget(title=f"Plot {i+1}")
            self.plots.append(plot_item)
            self.rv.addWidget(plot_item)
            info, traces = self._create_graph_ctrl(i+1, plot_item)
            self.graph_info.append(info)
            self.traces_data.append(traces)
            if i == 0:
                line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.HLine); lv.addWidget(line)
            lv.addWidget(info['widget'])

        left_vbox.addWidget(scroll)
        hsplit.addWidget(left_panel)
        hsplit.addWidget(right_panel)
        hsplit.setStretchFactor(1, 1)
        main_layout.addWidget(hsplit)

        # Bottom Bar
        bot_toolbar = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("Reset"); self.btn_reset.clicked.connect(self.reset_plots)
        self.btn_active = QtWidgets.QPushButton("Active"); self.btn_active.clicked.connect(self.toggle_active)
        self.btn_import = QtWidgets.QPushButton("Import..."); self.btn_import.clicked.connect(self._import_clicked)
        
        bot_toolbar.addWidget(self.btn_reset); bot_toolbar.addWidget(self.btn_active); bot_toolbar.addWidget(self.btn_import)
        bot_toolbar.addStretch(1)
        main_layout.addLayout(bot_toolbar)

    def _import_clicked(self):
        if self.on_import_callback:
            self.on_import_callback()

    def reset_plots(self):
        for p in self.plots: p.autoRange()

    def toggle_active(self):
        any_unchecked = any(not t['active'].isChecked() for g in self.graph_info for t in g['traces'])
        target = True if any_unchecked else False
        for g in self.graph_info:
            for t in g['traces']: t['active'].setChecked(target)

    def _create_graph_ctrl(self, idx, plot_widget):
        container = QtWidgets.QWidget()
        path_frame = container
        lv = QtWidgets.QVBoxLayout(container); lv.setContentsMargins(2, 2, 2, 2)
        
        # Path/Title
        hl_path = QtWidgets.QHBoxLayout(); hl_path.setContentsMargins(0,0,0,0)
        hl_path.addWidget(QtWidgets.QLabel(f"Plot {idx}:"))
        graph_combo = QtWidgets.QComboBox()
        graph_combo.addItems(["Time Series", "Amplitude Spectral Density", "Power Spectral Density", "Coherence", "Squared Coherence", "Transfer Function", "Cross Spectral Density", "Spectrogram"])
        hl_path.addWidget(graph_combo, 1)
        lv.addLayout(hl_path)

        # Tabs for graph settings
        tab_row1 = QtWidgets.QTabBar(); tab_row1.setExpanding(False); [tab_row1.addTab(t) for t in ["Style", "X-axis", "Y-axis", "Legend", "Param"]]
        tab_row2 = QtWidgets.QTabBar(); tab_row2.setExpanding(False); [tab_row2.addTab(t) for t in ["Traces", "Range", "Units", "Cursor", "Config"]]
        lv.addWidget(tab_row1); lv.addWidget(tab_row2)

        main_stack = QtWidgets.QStackedWidget()
        lv.addWidget(main_stack)

        # Traces Tab
        traces_tab = QtWidgets.QWidget(); tl = QtWidgets.QVBoxLayout(traces_tab); tl.setContentsMargins(4,4,4,4)
        trace_controls = []; traces_display = []
        colors = [('white','white'), ('yellow','yellow'), ('cyan','cyan'), ('magenta','magenta'), ('red','red'), ('green','green'), ('blue','blue'), ('darkGray','gray')]
        
        for i in range(8):
            row = QtWidgets.QHBoxLayout()
            chk = QtWidgets.QCheckBox(str(i))
            col_box = QtWidgets.QComboBox(); col_box.setFixedWidth(25); col_box.addItem(""); col_box.setItemData(0, QtGui.QColor(colors[i][1]), QtCore.Qt.BackgroundRole)
            ch1 = QtWidgets.QComboBox(); ch2 = QtWidgets.QComboBox()
            ch1.setMinimumWidth(120); ch2.setMinimumWidth(120)
            [row.addWidget(w) for w in [chk, col_box, ch1, ch2]]
            tl.addLayout(row)
            
            # Display components
            curve = pg.PlotDataItem(pen=pg.mkPen(colors[i][1], width=1))
            plot_widget.addItem(curve)
            bar = pg.BarGraphItem(x=[0], height=[0], width=1, brush=colors[i][1])
            plot_widget.addItem(bar); bar.setVisible(False)
            img = pg.ImageItem(); plot_widget.addItem(img); img.setVisible(False)
            
            trace_controls.append({'active': chk, 'color': col_box, 'chan_a': ch1, 'chan_b': ch2})
            traces_display.append({'curve': curve, 'bar': bar, 'img': img})

        main_stack.addWidget(traces_tab) # 0: Traces
        # Simplification: only Traces tab for now in refactor, adding others as needed
        
        def row1_changed(i): tab_row2.setCurrentIndex(-1); main_stack.setCurrentIndex(i + 5)
        def row2_changed(i): 
            if i >= 0: tab_row1.setCurrentIndex(-1); main_stack.setCurrentIndex(i)

        tab_row1.currentChanged.connect(row1_changed)
        tab_row2.currentChanged.connect(row2_changed)

        info = {'widget': container, 'graph_combo': graph_combo, 'traces': trace_controls}
        return info, traces_display

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyaggui : refactored gwexpy GUI")
        self.resize(1100, 850)

        self.engine = Engine()
        self.thread_pool = QtCore.QThreadPool()
        self.is_computing = False
        self.is_file_mode = False
        self.is_loading_file = False
        self.loaded_products = {}
        self.time_counter = 0.0

        self.setup_ui()
        self.setup_timer()

    def setup_ui(self):
        # Menu
        mb = self.menuBar()
        file_menu = mb.addMenu("File")
        open_action = file_menu.addAction("Open...")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction("Exit").triggered.connect(self.close)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tab_input = InputTab()
        self.tab_meas = MeasurementTab()
        self.tab_excit = ExcitationTab()
        self.tab_result = ResultTab(on_import=self.open_file_dialog)

        self.tabs.addTab(self.tab_input, "Input")
        self.tabs.addTab(self.tab_meas, "Measurement")
        self.tabs.addTab(self.tab_excit, "Excitation")
        self.tabs.addTab(self.tab_result, "Result")

        # Bottom Buttons
        bottom_widget = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bottom_widget)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_pause = QtWidgets.QPushButton("Pause"); self.btn_pause.setEnabled(False)
        self.btn_resume = QtWidgets.QPushButton("Resume"); self.btn_resume.setEnabled(False)
        self.btn_abort = QtWidgets.QPushButton("Abort")
        
        for b in [self.btn_start, self.btn_pause, self.btn_resume, self.btn_abort]:
            b.setMinimumWidth(100)
            bl.addWidget(b)
            if b != self.btn_abort: bl.addStretch(1)

        self.btn_start.clicked.connect(self.start_animation)
        self.btn_pause.clicked.connect(self.pause_animation)
        self.btn_resume.clicked.connect(self.resume_animation)
        self.btn_abort.clicked.connect(self.stop_animation)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        cv = QtWidgets.QVBoxLayout(central)
        cv.addWidget(self.tabs)
        cv.addWidget(bottom_widget)

        # Connect Trace Channel Combos to Update Logic (for File Mode)
        for g_info in self.tab_result.graph_info:
            for ctrl in g_info['traces']:
                ctrl['chan_a'].currentTextChanged.connect(self.on_trace_channel_changed)
                ctrl['chan_b'].currentTextChanged.connect(self.on_trace_channel_changed)

    def setup_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graphs)

    def start_animation(self):
        self.tabs.setCurrentIndex(3) # Switch to Result
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.is_file_mode = False
        self.timer.start(100) # 100ms interval

    def pause_animation(self):
        self.timer.stop()
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)

    def resume_animation(self):
        self.timer.start(100)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)

    def stop_animation(self):
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.time_counter = 0.0
        # Clear plots
        for g_traces in self.tab_result.traces_data:
            for t in g_traces:
                t['curve'].setData([], [])
                t['img'].clear()

    def update_graphs(self):
        if self.is_file_mode or self.is_computing:
            return

        self.time_counter += 0.1
        params = self.tab_meas.get_params()
        self.engine.configure(params)

        # Simulate data
        fs = 16384
        duration = max(10.0, 2.0 / params.get('bw', 1.0))
        n_samples = int(fs * duration)
        t0 = self.time_counter
        times = np.linspace(t0, t0 + duration, n_samples, endpoint=False)
        
        channel_names = ["HF_sine", "LF_sine", "beating_sine", "white_noise", "sine_plus_noise", "square_wave", "sawtooth_wave", "random_walk"]
        data_map = {}
        # Simple simulation formulas
        data_map["HF_sine"] = TimeSeries(np.sin(2 * np.pi * 500 * times), t0=t0, sample_rate=fs, name="HF_sine")
        data_map["LF_sine"] = TimeSeries(np.sin(2 * np.pi * 50 * times), t0=t0, sample_rate=fs, name="LF_sine")
        data_map["beating_sine"] = TimeSeries(np.sin(2 * np.pi * 100 * times) + 0.5 * np.sin(2 * np.pi * 105 * times), t0=t0, sample_rate=fs)
        data_map["white_noise"] = TimeSeries(np.random.normal(0, 1, n_samples), t0=t0, sample_rate=fs)

        # Process each plot widget
        for plot_idx in range(len(self.tab_result.plots)):
            info = self.tab_result.graph_info[plot_idx]
            g_type = info['graph_combo'].currentText()
            
            trace_reqs = []
            for ctrl in info['traces']:
                trace_reqs.append({
                    'active': ctrl['active'].isChecked(),
                    'ch_a': ctrl['chan_a'].currentText(),
                    'ch_b': ctrl['chan_b'].currentText()
                })

            # Start Worker
            self.is_computing = True
            worker = ComputeWorker(self.engine, data_map, g_type, trace_reqs)
            worker.finished.connect(lambda results, idx=plot_idx: self.handle_results(results, idx))
            worker.error.connect(lambda err: logger.error(f"Calculation Error: {err}"))
            
            # Using QThread manually for simplicity here, or just wrap in QRunnable
            # For brevity in this refactor, let's use a simpler threading approach
            thread = QtCore.QThread()
            worker.moveToThread(thread)
            worker.finished.connect(thread.quit)
            worker.finished.connect(lambda: setattr(self, 'is_computing', False))
            thread.started.connect(worker.run)
            thread.start()
            self._current_threads = getattr(self, '_current_threads', []) + [thread]

    def handle_results(self, results, plot_idx):
        display_traces = self.tab_result.traces_data[plot_idx]
        for t_idx, result in enumerate(results):
            if t_idx >= len(display_traces): break
            trace = display_traces[t_idx]
            if result is None:
                trace['curve'].setData([], [])
                continue
            
            if isinstance(result, dict) and result.get('type') == 'spectrogram':
                # Handle spectrogram...
                trace['img'].setImage(result['value'], autoLevels=True)
                trace['img'].setVisible(True)
                trace['curve'].setData([], [])
            else:
                x_vals, y_vals = result
                # Handle dB/etc if needed...
                trace['curve'].setData(x_vals, np.abs(y_vals))
                trace['img'].setVisible(False)

    def on_trace_channel_changed(self):
        if not self.is_loading_file and self.is_file_mode:
            self.update_file_plot()

    def open_file_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Data File", "", "Data Files (*.xml *.gwf *.h5)")
        if filename: self.open_file(filename)

    def open_file(self, filename):
        try:
            self.is_loading_file = True
            ext = Path(filename).suffix.lower()
            if ext == '.xml':
                self.loaded_products = load_dttxml_products(filename)
                self.is_file_mode = True
                self.timer.stop()
                self.populate_channels()
                self.update_file_plot()
            self.is_loading_file = False
        except Exception as e:
            self.is_loading_file = False
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def populate_channels(self):
        all_ch = set()
        for prod in self.loaded_products.values():
            for k in prod.keys():
                if isinstance(k, tuple): [all_ch.add(str(x)) for x in k]
                else: all_ch.add(str(k))
        sorted_ch = sorted(list(all_ch))
        for g_info in self.tab_result.graph_info:
            for ctrl in g_info['traces']:
                ctrl['chan_a'].clear(); ctrl['chan_a'].addItems(sorted_ch)
                ctrl['chan_b'].clear(); ctrl['chan_b'].addItems(sorted_ch)

    def update_file_plot(self):
        # Simplified update logic for refactor
        pass

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
