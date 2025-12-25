from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from graph_panel import GraphPanel

def _h_spacer():
    s = QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
    return s

def _small_spin_int(min_val=-1000000000, max_val=1000000000, width=None):
    w = QtWidgets.QSpinBox()
    w.setRange(min_val, max_val)
    if width: w.setFixedWidth(width)
    return w

def _small_spin_dbl(decimals=1, width=None, min_val=-1e12, max_val=1e12, step=0.1):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(min_val, max_val)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    if width: w.setFixedWidth(width)
    w.setKeyboardTracking(False)
    return w

def _create_group(title, layout_type='grid'):
    gb = QtWidgets.QGroupBox(title)
    if layout_type == 'grid': layout = QtWidgets.QGridLayout(gb)
    elif layout_type == 'v': layout = QtWidgets.QVBoxLayout(gb)
    else: layout = QtWidgets.QHBoxLayout(gb)
    layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(6)
    return gb, layout

def create_input_tab():
    tab = QtWidgets.QWidget(); vbox = QtWidgets.QVBoxLayout(tab); vbox.setSpacing(10); vbox.setContentsMargins(10, 10, 10, 10)
    gb_ds, l_ds = _create_group("Data Source Selection", 'h')
    rb1 = QtWidgets.QRadioButton("Online system", checked=True); rb2 = QtWidgets.QRadioButton("User NDS"); rb3 = QtWidgets.QRadioButton("NDS2"); rb4 = QtWidgets.QRadioButton("LiDaX")
    [l_ds.addWidget(r) for r in [rb1, rb2, rb3, rb4]]
    l_ds.addSpacing(20); l_ds.addWidget(QtWidgets.QCheckBox("Reconnect")); l_ds.addItem(_h_spacer()); l_ds.addWidget(QtWidgets.QPushButton("Clear cache"))
    vbox.addWidget(gb_ds)

    gb_nds2, l_nds2 = _create_group("NDS2 Selection", 'grid')
    h1 = QtWidgets.QHBoxLayout(); h1.addWidget(QtWidgets.QLabel("Server:")); cb_serv2 = QtWidgets.QComboBox(); cb_serv2.addItems(["k1nds2"]); h1.addWidget(cb_serv2); h1.addWidget(QtWidgets.QLabel("Port:")); h1.addWidget(_small_spin_int(0, 65535, 80)); h1.addStretch(1); l_nds2.addLayout(h1, 0, 0, 1, 2)
    vbox.addWidget(gb_nds2)
    vbox.addStretch(1)
    return tab

def create_measurement_tab():
    tab = QtWidgets.QWidget(); outer = QtWidgets.QVBoxLayout(tab); outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(10)
    gb_meas, vb = _create_group("Measurement", 'v')
    hbox_m = QtWidgets.QHBoxLayout(); [hbox_m.addWidget(QtWidgets.QRadioButton(t)) for t in ["Fourier Tools", "Swept Sine Response", "Sine Response", "Triggered Time Response"]]; hbox_m.addStretch(1); vb.addLayout(hbox_m); outer.addWidget(gb_meas)

    gb_chan, v_chan = _create_group("Measurement Channels", 'v')
    banks = ["Channels 0 to 15", "Channels 16 to 31", "Channels 32 to 47", "Channels 48 to 63", "Channels 64 to 79", "Channels 80 to 95"]
    hbox_banks = QtWidgets.QHBoxLayout(); rb_list = []
    for i, b in enumerate(banks):
        rb = QtWidgets.QRadioButton(b); rb.setChecked(i==0); hbox_banks.addWidget(rb); rb_list.append(rb)
    v_chan.addLayout(hbox_banks)
    
    c_grid = QtWidgets.QGridLayout(); chan_grid_refs = []
    for i in range(8):
        l_label = QtWidgets.QLabel(str(i)); l_chk = QtWidgets.QCheckBox(); l_combo = QtWidgets.QComboBox()
        c_grid.addWidget(l_label, i, 0); c_grid.addWidget(l_chk, i, 1); c_grid.addWidget(l_combo, i, 2)
        r_label = QtWidgets.QLabel(str(i+8)); r_chk = QtWidgets.QCheckBox(); r_combo = QtWidgets.QComboBox()
        c_grid.addWidget(r_label, i, 3); c_grid.addWidget(r_chk, i, 4); c_grid.addWidget(r_combo, i, 5)
        chan_grid_refs.append({'l_lbl': l_label, 'r_lbl': r_label})
    v_chan.addLayout(c_grid); outer.addWidget(gb_chan)

    def update_chan_bank(idx):
        for i in range(8):
            chan_grid_refs[i]['l_lbl'].setText(str(idx*16+i)); chan_grid_refs[i]['r_lbl'].setText(str(idx*16+i+8))
    [rb.toggled.connect(lambda checked, idx=i: update_chan_bank(idx) if checked else None) for i, rb in enumerate(rb_list)]

    gb_fft, g_fft = _create_group("Fourier Tools", 'grid'); controls = {}
    def _add_p(r, c, lbl, w, unit=None, key=None):
        g_fft.addWidget(QtWidgets.QLabel(lbl), r, c); g_fft.addWidget(w, r, c+1)
        if unit: g_fft.addWidget(QtWidgets.QLabel(unit), r, c+2)
        if key: controls[key] = w
    _add_p(0, 0, "Start:", _small_spin_dbl(width=80), "Hz", 'start_freq')
    _add_p(0, 3, "Stop:", _small_spin_dbl(width=80, max_val=1e5), "Hz", 'stop_freq'); controls['stop_freq'].setValue(1000)
    _add_p(0, 6, "BW:", _small_spin_dbl(width=80), "Hz", 'bw'); controls['bw'].setValue(1)
    g_fft.addWidget(QtWidgets.QLabel("Window:"), 1, 0); cb_win = QtWidgets.QComboBox(); cb_win.addItems(["Hanning", "Flattop", "Uniform"]); controls['window'] = cb_win; g_fft.addWidget(cb_win, 1, 1, 1, 2)
    _add_p(1, 3, "Overlap:", _small_spin_dbl(width=80), "%", 'overlap'); controls['overlap'].setValue(50)
    _add_p(2, 0, "Averages:", _small_spin_int(width=80), None, 'averages'); controls['averages'].setValue(10)
    l_avg = QtWidgets.QHBoxLayout(); l_avg.addWidget(QtWidgets.QLabel("Average Type:"))
    rb_fixed = QtWidgets.QRadioButton("Fixed", checked=True); rb_exp = QtWidgets.QRadioButton("Exponential"); rb_accum = QtWidgets.QRadioButton("Accumulative")
    controls.update({'avg_type_fixed': rb_fixed, 'avg_type_exp': rb_exp, 'avg_type_accum': rb_accum})
    [l_avg.addWidget(r) for r in [rb_fixed, rb_exp, rb_accum]]; l_avg.addStretch(1); g_fft.addLayout(l_avg, 2, 3, 1, 6)
    outer.addWidget(gb_fft); outer.addStretch(1)
    return tab, controls

def create_excitation_tab():
    tab = QtWidgets.QWidget(); outer = QtWidgets.QVBoxLayout(tab); outer.setContentsMargins(10, 10, 10, 10); outer.addWidget(QtWidgets.QLabel("Excitation settings (Placeholder)")); outer.addStretch(1)
    return tab

def create_result_tab(on_import=None):
    tab = QtWidgets.QWidget(); hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    left_panel = QtWidgets.QWidget(); left_vbox = QtWidgets.QVBoxLayout(left_panel); scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    left_content = QtWidgets.QWidget(); scroll.setWidget(left_content); lv = QtWidgets.QVBoxLayout(left_content); lv.setContentsMargins(6, 6, 6, 6)
    
    right_panel = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(right_panel); rv.setContentsMargins(0,0,0,0)
    plot1 = pg.PlotWidget(title="Graph 1"); plot2 = pg.PlotWidget(title="Graph 2"); rv.addWidget(plot1); rv.addWidget(plot2)
    
    traces1 = [{'curve': plot1.plot(pen='r'), 'bar': pg.BarGraphItem(x=[0], height=[0], width=0.1, brush='r'), 'img': pg.ImageItem()} for _ in range(8)]
    traces2 = [{'curve': plot2.plot(pen='b'), 'bar': pg.BarGraphItem(x=[0], height=[0], width=0.1, brush='b'), 'img': pg.ImageItem()} for _ in range(8)]
    for t in traces1:
        plot1.addItem(t['bar']); plot1.addItem(t['img'])
        t['bar'].setVisible(False); t['img'].setVisible(False)
    for t in traces2:
        plot2.addItem(t['bar']); plot2.addItem(t['img'])
        t['bar'].setVisible(False); t['img'].setVisible(False)

    info1_panel = GraphPanel(1, plot1, traces1); lv.addWidget(info1_panel); info1 = info1_panel.to_graph_info()
    lv.addWidget(QtWidgets.QFrame()); lv.itemAt(lv.count()-1).widget().setFrameShape(QtWidgets.QFrame.HLine)
    info2_panel = GraphPanel(2, plot2, traces2); lv.addWidget(info2_panel); info2 = info2_panel.to_graph_info()
    lv.addStretch(1); left_vbox.addWidget(scroll); hsplit.addWidget(left_panel); hsplit.addWidget(right_panel); hsplit.setStretchFactor(1, 1)

    main_layout = QtWidgets.QVBoxLayout(tab); main_layout.addWidget(hsplit); bot_toolbar = QtWidgets.QHBoxLayout()
    btn_reset = QtWidgets.QPushButton("Reset"); btn_reset.clicked.connect(lambda: (plot1.autoRange(), plot2.autoRange())); bot_toolbar.addWidget(btn_reset)
    btn_import = QtWidgets.QPushButton("Import..."); bot_toolbar.addWidget(btn_import)
    if on_import: btn_import.clicked.connect(on_import)
    bot_toolbar.addStretch(1); main_layout.addLayout(bot_toolbar)
    return tab, info1, info2, traces1, traces2
