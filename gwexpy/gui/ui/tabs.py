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

    # -- Data Source Selection --
    gb_ds, l_ds = _create_group("Data Source Selection", 'h')
    
    # Hidden combo for compatibility with MainWindow logic
    ds_combo = QtWidgets.QComboBox(); ds_combo.setVisible(False); ds_combo.addItems(["SIM", "NDS", "FILE"])
    
    rb1 = QtWidgets.QRadioButton("Online system"); rb1.setChecked(True) # Maps to NDS
    rb2 = QtWidgets.QRadioButton("Simulation") # Added for SIM support
    rb3 = QtWidgets.QRadioButton("User NDS")
    rb4 = QtWidgets.QRadioButton("LiDaX")
    
    # Logic to update hidden combo
    def update_ds():
        if rb2.isChecked(): ds_combo.setCurrentText("SIM")
        elif rb1.isChecked() or rb3.isChecked(): ds_combo.setCurrentText("NDS")
        # else: ds_combo.setCurrentText("FILE") # simplified
    
    for r in [rb1, rb2, rb3, rb4]:
        r.toggled.connect(update_ds)
        l_ds.addWidget(r)
        
    # Trigger initial update
    update_ds()

    l_ds.addSpacing(20)
    l_ds.addWidget(QtWidgets.QCheckBox("Reconnect"))
    l_ds.addItem(_h_spacer())
    l_ds.addWidget(QtWidgets.QPushButton("Clear cache"))
    
    vbox.addWidget(gb_ds)

    # -- NDS Selection (for Online system) --
    gb_nds, l_nds = _create_group("NDS Selection", 'h')
    l_nds.addWidget(QtWidgets.QLabel("Server:"))
    cb_serv = QtWidgets.QComboBox(); cb_serv.addItems(["k1nds1"]); cb_serv.setMinimumWidth(200); cb_serv.setEditable(True)
    l_nds.addWidget(cb_serv)
    l_nds.addWidget(QtWidgets.QLabel("Port:"))
    l_nds.addWidget(_small_spin_int(0, 65535, width=80)) # Default 8088
    l_nds.addItem(_h_spacer())
    
    # Add simple NDS Window control here for functionality, even if not in original looks (it's essential for online)
    l_nds.addWidget(QtWidgets.QLabel("Window(s):"))
    nds_win_spin = _small_spin_int(min_val=10, max_val=3600, width=60); nds_win_spin.setValue(30)
    l_nds.addWidget(nds_win_spin)
    
    vbox.addWidget(gb_nds)

    # -- NDS2 Selection --
    gb_nds2, l_nds2 = _create_group("NDS2 Selection", 'grid')
    h1 = QtWidgets.QHBoxLayout(); h1.addWidget(QtWidgets.QLabel("Server:")); cb_serv2 = QtWidgets.QComboBox(); cb_serv2.addItems(["k1nds2"]); h1.addWidget(cb_serv2); h1.addWidget(QtWidgets.QLabel("Port:")); h1.addWidget(_small_spin_int(0, 65535, 80)); h1.addStretch(1); l_nds2.addLayout(h1, 0, 0, 1, 2)
    vbox.addWidget(gb_nds2)
    
    # -- Simulation Settings (Hidden or integrated?) -- 
    # To keep "looks", we put this in a separate small box or re-use LiDaX?
    # Let's add a small "Simulation Settings" group at bottom
    gb_sim, l_sim = _create_group("Simulation Settings", 'h')
    l_sim.addWidget(QtWidgets.QLabel("Duration (s):"))
    sim_dur_spin = _small_spin_dbl(1, 60, min_val=0.1, max_val=1e6); sim_dur_spin.setValue(10.0)
    l_sim.addWidget(sim_dur_spin)
    l_sim.addStretch(1)
    vbox.addWidget(gb_sim)
    
    vbox.addStretch(1)
    
    controls = {'ds_combo': ds_combo, 'sim_dur': sim_dur_spin, 'nds_win': nds_win_spin}
    return tab, controls

def create_measurement_tab():
    tab = QtWidgets.QWidget()
    outer = QtWidgets.QVBoxLayout(tab)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(10)

    # Group: Measurement
    gb_meas, vb = _create_group("Measurement", 'v')
    hbox_m = QtWidgets.QHBoxLayout()
    hbox_m.addWidget(QtWidgets.QRadioButton("Fourier Tools", checked=True))
    hbox_m.addWidget(QtWidgets.QRadioButton("Swept Sine Response"))
    hbox_m.addWidget(QtWidgets.QRadioButton("Sine Response"))
    hbox_m.addWidget(QtWidgets.QRadioButton("Triggered Time Response"))
    hbox_m.addStretch(1)
    vb.addLayout(hbox_m)
    outer.addWidget(gb_meas)

    # Group: Measurement Channels
    gb_chan, v_chan = _create_group("Measurement Channels", 'v')
    
    # Radio buttons for channel banks
    hbox_banks = QtWidgets.QHBoxLayout()
    banks = ["Channels 0 to 15", "Channels 16 to 31", "Channels 32 to 47", 
             "Channels 48 to 63", "Channels 64 to 79", "Channels 80 to 95"]
    rb_list = []
    for i, b in enumerate(banks):
        rb = QtWidgets.QRadioButton(b)
        if i==0: rb.setChecked(True)
        hbox_banks.addWidget(rb)
        rb_list.append(rb)
    hbox_banks.addStretch(1)
    v_chan.addLayout(hbox_banks)
    
    # Grid of channels
    c_grid = QtWidgets.QGridLayout()
    c_grid.setContentsMargins(0,0,0,0)
    c_grid.setHorizontalSpacing(15)
    
    chan_grid_refs = []
    for i in range(8):
        # Left column (0-7)
        l_lbl = QtWidgets.QLabel(str(i)); c_grid.addWidget(l_lbl, i, 0)
        c_grid.addWidget(QtWidgets.QCheckBox(), i, 1)
        combo = QtWidgets.QComboBox(); combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        c_grid.addWidget(combo, i, 2)
        
        # Right column (8-15)
        r_lbl = QtWidgets.QLabel(str(i+8)); c_grid.addWidget(r_lbl, i, 3)
        c_grid.addWidget(QtWidgets.QCheckBox(), i, 4)
        combo2 = QtWidgets.QComboBox(); combo2.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        c_grid.addWidget(combo2, i, 5)
        chan_grid_refs.append((l_lbl, r_lbl))
        
    v_chan.addLayout(c_grid)
    outer.addWidget(gb_chan)
    
    def update_chan_bank(idx):
        for i in range(8):
            chan_grid_refs[i][0].setText(str(idx*16+i))
            chan_grid_refs[i][1].setText(str(idx*16+i+8))
    [rb.toggled.connect(lambda checked, idx=i: update_chan_bank(idx) if checked else None) for i, rb in enumerate(rb_list)]

    # Group: Fourier Tools
    gb_fft, g_fft = _create_group("Fourier Tools", 'grid')
    controls = {}
    
    def _add_param(row, col, label, spin_widget, unit=None, key=None):
        g_fft.addWidget(QtWidgets.QLabel(label), row, col)
        g_fft.addWidget(spin_widget, row, col+1)
        if unit:
            g_fft.addWidget(QtWidgets.QLabel(unit), row, col+2)
        if key: controls[key] = spin_widget
        return col + 3

    # Row 0
    c = 0
    c = _add_param(0, c, "Start:", _small_spin_dbl(width=80), "Hz", 'start_freq')
    c = _add_param(0, c, "Stop:", _small_spin_dbl(width=80, max_val=1e5), "Hz", 'stop_freq'); controls['stop_freq'].setValue(1000)
    c = _add_param(0, c, "BW:", _small_spin_dbl(width=80), "Hz", 'bw'); controls['bw'].setValue(1)
    c = _add_param(0, c, "TimeSpan:", _small_spin_dbl(width=80), "s")
    c = _add_param(0, c, "Settling Time:", _small_spin_dbl(width=80), "%")
    c = _add_param(0, c, "Ramp Down:", _small_spin_dbl(width=80), "Sec")
    c = _add_param(0, c, "Ramp Up:", _small_spin_dbl(width=80), "")

    # Row 1
    g_fft.addWidget(QtWidgets.QLabel("Window:"), 1, 0)
    cb_win = QtWidgets.QComboBox(); cb_win.addItems(["Hanning", "Flattop", "Uniform"]); controls['window'] = cb_win
    g_fft.addWidget(cb_win, 1, 1, 1, 2)
    
    g_fft.addWidget(QtWidgets.QLabel("Overlap:"), 1, 3)
    sb_ov = _small_spin_dbl(width=80); sb_ov.setValue(50); controls['overlap'] = sb_ov
    g_fft.addWidget(sb_ov, 1, 4)
    g_fft.addWidget(QtWidgets.QLabel("%"), 1, 5)
    
    g_fft.addWidget(QtWidgets.QCheckBox("Remove mean", checked=True), 1, 6, 1, 2)
    
    g_fft.addWidget(QtWidgets.QLabel("Number of A channels:"), 1, 9)
    g_fft.addWidget(_small_spin_int(width=60), 1, 10)

    # Row 2
    g_fft.addWidget(QtWidgets.QLabel("Averages:"), 2, 0)
    sb_avg = _small_spin_int(width=80); sb_avg.setValue(10); controls['averages'] = sb_avg
    g_fft.addWidget(sb_avg, 2, 1)
    
    l_avg = QtWidgets.QHBoxLayout()
    l_avg.addWidget(QtWidgets.QLabel("Average Type:"))
    rb_fixed = QtWidgets.QRadioButton("Fixed", checked=True); rb_exp = QtWidgets.QRadioButton("Exponential"); rb_accum = QtWidgets.QRadioButton("Accumulative")
    controls.update({'avg_type_fixed': rb_fixed, 'avg_type_exp': rb_exp, 'avg_type_accum': rb_accum})
    l_avg.addWidget(rb_fixed); l_avg.addWidget(rb_exp); l_avg.addWidget(rb_accum)
    l_avg.addStretch(1)
    g_fft.addLayout(l_avg, 2, 3, 1, 6)
    
    g_fft.addWidget(QtWidgets.QLabel("Burst Noise Quiet Time"), 2, 12)
    g_fft.addWidget(_small_spin_dbl(decimals=2, width=80), 2, 13)
    g_fft.addWidget(QtWidgets.QLabel("sec"), 2, 14)

    outer.addWidget(gb_fft)

    # Group: Start Time (Visual Only for now)
    gb_time, g_time = _create_group("Start Time", 'grid')
    rb_now = QtWidgets.QRadioButton("Now", checked=True); g_time.addWidget(rb_now, 0, 0)
    rb_gps = QtWidgets.QRadioButton("GPS:"); g_time.addWidget(rb_gps, 1, 0)
    g_time.addWidget(_small_spin_int(width=100, min_val=0), 1, 1); g_time.addWidget(QtWidgets.QLabel("sec"), 1, 2)
    g_time.addWidget(_small_spin_int(width=80, min_val=0), 1, 3); g_time.addWidget(QtWidgets.QLabel("nsec"), 1, 4)
    rb_dt = QtWidgets.QRadioButton("Date/time:"); g_time.addWidget(rb_dt, 2, 0)
    g_time.addWidget(QtWidgets.QDateEdit(QtCore.QDate.currentDate()), 2, 1); g_time.addWidget(QtWidgets.QTimeEdit(QtCore.QTime.currentTime()), 2, 2); g_time.addWidget(QtWidgets.QLabel("UTC"), 2, 3)
    g_time.addWidget(QtWidgets.QRadioButton("In the future:"), 0, 6); g_time.addWidget(QtWidgets.QTimeEdit(), 0, 7); g_time.addWidget(QtWidgets.QLabel("hh:mm:ss"), 0, 8)
    g_time.addWidget(QtWidgets.QRadioButton("In the past:"), 1, 6); g_time.addWidget(QtWidgets.QTimeEdit(), 1, 7); g_time.addWidget(QtWidgets.QLabel("hh:mm:ss"), 1, 8)
    g_time.addWidget(QtWidgets.QPushButton("Time now"), 2, 6); g_time.addWidget(QtWidgets.QPushButton("Lookup..."), 2, 7)
    l_slow = QtWidgets.QHBoxLayout(); l_slow.addStretch(1); l_slow.addWidget(QtWidgets.QLabel("Slow down:")); l_slow.addWidget(_small_spin_int(width=60, min_val=0)); l_slow.addWidget(QtWidgets.QLabel("sec/avrg.")); g_time.addLayout(l_slow, 2, 9, 1, 4)
    outer.addWidget(gb_time)

    # Group: Measurement Information
    gb_info, g_info = _create_group("Measurement Information", 'grid')
    g_info.addWidget(QtWidgets.QLabel("Measurement Time:"), 0, 0)
    g_info.addWidget(QtWidgets.QLineEdit("06/01/1980 00:00:00 UTC"), 0, 1)
    g_info.addWidget(QtWidgets.QLabel("Comment / Description:"), 0, 2)
    outer.addWidget(gb_info)
    txt_comment = QtWidgets.QLineEdit(); txt_comment.setMinimumHeight(30)
    g_info.addWidget(txt_comment, 1, 0, 1, 3)

    outer.addStretch(1)
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
