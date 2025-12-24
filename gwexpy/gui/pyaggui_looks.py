
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np

# Set white background for pyqtgraph to match the screenshot
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


def _h_spacer():
    s = QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
    return s


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
    return w


def _create_group(title, layout_type='grid'):
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


# =================================================================================
# INPUT TAB
# =================================================================================
def create_input_tab():
    tab = QtWidgets.QWidget()
    vbox = QtWidgets.QVBoxLayout(tab)
    vbox.setSpacing(10)
    vbox.setContentsMargins(10, 10, 10, 10)

    # -- Data Source Selection --
    gb_ds, l_ds = _create_group("Data Source Selection", 'h')
    
    rb1 = QtWidgets.QRadioButton("Online system"); rb1.setChecked(True)
    rb2 = QtWidgets.QRadioButton("User NDS")
    rb3 = QtWidgets.QRadioButton("NDS2")
    rb4 = QtWidgets.QRadioButton("LiDaX")
    chk_rec = QtWidgets.QCheckBox("Reconnect")
    btn_clear = QtWidgets.QPushButton("Clear cache")

    l_ds.addWidget(rb1)
    l_ds.addWidget(rb2)
    l_ds.addWidget(rb3)
    l_ds.addWidget(rb4)
    l_ds.addSpacing(20)
    l_ds.addWidget(chk_rec)
    l_ds.addItem(_h_spacer())
    l_ds.addWidget(btn_clear)
    
    vbox.addWidget(gb_ds)

    # -- NDS Selection --
    gb_nds, l_nds = _create_group("NDS Selection", 'h')
    l_nds.addWidget(QtWidgets.QLabel("Server:"))
    cb_serv = QtWidgets.QComboBox(); cb_serv.addItems(["k1nds1"])
    cb_serv.setMinimumWidth(200)
    l_nds.addWidget(cb_serv)
    l_nds.addWidget(QtWidgets.QLabel("Port:"))
    l_nds.addWidget(_small_spin_int(0, 65535, width=80)) # Default 8088
    l_nds.addItem(_h_spacer())
    vbox.addWidget(gb_nds)

    # -- NDS2 Selection --
    gb_nds2, l_nds2 = _create_group("NDS2 Selection", 'grid')
    
    # Row 1
    h1 = QtWidgets.QHBoxLayout()
    h1.addWidget(QtWidgets.QLabel("Server:"))
    cb_serv2 = QtWidgets.QComboBox(); cb_serv2.addItems(["k1nds2"])
    cb_serv2.setMinimumWidth(200)
    h1.addWidget(cb_serv2)
    h1.addWidget(QtWidgets.QLabel("Port:"))
    sb_port2 = _small_spin_int(0, 65535, width=80); sb_port2.setValue(31200)
    h1.addWidget(sb_port2)
    h1.addWidget(QtWidgets.QLabel("Epoch:"))
    cb_epoch = QtWidgets.QComboBox(); cb_epoch.addItems(["User specified"])
    h1.addWidget(cb_epoch)
    h1.addStretch(1)
    l_nds2.addLayout(h1, 0, 0, 1, 2)

    # Epoch Start/Stop Frames
    gb_start = QtWidgets.QGroupBox("Epoch Start")
    ls = QtWidgets.QGridLayout(gb_start)
    ls.addWidget(QtWidgets.QLabel("GPS:"), 0, 0)
    ls.addWidget(_small_spin_int(width=120), 0, 1)
    ls.addWidget(QtWidgets.QLabel("sec"), 0, 2)
    
    ls.addWidget(QtWidgets.QLabel("Date/Time:"), 1, 0)
    de = QtWidgets.QDateEdit(); de.setDate(QtCore.QDate(1980, 1, 6))
    te = QtWidgets.QTimeEdit()
    ls.addWidget(de, 1, 1)
    ls.addWidget(te, 1, 2)
    ls.addWidget(QtWidgets.QLabel("hh:mm:ss UTC"), 1, 3)

    gb_stop = QtWidgets.QGroupBox("Epoch Stop")
    le = QtWidgets.QGridLayout(gb_stop)
    le.addWidget(QtWidgets.QLabel("GPS:"), 0, 0)
    sb_gps_stop = _small_spin_int(width=120); sb_gps_stop.setValue(1450499670)
    le.addWidget(sb_gps_stop, 0, 1)
    le.addWidget(QtWidgets.QLabel("sec"), 0, 2)
    
    le.addWidget(QtWidgets.QLabel("Date/Time:"), 1, 0)
    de2 = QtWidgets.QDateEdit(); de2.setDate(QtCore.QDate(2025, 12, 23))
    te2 = QtWidgets.QTimeEdit(); te2.setTime(QtCore.QTime(4, 34, 12))
    le.addWidget(de2, 1, 1)
    le.addWidget(te2, 1, 2)
    le.addWidget(QtWidgets.QLabel("hh:mm:ss UTC"), 1, 3)

    l_nds2.addWidget(gb_start, 1, 0)
    l_nds2.addWidget(gb_stop, 1, 1)
    
    vbox.addWidget(gb_nds2)

    # -- LiDaX Data Source --
    gb_lidax, l_lidax = _create_group("LiDaX Data Source", 'grid')
    
    l_lidax.addWidget(QtWidgets.QLabel("Server:"), 0, 0)
    cb_sing = QtWidgets.QComboBox(); cb_sing.addItems(["single"])
    l_lidax.addWidget(cb_sing, 0, 1)
    cb_local = QtWidgets.QComboBox(); cb_local.addItems(["Local file system"])
    cb_local.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    l_lidax.addWidget(cb_local, 0, 2)
    
    btn_add = QtWidgets.QPushButton("Add...")
    l_lidax.addWidget(btn_add, 0, 3)

    l_lidax.addWidget(QtWidgets.QLabel("Channels:"), 0, 4)
    l_lidax.addWidget(QtWidgets.QLineEdit(), 0, 5)
    l_lidax.addWidget(QtWidgets.QPushButton("Select..."), 0, 6)
    
    l_lidax.addWidget(QtWidgets.QLabel("UDN:"), 1, 0)
    cb_udn = QtWidgets.QComboBox()
    cb_udn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    l_lidax.addWidget(cb_udn, 1, 1, 1, 2)
    
    btn_more = QtWidgets.QPushButton("More..."); btn_more.setEnabled(False)
    l_lidax.addWidget(btn_more, 1, 3)

    l_lidax.addWidget(QtWidgets.QLabel("Keep:"), 1, 4)
    sb_keep = _small_spin_dbl(width=80); sb_keep.setSuffix("   hh:mm"); sb_keep.setValue(0.30)
    l_lidax.addWidget(sb_keep, 1, 5)
    
    btn_stage = QtWidgets.QPushButton("Staging..."); btn_stage.setEnabled(False)
    l_lidax.addWidget(btn_stage, 1, 6)

    vbox.addWidget(gb_lidax)

    vbox.addStretch(1)
    return tab


# =================================================================================
# MEASUREMENT TAB
# =================================================================================
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
    for i, b in enumerate(banks):
        rb = QtWidgets.QRadioButton(b)
        if i==0: rb.setChecked(True)
        hbox_banks.addWidget(rb)
    hbox_banks.addStretch(1)
    v_chan.addLayout(hbox_banks)
    
    # Grid of channels
    c_grid = QtWidgets.QGridLayout()
    c_grid.setContentsMargins(0,0,0,0)
    c_grid.setHorizontalSpacing(15)
    
    for i in range(8):
        # Left column (0-7)
        c_grid.addWidget(QtWidgets.QLabel(str(i)), i, 0)
        c_grid.addWidget(QtWidgets.QCheckBox(), i, 1)
        combo = QtWidgets.QComboBox()
        combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        c_grid.addWidget(combo, i, 2)
        
        # Right column (8-15)
        c_grid.addWidget(QtWidgets.QLabel(str(i+8)), i, 3)
        c_grid.addWidget(QtWidgets.QCheckBox(), i, 4)
        combo2 = QtWidgets.QComboBox()
        combo2.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        c_grid.addWidget(combo2, i, 5)
        
    v_chan.addLayout(c_grid)
    outer.addWidget(gb_chan)

    # Group: Fourier Tools
    gb_fft, g_fft = _create_group("Fourier Tools", 'grid')
    
    def _add_param(row, col, label, spin_widget, unit=None):
        g_fft.addWidget(QtWidgets.QLabel(label), row, col)
        g_fft.addWidget(spin_widget, row, col+1)
        if unit:
            g_fft.addWidget(QtWidgets.QLabel(unit), row, col+2)
        return col + 3

    # Row 0
    c = 0
    c = _add_param(0, c, "Start:", _small_spin_dbl(width=80), "Hz")
    c = _add_param(0, c, "Stop:", _small_spin_dbl(width=80), "Hz")
    c = _add_param(0, c, "BW:", _small_spin_dbl(width=80), "Hz")
    c = _add_param(0, c, "TimeSpan:", _small_spin_dbl(width=80), "s")
    c = _add_param(0, c, "Settling Time:", _small_spin_dbl(width=80), "%")
    c = _add_param(0, c, "Ramp Down:", _small_spin_dbl(width=80), "Sec")
    c = _add_param(0, c, "Ramp Up:", _small_spin_dbl(width=80), "")

    # Row 1
    g_fft.addWidget(QtWidgets.QLabel("Window:"), 1, 0)
    cb_win = QtWidgets.QComboBox(); cb_win.addItems(["Hanning"])
    g_fft.addWidget(cb_win, 1, 1, 1, 2)
    
    g_fft.addWidget(QtWidgets.QLabel("Overlap:"), 1, 3)
    g_fft.addWidget(_small_spin_dbl(width=80), 1, 4)
    g_fft.addWidget(QtWidgets.QLabel("%"), 1, 5)
    
    g_fft.addWidget(QtWidgets.QCheckBox("Remove mean", checked=True), 1, 6, 1, 2)
    
    g_fft.addWidget(QtWidgets.QLabel("Number of A channels:"), 1, 9)
    g_fft.addWidget(_small_spin_int(width=60), 1, 10)

    # Row 2
    g_fft.addWidget(QtWidgets.QLabel("Averages:"), 2, 0)
    sb_avg = _small_spin_int(width=80); sb_avg.setValue(10)
    g_fft.addWidget(sb_avg, 2, 1)
    
    l_avg = QtWidgets.QHBoxLayout()
    l_avg.addWidget(QtWidgets.QLabel("Average Type:"))
    l_avg.addWidget(QtWidgets.QRadioButton("Fixed", checked=True))
    l_avg.addWidget(QtWidgets.QRadioButton("Exponential"))
    l_avg.addWidget(QtWidgets.QRadioButton("Accumulative"))
    l_avg.addStretch(1)
    g_fft.addLayout(l_avg, 2, 3, 1, 6)
    
    g_fft.addWidget(QtWidgets.QLabel("Burst Noise Quiet Time"), 2, 12)
    g_fft.addWidget(_small_spin_dbl(decimals=2, width=80), 2, 13)
    g_fft.addWidget(QtWidgets.QLabel("sec"), 2, 14)

    outer.addWidget(gb_fft)

    # Group: Start Time
    gb_time, g_time = _create_group("Start Time", 'grid')
    
    rb_now = QtWidgets.QRadioButton("Now", checked=True)
    g_time.addWidget(rb_now, 0, 0)
    
    rb_gps = QtWidgets.QRadioButton("GPS:")
    g_time.addWidget(rb_gps, 1, 0)
    g_time.addWidget(_small_spin_int(width=100, min_val=0), 1, 1) # GPS sec
    g_time.addWidget(QtWidgets.QLabel("sec"), 1, 2)
    g_time.addWidget(_small_spin_int(width=80, min_val=0), 1, 3) # GPS nsec
    g_time.addWidget(QtWidgets.QLabel("nsec"), 1, 4)
    
    rb_dt = QtWidgets.QRadioButton("Date/time:")
    g_time.addWidget(rb_dt, 2, 0)
    g_time.addWidget(QtWidgets.QDateEdit(QtCore.QDate(2025, 12, 23)), 2, 1)
    g_time.addWidget(QtWidgets.QTimeEdit(QtCore.QTime(4, 34, 12)), 2, 2)
    g_time.addWidget(QtWidgets.QLabel("UTC"), 2, 3) # Should be hh:mm:ss UTC label really

    # Right side checks
    g_time.addWidget(QtWidgets.QRadioButton("In the future:"), 0, 6)
    g_time.addWidget(QtWidgets.QTimeEdit(), 0, 7)
    g_time.addWidget(QtWidgets.QLabel("hh:mm:ss"), 0, 8)
    
    g_time.addWidget(QtWidgets.QRadioButton("In the past:"), 1, 6)
    g_time.addWidget(QtWidgets.QTimeEdit(), 1, 7)
    g_time.addWidget(QtWidgets.QLabel("hh:mm:ss"), 1, 8)

    g_time.addWidget(QtWidgets.QPushButton("Time now"), 2, 6)
    g_time.addWidget(QtWidgets.QPushButton("Lookup..."), 2, 7)
    
    # Slow down
    l_slow = QtWidgets.QHBoxLayout()
    l_slow.addStretch(1)
    l_slow.addWidget(QtWidgets.QLabel("Slow down:"))
    l_slow.addWidget(_small_spin_int(width=60, min_val=0))
    l_slow.addWidget(QtWidgets.QLabel("sec/avrg."))
    g_time.addLayout(l_slow, 2, 9, 1, 4)

    outer.addWidget(gb_time)

    # Group: Measurement Information
    gb_info, g_info = _create_group("Measurement Information", 'grid')
    g_info.addWidget(QtWidgets.QLabel("Measurement Time:"), 0, 0)
    g_info.addWidget(QtWidgets.QLineEdit("06/01/1980 00:00:00 UTC"), 0, 1)
    g_info.addWidget(QtWidgets.QLabel("Comment / Description:"), 0, 2)
    outer.addWidget(gb_info)
    
    txt_comment = QtWidgets.QLineEdit()
    txt_comment.setMinimumHeight(30)
    g_info.addWidget(txt_comment, 1, 0, 1, 3)

    outer.addStretch(1)
    return tab


# =================================================================================
# EXCITATION TAB
# =================================================================================
def create_excitation_tab():
    tab = QtWidgets.QWidget()
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    tab_inner = QtWidgets.QWidget()
    scroll.setWidget(tab_inner)
    
    # Outer layout
    tab_layout = QtWidgets.QVBoxLayout(tab)
    tab_layout.setContentsMargins(0,0,0,0)
    tab_layout.addWidget(scroll)

    outer = QtWidgets.QVBoxLayout(tab_inner)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(10)

    # CS Group
    gb_cs, l_cs = _create_group("Channel Selection", 'h')
    l_cs.addWidget(QtWidgets.QRadioButton("Channels 0 to 3", checked=True))
    l_cs.addWidget(QtWidgets.QRadioButton("Channels 4 to 7"))
    l_cs.addWidget(QtWidgets.QRadioButton("Channels 8 to 11"))
    l_cs.addWidget(QtWidgets.QRadioButton("Channels 12 to 15"))
    l_cs.addWidget(QtWidgets.QRadioButton("Channels 16 to 19"))
    l_cs.addStretch(1)
    outer.addWidget(gb_cs)

    # 4 Channel panels
    for i in range(4):
        gb, gl = _create_group(f"Channel {i}", 'grid')
        
        # Row 0
        gl.addWidget(QtWidgets.QCheckBox("Active"), 0, 0)
        gl.addWidget(QtWidgets.QLabel("Excitation Channel:"), 0, 1)
        gl.addWidget(QtWidgets.QComboBox(), 0, 2, 1, 10) # Span wide

        # Row 1
        gl.addWidget(QtWidgets.QLabel("Readback Channel:"), 1, 0)
        
        h_rb = QtWidgets.QHBoxLayout()
        h_rb.addWidget(QtWidgets.QRadioButton("Default", checked=True))
        h_rb.addWidget(QtWidgets.QRadioButton("None"))
        h_rb.addWidget(QtWidgets.QRadioButton("User:"))
        gl.addLayout(h_rb, 1, 1)
        
        gl.addWidget(QtWidgets.QComboBox(), 1, 2, 1, 10) # Span wide

        # Row 2
        gl.addWidget(QtWidgets.QLabel("Waveform:"), 2, 0)
        gl.addWidget(QtWidgets.QComboBox(), 2, 1) # None
        
        gl.addWidget(QtWidgets.QLabel("Waveform File:"), 2, 2)
        gl.addWidget(QtWidgets.QLineEdit(), 2, 3, 1, 8)
        gl.addWidget(QtWidgets.QPushButton("Choose..."), 2, 11)

        # Row 3
        gl.addWidget(QtWidgets.QLabel("Frequency:"), 3, 0)
        
        h_vals = QtWidgets.QHBoxLayout()
        h_vals.addWidget(_small_spin_dbl(width=80)); h_vals.addWidget(QtWidgets.QLabel("Hz"))
        h_vals.addWidget(QtWidgets.QLabel("Amplitude:")); h_vals.addWidget(_small_spin_dbl(width=80))
        h_vals.addWidget(QtWidgets.QLabel("Offset:")); h_vals.addWidget(_small_spin_dbl(width=80))
        h_vals.addWidget(QtWidgets.QLabel("Phase:")); h_vals.addWidget(_small_spin_dbl(width=80)); h_vals.addWidget(QtWidgets.QLabel("deg"))
        h_vals.addWidget(QtWidgets.QLabel("Ratio:")); h_vals.addWidget(_small_spin_dbl(width=80)); h_vals.addWidget(QtWidgets.QLabel("%"))
        gl.addLayout(h_vals, 3, 1, 1, 11)

        # Row 4
        gl.addWidget(QtWidgets.QLabel("Freq. Range:"), 4, 0)
        
        h_bot = QtWidgets.QHBoxLayout()
        h_bot.addWidget(_small_spin_dbl(width=80))
        h_bot.addWidget(QtWidgets.QLabel("Hz"))
        h_bot.addWidget(QtWidgets.QLabel("Ampl. Range:"))
        h_bot.addWidget(_small_spin_dbl(width=80))
        h_bot.addWidget(QtWidgets.QLabel("Filter:"))
        h_bot.addWidget(QtWidgets.QLineEdit())
        
        gl.addLayout(h_bot, 4, 1, 1, 10)
        gl.addWidget(QtWidgets.QPushButton("Foton..."), 4, 11)

        outer.addWidget(gb)

    outer.addStretch(1)
    return tab


# =================================================================================
# RESULT TAB (Updated for direct access)
# =================================================================================
def create_result_tab():
    tab = QtWidgets.QWidget()
    hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    
    # Define Plot Widgets first to pass references
    right_panel = QtWidgets.QWidget()
    rv = QtWidgets.QVBoxLayout(right_panel)
    rv.setContentsMargins(0, 0, 0, 0)
    
    plot1 = pg.PlotWidget(title="Graph 1")
    plot1.showGrid(x=True, y=True)
    plot1.setBackground('w')
    traces1 = []
    for i in range(8):
        c = plot1.plot(pen=None)
        # Higher index = lower Z (drawn behind)
        c.setZValue(10 - i)
        b = pg.BarGraphItem(x=[0], height=[0], width=0.04, brush=None)
        b.setZValue(10 - i)
        plot1.addItem(b)
        b.setVisible(False)
        img = pg.ImageItem()
        img.setZValue(10 - i)
        plot1.addItem(img)
        img.setVisible(False)
        traces1.append({'curve': c, 'bar': b, 'img': img})
    
    plot2 = pg.PlotWidget(title="Graph 2")
    plot2.showGrid(x=True, y=True)
    plot2.setBackground('w')
    traces2 = []
    for i in range(8):
        c = plot2.plot(pen=None)
        c.setZValue(10 - i)
        b = pg.BarGraphItem(x=[0], height=[0], width=0.04, brush=None)
        b.setZValue(10 - i)
        plot2.addItem(b)
        b.setVisible(False)
        img = pg.ImageItem()
        img.setZValue(10 - i)
        plot2.addItem(img)
        img.setVisible(False)
        traces2.append({'curve': c, 'bar': b, 'img': img})

    
    rv.addWidget(plot1)
    rv.addWidget(plot2)

    # Left Control Panel
    left_panel = QtWidgets.QWidget()
    left_vbox = QtWidgets.QVBoxLayout(left_panel)
    left_vbox.setContentsMargins(0, 0, 0, 0)
    left_vbox.setSpacing(0)
    
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setMinimumWidth(320)
    left_content = QtWidgets.QWidget()
    scroll.setWidget(left_content)
    
    lv = QtWidgets.QVBoxLayout(left_content)
    lv.setContentsMargins(6, 6, 6, 6)
    
    def add_graph_ctrl(plot_idx, target_plot, traces):
        path_frame = QtWidgets.QFrame()
        path_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        pv = QtWidgets.QVBoxLayout(path_frame)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)
        
        # --- Main Control Tabs (Simulated Multi-row via two QTabWidgets) ---
        # diaggui uses two rows of tabs. One for Style/X-axis... and another for Traces/Range...
        # We can implement this by using two QTabWidgets and a QStackedWidget to show/hide content,
        # but a simpler way that looks right is using two QTabBars and a StackedWidget.
        
        tab_row1 = QtWidgets.QTabBar()
        tab_row1.addTab("Style"); tab_row1.addTab("X-axis"); tab_row1.addTab("Y-axis"); tab_row1.addTab("Legend"); tab_row1.addTab("Param")
        
        tab_row2 = QtWidgets.QTabBar()
        tab_row2.addTab("Traces"); tab_row2.addTab("Range"); tab_row2.addTab("Units"); tab_row2.addTab("Cursor"); tab_row2.addTab("Config")
        
        # Styles
        # Normal state (for the row that is NOT currently active)
        BASE_TAB_STYLE = "QTabBar::tab { height: 25px; padding: 2px; border: 1px solid #C0C0C0; border-bottom: none; min-width: 55px; background: #E0E0E0; font-weight: normal; }"
        # Active state (for the row that IS currently active)
        ACTIVE_TAB_STYLE = BASE_TAB_STYLE + " QTabBar::tab:selected { font-weight: bold; background: #FFFFFF; border-bottom: 2px solid #FFFFFF; }"
        
        tab_row1.setStyleSheet(BASE_TAB_STYLE)
        tab_row2.setStyleSheet(ACTIVE_TAB_STYLE)
        
        pv.addWidget(tab_row1)
        pv.addWidget(tab_row2)
        
        main_stack = QtWidgets.QStackedWidget()
        pv.addWidget(main_stack)
        
        # Logic to handle switching between two rows
        def row1_changed(idx):
            if idx == -1: return
            tab_row1.setStyleSheet(ACTIVE_TAB_STYLE)
            tab_row2.blockSignals(True)
            tab_row2.setStyleSheet(BASE_TAB_STYLE)
            tab_row2.setCurrentIndex(-1)
            tab_row2.blockSignals(False)
            main_stack.setCurrentIndex(idx + 5) # 5-9 in stack
            
        def row2_changed(idx):
            if idx == -1: return
            tab_row2.setStyleSheet(ACTIVE_TAB_STYLE)
            tab_row1.blockSignals(True)
            tab_row1.setStyleSheet(BASE_TAB_STYLE)
            tab_row1.setCurrentIndex(-1)
            tab_row1.blockSignals(False)
            main_stack.setCurrentIndex(idx) # 0-4 in stack (Traces=0, Range=1...)

        # Prepare initial state (Traces selected)
        tab_row1.blockSignals(True)
        tab_row1.setCurrentIndex(-1)
        tab_row1.blockSignals(False)
        
        # --- Traces Tab (Contains the Trace 0-7 tabs) ---
        traces_tab_widget = QtWidgets.QWidget()
        traces_vbox = QtWidgets.QVBoxLayout(traces_tab_widget)
        traces_vbox.setContentsMargins(2, 2, 2, 2)
        
        # Global Graph Type (Now inside Traces tab or top level?)
        # Looking at screenshots, "Graph" is above the 0-7 tabs.
        r_graph = QtWidgets.QHBoxLayout()
        r_graph.addWidget(QtWidgets.QLabel("Graph:"))
        graph_combo = QtWidgets.QComboBox()
        graph_combo.addItems([
            "Time Series", "Amplitude Spectral Density", "Power Spectral Density",
            "Cross Spectral Density", "Coherence", "Squared Coherence",
            "Transfer Function", "Spectrogram"
        ])
        r_graph.addWidget(graph_combo)
        traces_vbox.addLayout(r_graph)

        # Trace Tabs (0-7) - Force to fit width
        trace_tab_ctrl = QtWidgets.QTabWidget()
        # Use stylesheet to ensure tabs take up equal space and don't scroll
        trace_tab_ctrl.setStyleSheet("""
            QTabBar::tab { height: 25px; width: 32px; margin: 0; padding: 0; background: #E0E0E0; }
            QTabBar::tab:selected { font-weight: bold; background: #FFFFFF; }
        """)
        # Prevent scroll buttons
        trace_tab_ctrl.setUsesScrollButtons(False)
        traces_vbox.addWidget(trace_tab_ctrl)
        
        trace_controls = []
        channel_names = ["HF_sine", "LF_sine", "beating_sine", "white_noise", "sine_plus_noise", "square_wave", "sawtooth_wave", "random_walk"]
        colors = [('Red', '#FF0000'), ('Blue', '#0000FF'), ('Green', '#00FF00'), ('Black', '#000000'), ('Magenta', '#FF00FF'), ('Cyan', '#00FFFF'), ('Yellow', '#FFFF00'), ('Orange', '#FFA500')]
        line_styles = [("Solid", QtCore.Qt.SolidLine), ("Dash", QtCore.Qt.DashLine), ("Dot", QtCore.Qt.DotLine)]
        symbols = [("Circle", "o"), ("Square", "s"), ("Triangle", "t")]
        fill_patterns = [("Solid", QtCore.Qt.SolidPattern), ("Dense", QtCore.Qt.Dense3Pattern)]

        def update_style(t_idx):
            ctrl = trace_controls[t_idx]
            target_curve = traces[t_idx]['curve']
            target_bar = traces[t_idx]['bar']
            target_img = traces[t_idx]['img']
            
            g_type = graph_combo.currentText()
            is_active = ctrl['active'].isChecked()
            is_spec_gram = g_type == "Spectrogram"
            
            dual = g_type in ["Cross Spectral Density", "Coherence", "Squared Coherence", "Transfer Function"]
            ctrl['chan_b'].setEnabled(dual)
            ctrl['g_style'].setEnabled(not is_spec_gram)

            if not is_active:
                target_curve.setPen(None); target_curve.setSymbol(None); target_bar.setVisible(False); target_img.setVisible(False)
                return

            if is_spec_gram:
                target_curve.setPen(None); target_curve.setSymbol(None); target_bar.setVisible(False); target_img.setVisible(True)
                if not hasattr(target_img, '_lut_set'):
                    try:
                        lut = pg.colormap.get('viridis').getLookupTable()
                        target_img.setLookupTable(lut); target_img._lut_set = True
                    except: pass
                return

            c_hex = colors[ctrl['line_c'].currentIndex()][1]
            ctrl['line_c'].setStyleSheet(f"background-color: {c_hex};")
            
            if ctrl['line_chk'].isChecked():
                pen = pg.mkPen(color=c_hex, width=ctrl['line_w'].value(), style=ctrl['line_s'].itemData(ctrl['line_s'].currentIndex()))
                target_curve.setPen(pen); target_bar.setVisible(False); target_img.setVisible(False)
            elif ctrl['bar_chk'].isChecked():
                target_curve.setPen(None); target_bar.setVisible(True); target_img.setVisible(False)
                brush = QtGui.QBrush(QtGui.QColor(c_hex), ctrl['bar_s'].itemData(ctrl['bar_s'].currentIndex()))
                target_bar.setOpts(brush=brush, width=ctrl['bar_w'].value())
            else:
                target_curve.setPen(None); target_bar.setVisible(False); target_img.setVisible(False)

            if ctrl['sym_chk'].isChecked():
                target_curve.setSymbol(ctrl['sym_s'].itemData(ctrl['sym_s'].currentIndex()))
                target_curve.setSymbolBrush(c_hex); target_curve.setSymbolPen(c_hex); target_curve.setSymbolSize(ctrl['sym_w'].value())
            else:
                target_curve.setSymbol(None)

        for i in range(8):
            page = QtWidgets.QWidget()
            pl = QtWidgets.QVBoxLayout(page)
            pl.setContentsMargins(4,4,4,4); pl.setSpacing(2)
            active_chk = QtWidgets.QCheckBox("Active"); active_chk.setChecked(i==0)
            pl.addWidget(active_chk)
            gc = QtWidgets.QGroupBox("Channels"); gl = QtWidgets.QGridLayout(gc); gl.setContentsMargins(4,4,4,4)
            gl.addWidget(QtWidgets.QLabel("A:"), 0, 0); ca = QtWidgets.QComboBox(); ca.addItems(channel_names); gl.addWidget(ca, 0, 1)
            gl.addWidget(QtWidgets.QLabel("B:"), 1, 0); cb = QtWidgets.QComboBox(); cb.addItems(channel_names); gl.addWidget(cb, 1, 1)
            pl.addWidget(gc)
            gs = QtWidgets.QGroupBox("Style"); gls = QtWidgets.QGridLayout(gs); gls.setContentsMargins(4,4,4,4)
            lchk = QtWidgets.QCheckBox("Line"); lchk.setChecked(True); gls.addWidget(lchk, 0, 0)
            lc = QtWidgets.QComboBox(); lc.setFixedWidth(40); [lc.addItem("") or lc.setItemData(j, QtGui.QColor(c[1]), QtCore.Qt.BackgroundRole) for j,c in enumerate(colors)]; gls.addWidget(lc, 0, 1)
            lc.setCurrentIndex(i % len(colors))
            ls = QtWidgets.QComboBox(); [ls.addItem(n, v) for n,v in line_styles]; gls.addWidget(ls, 0, 2)
            lw = _small_spin_int(1,10, 35); gls.addWidget(lw, 0, 3)
            schk = QtWidgets.QCheckBox("Symbol"); gls.addWidget(schk, 1, 0)
            sc = QtWidgets.QComboBox(); sc.setFixedWidth(40); [sc.addItem("") or sc.setItemData(j, QtGui.QColor(c[1]), QtCore.Qt.BackgroundRole) for j,c in enumerate(colors)]; gls.addWidget(sc, 1, 1)
            sc.setCurrentIndex(i % len(colors))
            ss = QtWidgets.QComboBox(); [ss.addItem(n, v) for n,v in symbols]; gls.addWidget(ss, 1, 2)
            sw = _small_spin_dbl(1, 35, 1, 50); sw.setValue(5); gls.addWidget(sw, 1, 3)
            bchk = QtWidgets.QCheckBox("Bar"); gls.addWidget(bchk, 2, 0)
            bc = QtWidgets.QComboBox(); bc.setFixedWidth(40); [bc.addItem("") or bc.setItemData(j, QtGui.QColor(c[1]), QtCore.Qt.BackgroundRole) for j,c in enumerate(colors)]; gls.addWidget(bc, 2, 1)
            bc.setCurrentIndex(i % len(colors))
            bs = QtWidgets.QComboBox(); [bs.addItem(n, v) for n,v in fill_patterns]; gls.addWidget(bs, 2, 2)
            bw = _small_spin_dbl(2, 35, 0.01, 10, 0.01); bw.setValue(0.04); gls.addWidget(bw, 2, 3)
            pl.addWidget(gs)
            trace_tab_ctrl.addTab(page, str(i))
            ctrl_set = {'active': active_chk, 'chan_a': ca, 'chan_b': cb, 'g_style': gs, 'line_chk': lchk, 'line_c': lc, 'line_s': ls, 'line_w': lw,'sym_chk': schk, 'bar_chk': bchk, 'bar_s': bs, 'bar_w': bw, 'sym_s': ss, 'sym_w': sw}
            trace_controls.append(ctrl_set)
            active_chk.toggled.connect(lambda _, x=i: update_style(x))
            ca.currentIndexChanged.connect(lambda _, x=i: update_style(x))
            cb.currentIndexChanged.connect(lambda _, x=i: update_style(x))
            lchk.toggled.connect(lambda _, x=i: (bchk.setChecked(False) if lchk.isChecked() else None, update_style(x)))
            bchk.toggled.connect(lambda _, x=i: (lchk.setChecked(False) if bchk.isChecked() else None, update_style(x)))
            for w in [lc, ls, lw, schk, sc, ss, sw, bc, bs, bw]:
                if isinstance(w, QtWidgets.QComboBox): w.currentIndexChanged.connect(lambda _, x=i: update_style(x))
                else: w.valueChanged.connect(lambda _, x=i: update_style(x)) if hasattr(w, 'valueChanged') else w.toggled.connect(lambda _, x=i: update_style(x))

        # --- Range Tab ---
        range_tab_widget = QtWidgets.QWidget()
        range_vbox = QtWidgets.QVBoxLayout(range_tab_widget)
        range_vbox.setContentsMargins(4,4,4,4)
        
        def update_range_logic():
            # Y Axis
            y_log = rb_y_log.isChecked()
            y_auto = rb_y_auto.isChecked()
            target_plot.setLogMode(y=y_log)
            if y_auto: target_plot.enableAutoRange(axis='y')
            else: target_plot.setYRange(sb_y_from.value(), sb_y_to.value(), padding=0)
            
            # X Axis
            x_log = rb_x_log.isChecked()
            x_auto = rb_x_auto.isChecked()
            target_plot.setLogMode(x=x_log)
            if x_auto: target_plot.enableAutoRange(axis='x')
            else: target_plot.setXRange(sb_x_from.value(), sb_x_to.value(), padding=0)

        gy = QtWidgets.QGroupBox("Y axis"); gly = QtWidgets.QGridLayout(gy)
        gly.addWidget(QtWidgets.QLabel("Scale:"), 0, 0); rb_y_lin = QtWidgets.QRadioButton("linear"); rb_y_log = QtWidgets.QRadioButton("log"); rb_y_lin.setChecked(True)
        hly1 = QtWidgets.QHBoxLayout(); hly1.addWidget(rb_y_lin); hly1.addWidget(rb_y_log); gly.addLayout(hly1, 0, 1)
        gly.addWidget(QtWidgets.QLabel("Range:"), 1, 0); rb_y_auto = QtWidgets.QRadioButton("automatic"); rb_y_man = QtWidgets.QRadioButton("manual"); rb_y_auto.setChecked(True)
        hly2 = QtWidgets.QHBoxLayout(); hly2.addWidget(rb_y_auto); hly2.addWidget(rb_y_man); gly.addLayout(hly2, 1, 1)
        gly.addWidget(QtWidgets.QLabel("From"), 2, 0); sb_y_from = _small_spin_dbl(2, 60); gly.addWidget(sb_y_from, 2, 1)
        gly.addWidget(QtWidgets.QLabel("To"), 2, 2); sb_y_to = _small_spin_dbl(2, 60, max_val=1e12); sb_y_to.setValue(1.1); gly.addWidget(sb_y_to, 2, 3)
        range_vbox.addWidget(gy)

        gx = QtWidgets.QGroupBox("X axis"); glx = QtWidgets.QGridLayout(gx)
        glx.addWidget(QtWidgets.QLabel("Scale:"), 0, 0); rb_x_lin = QtWidgets.QRadioButton("linear"); rb_x_log = QtWidgets.QRadioButton("log"); rb_x_lin.setChecked(True)
        hlx1 = QtWidgets.QHBoxLayout(); hlx1.addWidget(rb_x_lin); hlx1.addWidget(rb_x_log); glx.addLayout(hlx1, 0, 1)
        glx.addWidget(QtWidgets.QLabel("Range:"), 1, 0); rb_x_auto = QtWidgets.QRadioButton("automatic"); rb_x_man = QtWidgets.QRadioButton("manual"); rb_x_auto.setChecked(True)
        hlx2 = QtWidgets.QHBoxLayout(); hlx2.addWidget(rb_x_auto); hlx2.addWidget(rb_x_man); glx.addLayout(hlx2, 1, 1)
        glx.addWidget(QtWidgets.QLabel("From"), 2, 0); sb_x_from = _small_spin_dbl(2, 60); glx.addWidget(sb_x_from, 2, 1)
        glx.addWidget(QtWidgets.QLabel("To"), 2, 2); sb_x_to = _small_spin_dbl(2, 60, max_val=1e12); sb_x_to.setValue(10); glx.addWidget(sb_x_to, 2, 3)
        range_vbox.addWidget(gx)
        
        gb = QtWidgets.QGroupBox("Bin"); hl_b = QtWidgets.QHBoxLayout(gb)
        hl_b.addWidget(_small_spin_int(1, 1000, 50)); hl_b.addWidget(QtWidgets.QCheckBox("Log spacing")); range_vbox.addWidget(gb)

        for w in [rb_y_lin, rb_y_log, rb_y_auto, rb_y_man, sb_y_from, sb_y_to, rb_x_lin, rb_x_log, rb_x_auto, rb_x_man, sb_x_from, sb_x_to]:
            if isinstance(w, QtWidgets.QRadioButton): w.toggled.connect(update_range_logic)
            else: w.valueChanged.connect(update_range_logic)

        def update_axis_labels():
            txt = graph_combo.currentText()
            style_title_edit.setText(txt)
            if txt == "Time Series":
                xaxis_title_edit.setText("Time"); yaxis_title_edit.setText("Amplitude")
                rb_x_log.setChecked(False); rb_y_log.setChecked(False)
            elif txt == "Spectrogram":
                xaxis_title_edit.setText("Time"); yaxis_title_edit.setText("Frequency")
                rb_x_log.setChecked(False); rb_y_log.setChecked(False)
            elif "Coherence" in txt:
                xaxis_title_edit.setText("Frequency"); yaxis_title_edit.setText("|Coherence|" if "Squared" not in txt else "Coherence^2")
                rb_x_log.setChecked(True); rb_y_log.setChecked(False)
                sb_y_from.setValue(0); sb_y_to.setValue(1); rb_y_man.setChecked(True)
            else:
                xaxis_title_edit.setText("Frequency"); yaxis_title_edit.setText(txt)
                rb_x_log.setChecked(True); rb_y_log.setChecked(True)
            for i in range(8): update_style(i)
            # update_axis labels already connected to widgets, they will update plot titles
            update_range_logic()

        graph_combo.currentIndexChanged.connect(update_axis_labels)

        # --- Units Tab ---
        units_tab_widget = QtWidgets.QWidget()
        ul = QtWidgets.QVBoxLayout(units_tab_widget); ul.setContentsMargins(4,4,4,4)
        ug = QtWidgets.QGroupBox("Units"); ugl = QtWidgets.QGridLayout(ug)
        ugl.addWidget(QtWidgets.QLabel("X:"), 0, 0); uxa = QtWidgets.QComboBox(); uxa.addItems(["-", "s", "Hz"]); ugl.addWidget(uxa, 0, 1)
        uxb = QtWidgets.QComboBox(); uxb.addItems(["default"]); ugl.addWidget(uxb, 0, 2)
        ugl.addWidget(QtWidgets.QLabel("Y:"), 1, 0); uya = QtWidgets.QComboBox(); uya.addItems(["-", "m", "V", "pk/rtHz"]); ugl.addWidget(uya, 1, 1)
        uyb = QtWidgets.QComboBox(); uyb.addItems(["default"]); ugl.addWidget(uyb, 1, 2)
        ul.addWidget(ug)
        dg = QtWidgets.QGroupBox("Display"); dgl = QtWidgets.QGridLayout(dg)
        dgl.addWidget(QtWidgets.QLabel("X:"), 0, 0); dxa = QtWidgets.QComboBox(); dxa.addItems(["Standard", "Time"]); dgl.addWidget(dxa, 0, 1)
        dgl.addWidget(QtWidgets.QLabel("Y:"), 1, 0); dya = QtWidgets.QComboBox(); dya.addItems(["Magnitude", "Phase", "dB"]); dgl.addWidget(dya, 1, 1)
        ul.addWidget(dg)
        sg = QtWidgets.QGroupBox("Scaling"); sgl = QtWidgets.QGridLayout(sg)
        sgl.addWidget(QtWidgets.QLabel("X Slope:"), 0,0); sgl.addWidget(_small_spin_dbl(4, 50, -1e6, 1e6, 1), 0,1)
        sgl.addWidget(QtWidgets.QLabel("Offset:"), 0,2); sgl.addWidget(_small_spin_dbl(4, 50, -1e6, 1e6, 1), 0,3)
        sgl.addWidget(QtWidgets.QLabel("Y Slope:"), 1,0); sgl.addWidget(_small_spin_dbl(4, 50, -1e6, 1e6, 1), 1,1)
        sgl.addWidget(QtWidgets.QLabel("Offset:"), 1,2); sgl.addWidget(_small_spin_dbl(4, 50, -1e6, 1e6, 1), 1,3)
        ul.addWidget(sg)
        btn_cal = QtWidgets.QPushButton("Calibration..."); ul.addWidget(btn_cal, 0, QtCore.Qt.AlignRight)

        # --- Cursor Tab ---
        cursor_tab_widget = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(cursor_tab_widget); cl.setContentsMargins(4,4,4,4)
        cl.addWidget(QtWidgets.QLabel("Trace:"))
        ctabs = QtWidgets.QTabBar(); ctabs.setExpanding(False); [ctabs.addTab(str(i)) for i in range(8)]; cl.addWidget(ctabs)
        r_curs = QtWidgets.QHBoxLayout()
        ag = QtWidgets.QGroupBox("Active"); agl = QtWidgets.QVBoxLayout(ag); agl.addWidget(QtWidgets.QCheckBox("1")); agl.addWidget(QtWidgets.QCheckBox("2")); r_curs.addWidget(ag)
        stg = QtWidgets.QGroupBox("Style"); stgl = QtWidgets.QGridLayout(stg)
        stgl.addWidget(QtWidgets.QRadioButton("None"), 0,0); stgl.addWidget(QtWidgets.QRadioButton("Vert."), 0,1)
        stgl.addWidget(QtWidgets.QRadioButton("Cross"), 1,0); stgl.addWidget(QtWidgets.QRadioButton("Horiz."), 1,1); r_curs.addWidget(stg)
        tyg = QtWidgets.QGroupBox("Type"); tygl = QtWidgets.QVBoxLayout(tyg); tygl.addWidget(QtWidgets.QRadioButton("Abs.")); tygl.addWidget(QtWidgets.QRadioButton("Delta")); r_curs.addWidget(tyg)
        cl.addLayout(r_curs)
        vg = QtWidgets.QGroupBox("Values"); vgl = QtWidgets.QGridLayout(vg)
        vgl.addWidget(QtWidgets.QLabel("X1:"),0,0); vgl.addWidget(QtWidgets.QLineEdit(),0,1); vgl.addWidget(QtWidgets.QLabel("Y1:"),0,2); vgl.addWidget(QtWidgets.QLineEdit(),0,3)
        vgl.addWidget(QtWidgets.QLabel("X2:"),1,0); vgl.addWidget(QtWidgets.QLineEdit(),1,1); vgl.addWidget(QtWidgets.QLabel("Y2:"),1,2); vgl.addWidget(QtWidgets.QLineEdit(),1,3); cl.addWidget(vg)
        sttg = QtWidgets.QGroupBox("Statistics"); sttgl = QtWidgets.QHBoxLayout(sttg)
        sttgl.addWidget(QtWidgets.QComboBox()); [sttgl.addWidget(QtWidgets.QLabel("0.000")) for _ in range(2)]; cl.addWidget(sttg)

        # --- Config Tab ---
        config_tab_widget = QtWidgets.QWidget()
        cfl = QtWidgets.QVBoxLayout(config_tab_widget); cfl.setContentsMargins(4,4,4,4)
        afg = QtWidgets.QGroupBox("Auto configuration"); afgl = QtWidgets.QVBoxLayout(afg)
        afgl.addWidget(QtWidgets.QCheckBox("Plot settings")); achk = QtWidgets.QCheckBox("Respect user selection"); achk.setContentsMargins(20,0,0,0); afgl.addWidget(achk)
        afgl.addWidget(QtWidgets.QCheckBox("Axes title")); afgl.addWidget(QtWidgets.QCheckBox("Bin")); afgl.addWidget(QtWidgets.QCheckBox("Time Adjust"))
        cfl.addWidget(afg)
        btn_hl = QtWidgets.QHBoxLayout(); btn_hl.addStretch(1); btn_hl.addWidget(QtWidgets.QPushButton("Store...")); btn_hl.addWidget(QtWidgets.QPushButton("Restore...")); cfl.addLayout(btn_hl)
        cfl.addStretch(1)

        def _color_box():
            cb = QtWidgets.QComboBox(); cb.setFixedWidth(40)
            [cb.addItem("") or cb.setItemData(j, QtGui.QColor(c[1]), QtCore.Qt.BackgroundRole) for j,c in enumerate(colors)]; return cb

        # --- Style Tab ---
        style_tab_widget = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(style_tab_widget); sl.setContentsMargins(4,4,4,4)
        tg = QtWidgets.QGroupBox("Title"); tgl = QtWidgets.QGridLayout(tg); tgl.setSpacing(2)
        style_title_edit = QtWidgets.QLineEdit("Time series"); tgl.addWidget(style_title_edit, 0, 0, 1, 4)
        tgl.addWidget(QtWidgets.QComboBox(), 1, 0); tgl.addWidget(QtWidgets.QComboBox(), 1, 1); tgl.addWidget(_small_spin_dbl(3, 45, 0, 1, 0.001), 1, 2)
        rbl = QtWidgets.QHBoxLayout(); [rbl.addWidget(QtWidgets.QRadioButton(t)) for t in ["Left", "Center", "Right"]]; tgl.addLayout(rbl, 2, 0, 1, 3)
        tgl.addWidget(_color_box(), 2, 3); sl.addWidget(tg)
        
        def update_plot_title(): target_plot.setTitle(style_title_edit.text() if style_title_edit.text() else None)
        style_title_edit.textChanged.connect(update_plot_title)
        mg = QtWidgets.QGroupBox("Margins"); mgl = QtWidgets.QHBoxLayout(mg); mgl.setSpacing(2)
        [mgl.addWidget(QtWidgets.QLabel(t)) or mgl.addWidget(_small_spin_dbl(2, 40, 0, 1, 0.01)) for t in ["L", "R", "T", "B"]]; sl.addWidget(mg); sl.addStretch(1)

        # --- X-axis Tab ---
        xaxis_tab_widget = QtWidgets.QWidget(); xal = QtWidgets.QVBoxLayout(xaxis_tab_widget); xal.setContentsMargins(4,4,4,4)
        xtg = QtWidgets.QGroupBox("Title"); xtgl = QtWidgets.QGridLayout(xtg)
        xaxis_title_edit = QtWidgets.QLineEdit("Time"); xtgl.addWidget(xaxis_title_edit, 0, 0, 1, 4)
        xtgl.addWidget(QtWidgets.QLabel("Size:"), 1, 0); xtgl.addWidget(_small_spin_dbl(3, 45, 0, 1, 0.001), 1, 1)
        xtgl.addWidget(QtWidgets.QLabel("Offset:"), 1, 2); xtgl.addWidget(_small_spin_dbl(2, 45, 0, 10, 0.01), 1, 3)
        xtgl.addWidget(_color_box(), 1, 4); xal.addWidget(xtg)
        xkg = QtWidgets.QGroupBox("Ticks/Axis"); xkgl = QtWidgets.QGridLayout(xkg)
        xkgl.addWidget(QtWidgets.QLabel("Length:"), 0, 0); xkgl.addWidget(_small_spin_dbl(3, 45, 0, 1, 0.001), 0, 1)
        xkgl.addWidget(QtWidgets.QCheckBox("Both sides"), 0, 2); x_grid_chk = QtWidgets.QCheckBox("Grid"); xkgl.addWidget(x_grid_chk, 0, 3)
        xkgl.addWidget(QtWidgets.QLabel("Divisions:"), 1, 0); xkgl.addWidget(_small_spin_int(0, 100, 40), 1, 1); xkgl.addWidget(_small_spin_int(0, 100, 40), 1, 2); xkgl.addWidget(_small_spin_int(0, 100, 40), 1, 3)
        xkgl.addWidget(_color_box(), 1, 4); xal.addWidget(xkg)
        
        def update_xaxis(): target_plot.setLabel('bottom', xaxis_title_edit.text()); target_plot.showGrid(x=x_grid_chk.isChecked())
        xaxis_title_edit.textChanged.connect(update_xaxis); x_grid_chk.toggled.connect(update_xaxis)
        xlg = QtWidgets.QGroupBox("Labels"); xlgl = QtWidgets.QGridLayout(xlg)
        xlgl.addWidget(QtWidgets.QLabel("Size:"), 0, 0); xlgl.addWidget(_small_spin_dbl(3, 45, 0, 1, 0.001), 0, 1)
        xlgl.addWidget(QtWidgets.QLabel("Offset:"), 0, 2); xlgl.addWidget(_small_spin_dbl(3, 45, 0, 1, 0.001), 0, 3)
        xlgl.addWidget(_color_box(), 0, 4); xal.addWidget(xlg)
        xfg = QtWidgets.QGroupBox("Font"); xfgl = QtWidgets.QHBoxLayout(xfg); xfgl.addWidget(QtWidgets.QComboBox()); xfgl.addWidget(QtWidgets.QComboBox()); xfgl.addWidget(QtWidgets.QCheckBox("Center")); xal.addWidget(xfg)

        # --- Y-axis Tab ---
        yaxis_tab_widget = QtWidgets.QWidget(); yal = QtWidgets.QVBoxLayout(yaxis_tab_widget); yal.setContentsMargins(4,4,4,4)
        ytg = QtWidgets.QGroupBox("Title"); ytgl = QtWidgets.QGridLayout(ytg)
        yaxis_title_edit = QtWidgets.QLineEdit("Signal"); ytgl.addWidget(yaxis_title_edit, 0, 0, 1, 4)
        ytgl.addWidget(QtWidgets.QLabel("Size:"), 1, 0); ytgl.addWidget(_small_spin_dbl(3, 45, 0, 1, 0.001), 1, 1)
        ytgl.addWidget(QtWidgets.QLabel("Offset:"), 1, 2); ytgl.addWidget(_small_spin_dbl(2, 45, 0, 10, 0.01), 1, 3)
        ytgl.addWidget(_color_box(), 1, 4); yal.addWidget(ytg)
        ykg = QtWidgets.QGroupBox("Ticks/Axis"); ykgl = QtWidgets.QGridLayout(ykg)
        ykgl.addWidget(QtWidgets.QLabel("Length:"), 0, 0); ykgl.addWidget(_small_spin_dbl(3, 45, 0, 1, 0.001), 0, 1)
        ykgl.addWidget(QtWidgets.QCheckBox("Both sides"), 0, 2); y_grid_chk = QtWidgets.QCheckBox("Grid"); ykgl.addWidget(y_grid_chk, 0, 3)
        ykgl.addWidget(QtWidgets.QLabel("Divisions:"), 1, 0); ykgl.addWidget(_small_spin_int(0, 100, 40), 1, 1); ykgl.addWidget(_small_spin_int(0, 100, 40), 1, 2); ykgl.addWidget(_small_spin_int(0, 100, 40), 1, 3)
        ykgl.addWidget(_color_box(), 1, 4); yal.addWidget(ykg)
        
        def update_yaxis(): target_plot.setLabel('left', yaxis_title_edit.text()); target_plot.showGrid(y=y_grid_chk.isChecked())
        yaxis_title_edit.textChanged.connect(update_yaxis); y_grid_chk.toggled.connect(update_yaxis)
        ylg = QtWidgets.QGroupBox("Labels"); ylgl = QtWidgets.QGridLayout(ylg)
        ylgl.addWidget(QtWidgets.QLabel("Size:"), 0, 0); ylgl.addWidget(_small_spin_dbl(3, 45, 0, 1, 0.001), 0, 1)
        ylgl.addWidget(QtWidgets.QLabel("Offset:"), 0, 2); ylgl.addWidget(_small_spin_dbl(3, 45, 0, 1, 0.001), 0, 3)
        ylgl.addWidget(_color_box(), 0, 4); yal.addWidget(ylg)
        yfg = QtWidgets.QGroupBox("Font"); yfgl = QtWidgets.QHBoxLayout(yfg); yfgl.addWidget(QtWidgets.QComboBox()); yfgl.addWidget(QtWidgets.QComboBox()); yfgl.addWidget(QtWidgets.QCheckBox("Center")); yal.addWidget(yfg)

        # --- Legend Tab ---
        legend_tab_widget = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(legend_tab_widget); ll.setContentsMargins(4,4,4,4)
        legend_show_chk = QtWidgets.QCheckBox("Show"); ll.addWidget(legend_show_chk)
        lpg = QtWidgets.QGroupBox("Placement"); lpgl = QtWidgets.QGridLayout(lpg)
        
        target_legend = target_plot.addLegend(); target_legend.hide()
        def update_legend():
            if legend_show_chk.isChecked(): target_legend.show()
            else: target_legend.hide()
        legend_show_chk.toggled.connect(update_legend)
        lpgl.addWidget(QtWidgets.QRadioButton("Top left"), 0,0); lpgl.addWidget(QtWidgets.QRadioButton("Top Right"), 0,1)
        lpgl.addWidget(QtWidgets.QRadioButton("Bottom left"), 1,0); lpgl.addWidget(QtWidgets.QRadioButton("Bottom Right"), 1,1)
        lpgl.addWidget(QtWidgets.QLabel("X:"), 2,0); lpgl.addWidget(_small_spin_dbl(2, 45, 0, 10, 0.01), 2,1)
        lpgl.addWidget(QtWidgets.QLabel("Y:"), 2,2); lpgl.addWidget(_small_spin_dbl(2, 45, 0, 10, 0.01), 2,3)
        lpgl.addWidget(QtWidgets.QLabel("Size:"), 2,4); lpgl.addWidget(_small_spin_dbl(1, 45, 0, 10, 0.1), 2,5); ll.addWidget(lpg)
        lsg = QtWidgets.QGroupBox("Symbol style"); lsgl = QtWidgets.QHBoxLayout(lsg)
        lsgl.addWidget(QtWidgets.QRadioButton("Same as trace")); lsgl.addWidget(QtWidgets.QRadioButton("None")); ll.addWidget(lsg)
        ltg = QtWidgets.QGroupBox("Text"); ltgl = QtWidgets.QVBoxLayout(ltg)
        rhl = QtWidgets.QHBoxLayout(); rhl.addWidget(QtWidgets.QRadioButton("Auto")); rhl.addWidget(QtWidgets.QRadioButton("User")); ltgl.addLayout(rhl)
        tb = QtWidgets.QTabBar(); tb.setExpanding(False); [tb.addTab(str(i)) for i in range(8)]; ltgl.addWidget(tb)
        ltgl.addWidget(QtWidgets.QLineEdit()); ll.addWidget(ltg)

        # --- Param Tab ---
        param_tab_widget = QtWidgets.QWidget(); pl = QtWidgets.QVBoxLayout(param_tab_widget); pl.setContentsMargins(4,4,4,4)
        pl.addWidget(QtWidgets.QCheckBox("Show"))
        pfg = QtWidgets.QGroupBox("Time format"); pfgl = QtWidgets.QHBoxLayout(pfg)
        pfgl.addWidget(QtWidgets.QRadioButton("Date/time UTC")); pfgl.addWidget(QtWidgets.QRadioButton("GPS seconds")); pl.addWidget(pfg)
        vbg = QtWidgets.QGroupBox("Variable"); vbgl = QtWidgets.QVBoxLayout(vbg)
        [vbgl.addWidget(QtWidgets.QCheckBox(v)) for v in ["Start time", "Number of averages", "Third parameter", "Statistics", "Histogram Under/Overflow"]]; pl.addWidget(vbg); pl.addStretch(1)

        # Assemble Main Stack
        # Row 2 items (Indices 0-4)
        main_stack.addWidget(traces_tab_widget) # 0: Traces
        main_stack.addWidget(range_tab_widget)  # 1: Range
        main_stack.addWidget(units_tab_widget)  # 2: Units
        main_stack.addWidget(cursor_tab_widget) # 3: Cursor
        main_stack.addWidget(config_tab_widget) # 4: Config
        
        # Row 1 items (Indices 5-9)
        main_stack.addWidget(style_tab_widget) # 5: Style
        main_stack.addWidget(xaxis_tab_widget) # 6: X-axis
        main_stack.addWidget(yaxis_tab_widget) # 7: Y-axis
        main_stack.addWidget(legend_tab_widget)# 8: Legend
        main_stack.addWidget(param_tab_widget) # 9: Param
        
        # Connect signals and set initial selection
        tab_row1.currentChanged.connect(row1_changed)
        tab_row2.currentChanged.connect(row2_changed)
        tab_row2.setCurrentIndex(0) # This will trigger row2_changed and set its style to ACTIVE

        lv.addWidget(path_frame)
        update_axis_labels()
        
        return {'graph_combo': graph_combo, 'traces': trace_controls}

    info1 = add_graph_ctrl(1, plot1, traces1)
    line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.HLine)
    lv.addWidget(line)
    info2 = add_graph_ctrl(2, plot2, traces2)
    lv.addStretch(1)
    
    left_vbox.addWidget(scroll)
    
    hsplit.addWidget(left_panel)
    hsplit.addWidget(right_panel)
    hsplit.setStretchFactor(1, 1)
    
    main_layout = QtWidgets.QVBoxLayout(tab)
    main_layout.addWidget(hsplit)
    
    b_names = ["Reset", "Zoom", "Active", "New", "Options...", "Import...", "Export...", "Reference...", "Calibration...", "Print..."]
    bot_toolbar = QtWidgets.QHBoxLayout()
    for name in b_names:
        bot_toolbar.addWidget(QtWidgets.QPushButton(name))
    bot_toolbar.addStretch(1)
    main_layout.addLayout(bot_toolbar)

    return tab, info1, info2, traces1, traces2


# =================================================================================
# MAIN WINDOW
# =================================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CDS Diagnostic Test Tools")
        self.resize(1100, 850)

        mb = self.menuBar()
        for m in ["File", "Edit", "Measurement", "Plot", "Window", "Help"]:
            mb.addMenu(m)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(create_input_tab(), "Input")
        self.tabs.addTab(create_measurement_tab(), "Measurement")
        self.tabs.addTab(create_excitation_tab(), "Excitation")
        
        # Result tab returns info dicts
        res_tab, self.graph_info1, self.graph_info2, self.traces1, self.traces2 = create_result_tab()
        self.tabs.addTab(res_tab, "Result")

        bottom_widget = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bottom_widget)
        bl.setContentsMargins(20, 10, 20, 10)
        
        # Store buttons
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.setMinimumWidth(120)
        self.btn_start.setStyleSheet("font-weight: bold;")
        
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.setMinimumWidth(120); self.btn_pause.setEnabled(False)
        
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_resume.setMinimumWidth(120); self.btn_resume.setEnabled(False)
        
        self.btn_abort = QtWidgets.QPushButton("Abort")
        self.btn_abort.setMinimumWidth(120)

        bl.addWidget(self.btn_start)
        bl.addStretch(1)
        bl.addWidget(self.btn_pause)
        bl.addStretch(1)
        bl.addWidget(self.btn_resume)
        bl.addStretch(1)
        bl.addWidget(self.btn_abort)
        
        # Button logic connection
        self.btn_start.clicked.connect(self.start_animation)
        self.btn_pause.clicked.connect(self.pause_animation)
        self.btn_resume.clicked.connect(self.resume_animation)
        self.btn_abort.clicked.connect(self.stop_animation)

        self.status_line = QtWidgets.QWidget()
        sl = QtWidgets.QHBoxLayout(self.status_line)
        sl.setContentsMargins(2, 0, 2, 0)
        sl.addWidget(QtWidgets.QLabel(""))
        sl.addStretch(1)
        sl.addWidget(QtWidgets.QLabel("Repeat"))
        sl.addWidget(QtWidgets.QLabel("|"))
        sl.addWidget(QtWidgets.QLabel("Fourier tools"))
        sl.addWidget(QtWidgets.QLabel("|"))
        sl.addWidget(QtWidgets.QLabel(""))
        
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        cv = QtWidgets.QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.tabs)
        cv.addWidget(bottom_widget)
        cv.addWidget(self.status_line)
        
        # Timer Setup
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graphs)
        self.time_counter = 0.0
        self.x_data = np.linspace(0, 10, 150)

    def start_animation(self):
        # Switch to result tab automatically
        self.tabs.setCurrentIndex(3)
        
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_abort.setEnabled(True)
        
        self.timer.start(50) # 20 ms -> 50 fps

    def pause_animation(self):
        self.timer.stop()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.btn_abort.setEnabled(True)

    def resume_animation(self):
        self.timer.start(50)
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_abort.setEnabled(True)

    def stop_animation(self):
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_abort.setEnabled(True)
        
        # Reset data for all traces
        for t_list in [self.traces1, self.traces2]:
            for t in t_list:
                t['curve'].setData([], [])
                t['bar'].setOpts(x=[0], height=[0])
                t['img'].clear()
        self.time_counter = 0.0

    def update_graphs(self):
        self.time_counter += 0.05
        
        # Generate 8 base waveforms
        t = self.x_data
        tc = self.time_counter
        
        waves = []
        # CH 0: High Freq Sine
        waves.append(np.sin(2 * np.pi * 5 * (t + tc)))
        # CH 1: Low Freq Sine
        waves.append(np.sin(2 * np.pi * 1 * (t + tc)))
        # CH 2: Beating (Sine 1 + Sine 3)
        waves.append(np.sin(2 * np.pi * 2 * (t + tc)) + 0.5 * np.sin(2 * np.pi * 3.1 * (t + tc)))
        # CH 3: White Noise
        waves.append(np.random.normal(0, 0.5, len(t)))
        # CH 4: Sine + Noise
        waves.append(np.sin(2 * np.pi * 2 * (t + tc)) + np.random.normal(0, 0.2, len(t)))
        # CH 5: Square Wave
        waves.append(np.sign(np.sin(2 * np.pi * 1 * (t + tc))))
        # CH 6: Sawtooth
        waves.append(2 * ((t + tc) * 1 % 1) - 1)
        # CH 7: Random Walk
        waves.append(np.cumsum(np.random.normal(0, 0.1, len(t))))

        for plot_idx, info_root in enumerate([self.graph_info1, self.graph_info2]):
            traces_items = [self.traces1, self.traces2][plot_idx]
            g_type = info_root['graph_combo'].currentText()
            
            for t_idx, ctrl in enumerate(info_root['traces']):
                if not ctrl['active'].isChecked():
                    continue

                curve = traces_items[t_idx]['curve']
                bar = traces_items[t_idx]['bar']
                img = traces_items[t_idx]['img']
                
                ch_a_idx = ctrl['chan_a'].currentIndex()
                ch_b_idx = ctrl['chan_b'].currentIndex()
                
                is_spectral = g_type in ["Amplitude Spectral Density", "Power Spectral Density", "Cross Spectral Density", "Coherence", "Squared Coherence", "Transfer Function"]
                is_dual = g_type in ["Cross Spectral Density", "Coherence", "Squared Coherence", "Transfer Function"]
                is_spec_gram = g_type == "Spectrogram"

                if is_spec_gram:
                    freq_size = 64
                    time_size = len(t)
                    data = np.random.normal(0, 0.5, (time_size, freq_size))
                    peak_f = int(32 + 20 * np.sin(tc + t_idx * 0.5)) # Offset per trace
                    data[:, peak_f-2:peak_f+2] += (5 + t_idx)
                    img.setImage(data, levels=[0, 10])
                    img.setRect(QtCore.QRectF(0, 0, 10, 1000))
                    
                elif is_spectral:
                    freqs = np.logspace(0, 3, len(t))
                    y = 1e-6 * freqs**-1 + np.random.normal(0, 0.05, len(t))**2
                    peak_pos = [10, 50, 100, 200, 300, 400, 500, 800]
                    idx_p = peak_pos[(ch_a_idx + t_idx) % 8]
                    y += 10.0 / (1 + (freqs - idx_p)**2 / 20)
                    
                    if is_dual:
                        if ch_a_idx == ch_b_idx:
                            if "Coherence" in g_type: y = 0.99 + np.random.normal(0, 0.001, len(t))
                            else: y *= 1.2
                        else:
                            if "Coherence" in g_type: y = 0.3 * np.random.rand(len(t))
                            else: y = np.abs(y * 0.5 * np.exp(1j * np.random.rand(len(t))))
                    if "Coherence" in g_type: y = np.clip(y, 0, 1)

                    curve.setData(freqs, y)
                    if bar.isVisible(): bar.setOpts(x=freqs, height=y)
                else:
                    # Time Series - per trace offset for visibility
                    y = waves[ch_a_idx % 8] + (t_idx * 0.5)
                    curve.setData(t, y)
                    if bar.isVisible(): bar.setOpts(x=t, height=y)

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
