
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
    curve1 = plot1.plot(pen='r') # Keep reference
    bar1 = pg.BarGraphItem(x=[0], height=[0], width=0.04, brush='r')
    plot1.addItem(bar1)
    bar1.setVisible(False)
    
    plot2 = pg.PlotWidget(title="Graph 2")
    plot2.showGrid(x=True, y=True)
    plot2.setBackground('w')
    curve2 = plot2.plot(pen='b') # Keep reference
    bar2 = pg.BarGraphItem(x=[0], height=[0], width=0.04, brush='b')
    plot2.addItem(bar2)
    bar2.setVisible(False)

    
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
    
    def add_graph_ctrl(idx, target_curve, target_bar, default_color_idx=0):
        path_frame = QtWidgets.QFrame()
        path_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        pv = QtWidgets.QVBoxLayout(path_frame)
        pv.setContentsMargins(2, 2, 2, 2)
        pv.setSpacing(4)
        
        r1 = QtWidgets.QHBoxLayout()
        for t in ["Style", "X-axis", "Y-axis", "Legend", "Param"]:
             btn = QtWidgets.QPushButton(t); btn.setFlat(True); btn.setStyleSheet("text-align: left; padding: 2px;")
             r1.addWidget(btn)
        pv.addLayout(r1)
        
        r2 = QtWidgets.QHBoxLayout()
        for t in ["Traces", "Range", "Units", "Cursor", "Config"]:
             btn = QtWidgets.QPushButton(t); btn.setFlat(True); btn.setStyleSheet("text-align: left; padding: 2px;")
             r2.addWidget(btn)
        pv.addLayout(r2)
        
        r3 = QtWidgets.QHBoxLayout()
        r3.addWidget(QtWidgets.QLabel("Graph:"))
        r3.addWidget(QtWidgets.QComboBox())
        pv.addLayout(r3)
        
        r4 = QtWidgets.QHBoxLayout()
        for i in range(8):
            btn = QtWidgets.QPushButton(str(i))
            btn.setCheckable(True)
            if i==0: btn.setChecked(True)
            btn.setFixedWidth(24)
            r4.addWidget(btn)
        r4.addStretch(1)
        pv.addLayout(r4)
        
        pv.addWidget(QtWidgets.QCheckBox("Active"))

        g_chan = QtWidgets.QGroupBox("Channels")
        gl = QtWidgets.QGridLayout(g_chan)
        gl.addWidget(QtWidgets.QLabel("A:"), 0, 0)
        gl.addWidget(QtWidgets.QComboBox(), 0, 1)
        gl.addWidget(QtWidgets.QLabel("B:"), 1, 0)
        gl.addWidget(QtWidgets.QComboBox(), 1, 1)
        pv.addWidget(g_chan)

        g_style = QtWidgets.QGroupBox("Style")
        gls = QtWidgets.QGridLayout(g_style)
        
        # --- Value Definitions ---
        colors = [
            ('Red', '#FF0000'), ('Blue', '#0000FF'), ('Green', '#00FF00'),
            ('Black', '#000000'), ('Magenta', '#FF00FF'), ('Cyan', '#00FFFF'),
            ('Yellow', '#FFFF00')
        ]
        
        line_styles = [
            ("Solid", QtCore.Qt.SolidLine), ("Dash", QtCore.Qt.DashLine),
            ("Dot", QtCore.Qt.DotLine), ("DashDot", QtCore.Qt.DashDotLine),
            ("DashDotDot", QtCore.Qt.DashDotDotLine),
        ]
        
        symbols = [
            ("Circle", "o"), ("Square", "s"), ("Triangle", "t"),
            ("Diamond", "d"), ("Plus", "+"),
        ]
        
        fill_patterns = [
            ("Solid", QtCore.Qt.SolidPattern),
            ("Dense1", QtCore.Qt.Dense1Pattern),
            ("Dense3", QtCore.Qt.Dense3Pattern),
            ("Dense5", QtCore.Qt.Dense5Pattern),
            ("Hor", QtCore.Qt.HorPattern),
            ("Ver", QtCore.Qt.VerPattern),
            ("Cross", QtCore.Qt.CrossPattern),
            ("BDiag", QtCore.Qt.BDiagPattern),
            ("FDiag", QtCore.Qt.FDiagPattern),
            ("DiagCross", QtCore.Qt.DiagCrossPattern),
        ]

        # ====================
        # LINE ROW
        # ====================
        chk_line = QtWidgets.QCheckBox("Line"); chk_line.setChecked(True)
        gls.addWidget(chk_line, 0, 0)
        
        l_color_combo = QtWidgets.QComboBox(); l_color_combo.setFixedWidth(50)
        for i, (name, hex_code) in enumerate(colors):
            l_color_combo.addItem("")
            l_color_combo.setItemData(i, QtGui.QColor(hex_code), QtCore.Qt.BackgroundRole)
        
        l_style_combo = QtWidgets.QComboBox(); l_style_combo.setFixedWidth(90)
        for s_name, s_val in line_styles:
            l_style_combo.addItem(s_name, s_val)

        l_width_sb = _small_spin_int(1,10, 40); l_width_sb.setValue(1)

        gls.addWidget(l_color_combo, 0, 1)
        gls.addWidget(l_style_combo, 0, 2) 
        gls.addWidget(l_width_sb, 0, 3) 

        # ====================
        # SYMBOL ROW
        # ====================
        chk_symbol = QtWidgets.QCheckBox("Symbol"); chk_symbol.setChecked(False)
        gls.addWidget(chk_symbol, 1, 0)
        
        s_color_combo = QtWidgets.QComboBox(); s_color_combo.setFixedWidth(50)
        for i, (name, hex_code) in enumerate(colors):
            s_color_combo.addItem("")
            s_color_combo.setItemData(i, QtGui.QColor(hex_code), QtCore.Qt.BackgroundRole)
        
        s_style_combo = QtWidgets.QComboBox(); s_style_combo.setFixedWidth(90)
        for s_name, s_val in symbols:
            s_style_combo.addItem(s_name, s_val)

        # Use spin box for size
        s_size_sb = _small_spin_dbl(decimals=1, width=40, min_val=1, max_val=50)
        s_size_sb.setValue(5)

        gls.addWidget(s_color_combo, 1, 1)
        gls.addWidget(s_style_combo, 1, 2) 
        gls.addWidget(s_size_sb, 1, 3) 

        # ====================
        # BAR ROW
        # ====================
        chk_bar = QtWidgets.QCheckBox("Bar")
        gls.addWidget(chk_bar, 2, 0)
        
        b_color_combo = QtWidgets.QComboBox(); b_color_combo.setFixedWidth(50)
        for i, (name, hex_code) in enumerate(colors):
            b_color_combo.addItem("")
            b_color_combo.setItemData(i, QtGui.QColor(hex_code), QtCore.Qt.BackgroundRole)
            
        b_style_combo = QtWidgets.QComboBox(); b_style_combo.setFixedWidth(90)
        for p_name, p_val in fill_patterns:
            b_style_combo.addItem(p_name, p_val)

        b_width_sb = _small_spin_dbl(decimals=2, width=40, min_val=0.01, max_val=10.0, step=0.01)
        b_width_sb.setValue(0.04)

        gls.addWidget(b_color_combo, 2, 1) 
        gls.addWidget(b_style_combo, 2, 2) 
        gls.addWidget(b_width_sb, 2, 3) 

        # ====================
        # LOGIC
        # ====================
        
        def toggle_line(state):
            if state:
                chk_bar.blockSignals(True)
                chk_bar.setChecked(False)
                chk_bar.blockSignals(False)
            update_style()
            
        def toggle_bar(state):
            if state:
                chk_line.blockSignals(True)
                chk_line.setChecked(False)
                chk_line.blockSignals(False)
            update_style()
            
        chk_line.toggled.connect(toggle_line)
        chk_bar.toggled.connect(toggle_bar)

        def update_style(idx=None):
            # --- Line ---
            l_c_idx = l_color_combo.currentIndex()
            if l_c_idx < 0: l_c_idx = 0
            _, l_hex_code = colors[l_c_idx]
            
            # --- Bar ---
            b_c_idx = b_color_combo.currentIndex()
            if b_c_idx < 0: b_c_idx = 0
            _, b_hex_code = colors[b_c_idx]

            if chk_line.isChecked():
                # Line UI Update
                l_color_combo.setStyleSheet(f"background-color: {l_hex_code}; selection-background-color: {l_hex_code};")
                
                st_idx = l_style_combo.currentIndex()
                if st_idx < 0: l_style_code = QtCore.Qt.SolidLine
                else: l_style_code = l_style_combo.itemData(st_idx)
                
                l_width = l_width_sb.value()
                
                pen = pg.mkPen(color=l_hex_code, width=l_width, style=l_style_code)
                target_curve.setPen(pen)
                target_bar.setVisible(False)
                
                # Visual feedback even if inactive
                b_color_combo.setStyleSheet(f"background-color: {b_hex_code}; selection-background-color: {b_hex_code};")
                
            elif chk_bar.isChecked():
                # Bar UI Update
                b_color_combo.setStyleSheet(f"background-color: {b_hex_code}; selection-background-color: {b_hex_code};")
                
                target_curve.setPen(None)
                target_bar.setVisible(True)
                
                # Pattern
                p_idx = b_style_combo.currentIndex()
                if p_idx < 0: pattern = QtCore.Qt.SolidPattern
                else: pattern = b_style_combo.itemData(p_idx)
                
                # Create Brush
                brush = QtGui.QBrush(QtGui.QColor(b_hex_code), pattern)
                
                width = b_width_sb.value()
                target_bar.setOpts(brush=brush, width=width)
                
                # Visual feedback even if inactive
                l_color_combo.setStyleSheet(f"background-color: {l_hex_code}; selection-background-color: {l_hex_code};")

            else:
                target_curve.setPen(None)
                target_bar.setVisible(False)
                l_color_combo.setStyleSheet(f"background-color: {l_hex_code}; selection-background-color: {l_hex_code};")
                b_color_combo.setStyleSheet(f"background-color: {b_hex_code}; selection-background-color: {b_hex_code};")

            # --- Symbol ---
            if chk_symbol.isChecked():
                sc_idx = s_color_combo.currentIndex()
                if sc_idx < 0: sc_idx = 0
                _, s_hex_code = colors[sc_idx]
                s_color_combo.setStyleSheet(f"background-color: {s_hex_code}; selection-background-color: {s_hex_code};")

                ss_idx = s_style_combo.currentIndex()
                if ss_idx < 0: sym_code = 'o'
                else: sym_code = s_style_combo.itemData(ss_idx)
                
                s_size = s_size_sb.value()
                
                target_curve.setSymbol(sym_code)
                target_curve.setSymbolSize(s_size)
                target_curve.setSymbolBrush(s_hex_code)
                target_curve.setSymbolPen(s_hex_code)
            else:
                target_curve.setSymbol(None)
                sc_idx = s_color_combo.currentIndex()
                _, s_hex_code = colors[sc_idx]
                s_color_combo.setStyleSheet(f"background-color: {s_hex_code}; selection-background-color: {s_hex_code};")

        # Connect signals
        l_color_combo.currentIndexChanged.connect(lambda: update_style())
        l_style_combo.currentIndexChanged.connect(lambda: update_style())
        l_width_sb.valueChanged.connect(lambda: update_style())

        chk_symbol.toggled.connect(lambda: update_style())
        s_color_combo.currentIndexChanged.connect(lambda: update_style())
        s_style_combo.currentIndexChanged.connect(lambda: update_style())
        s_size_sb.valueChanged.connect(lambda: update_style())
        
        b_color_combo.currentIndexChanged.connect(lambda: update_style())
        b_style_combo.currentIndexChanged.connect(lambda: update_style())
        b_width_sb.valueChanged.connect(lambda: update_style())
        
        l_color_combo.setCurrentIndex(default_color_idx)
        s_color_combo.setCurrentIndex(default_color_idx)
        b_color_combo.setCurrentIndex(default_color_idx)
        
        update_style() 

        pv.addWidget(g_style)
        lv.addWidget(path_frame)

    add_graph_ctrl(1, curve1, bar1, default_color_idx=0)
    line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.HLine)
    lv.addWidget(line)
    add_graph_ctrl(2, curve2, bar2, default_color_idx=1)
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

    return tab, curve1, curve2, bar1, bar2


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
        
        # Result tab returns widgets now
        res_tab, self.curve1, self.curve2, self.bar1, self.bar2 = create_result_tab()
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
        
        # Reset data
        self.curve1.setData([], [])
        self.curve2.setData([], [])
        self.bar1.setOpts(x=[0], height=[0])
        self.bar2.setOpts(x=[0], height=[0])
        self.time_counter = 0.0

    def update_graphs(self):
        self.time_counter += 0.05
        
        # Generate dummy data
        # Wave 1: Sine + Noise
        y1 = np.sin(2 * np.pi * (self.x_data + self.time_counter)) + np.random.normal(0, 0.1, len(self.x_data))
        
        # Wave 2: Modulated Cosine
        y2 = np.cos(4 * np.pi * (self.x_data - self.time_counter)) * np.sin(self.x_data)
        
        self.curve1.setData(self.x_data, y1)
        self.curve2.setData(self.x_data, y2)
        
        self.bar1.setOpts(x=self.x_data, height=y1)
        self.bar2.setOpts(x=self.x_data, height=y2)

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
