from __future__ import annotations

from typing import Any, Callable, Optional

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from .graph_panel import GraphPanel


def _h_spacer():
    s = QtWidgets.QSpacerItem(
        10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
    )
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
    w.setKeyboardTracking(False)
    return w


def _create_group(title, layout_type="grid"):
    gb = QtWidgets.QGroupBox(title)
    if layout_type == "grid":
        layout = QtWidgets.QGridLayout(gb)
    elif layout_type == "v":
        layout = QtWidgets.QVBoxLayout(gb)
    else:
        layout = QtWidgets.QHBoxLayout(gb)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(6)
    return gb, layout


def create_input_tab():
    tab = QtWidgets.QWidget()
    vbox = QtWidgets.QVBoxLayout(tab)
    vbox.setSpacing(10)
    vbox.setContentsMargins(10, 10, 10, 10)

    # -- Data Source Selection --
    gb_ds, l_ds = _create_group("Data Source Selection", "h")
    ds_combo = QtWidgets.QComboBox()
    ds_combo.setVisible(False)
    ds_combo.addItems(["NDS", "NDS2", "FILE", "Simulation"])

    rb1 = QtWidgets.QRadioButton("Online system")
    rb1.setChecked(True)
    rb2 = QtWidgets.QRadioButton("User NDS")
    rb3 = QtWidgets.QRadioButton("NDS2")
    rb4 = QtWidgets.QRadioButton("LiDaX")
    rb5 = QtWidgets.QRadioButton("Simulation")

    def update_ds():
        if rb3.isChecked():
            ds_combo.setCurrentText("NDS2")
        elif rb5.isChecked():
            ds_combo.setCurrentText("Simulation")
        else:
            ds_combo.setCurrentText("NDS")  # Maps others to NDS for now

    for r in [rb1, rb2, rb3, rb4, rb5]:
        r.toggled.connect(update_ds)
        l_ds.addWidget(r)

    l_ds.addSpacing(10)
    chk_pcaudio = QtWidgets.QCheckBox("PC Audio")
    l_ds.addWidget(chk_pcaudio)

    l_ds.addSpacing(20)
    chk_reconnect = QtWidgets.QCheckBox("Reconnect")
    l_ds.addWidget(chk_reconnect)
    l_ds.addStretch(1)
    btn_clear = QtWidgets.QPushButton("Clear cache")
    l_ds.addWidget(btn_clear)
    vbox.addWidget(gb_ds)

    # -- NDS Selection --
    gb_nds, l_nds = _create_group("NDS Selection", "h")
    l_nds.addWidget(QtWidgets.QLabel("Server:"))
    cb_serv = QtWidgets.QComboBox()
    cb_serv.addItems(["k1nds1", "localhost"])
    cb_serv.setMinimumWidth(250)
    cb_serv.setEditable(True)
    l_nds.addWidget(cb_serv)
    l_nds.addWidget(QtWidgets.QLabel("Port:"))
    sb_port = _small_spin_int(0, 65535, width=80)
    sb_port.setValue(8088)
    l_nds.addWidget(sb_port)
    l_nds.addStretch(1)
    vbox.addWidget(gb_nds)

    # -- NDS2 Selection --
    gb_nds2, l_nds2 = _create_group("NDS2 Selection", "v")
    h_top2 = QtWidgets.QHBoxLayout()
    h_top2.addWidget(QtWidgets.QLabel("Server:"))
    cb_serv2 = QtWidgets.QComboBox()
    cb_serv2.addItems(["k1nds2"])
    cb_serv2.setMinimumWidth(250)
    cb_serv2.setEditable(True)
    h_top2.addWidget(cb_serv2)
    h_top2.addWidget(QtWidgets.QLabel("Port:"))
    sb_port2 = _small_spin_int(0, 65535, width=80)
    sb_port2.setValue(31200)
    h_top2.addWidget(sb_port2)
    h_top2.addWidget(QtWidgets.QLabel("Epoch:"))
    cb_epoch = QtWidgets.QComboBox()
    cb_epoch.addItems(["User specified"])
    cb_epoch.setMinimumWidth(150)
    h_top2.addWidget(cb_epoch)
    h_top2.addStretch(1)
    l_nds2.addLayout(h_top2)

    h_epochs = QtWidgets.QHBoxLayout()
    # Epoch Start
    gb_start, l_start = _create_group("Epoch Start", "grid")
    l_start.addWidget(QtWidgets.QLabel("GPS:"), 0, 0)
    l_start.addWidget(_small_spin_int(0, 2000000000, 100), 0, 1)
    l_start.addWidget(QtWidgets.QLabel("sec"), 0, 2)
    l_start.addWidget(QtWidgets.QLabel("Date/Time:"), 1, 0)
    de_start = QtWidgets.QDateEdit()
    de_start.setDisplayFormat("dd/MM/yyyy")
    l_start.addWidget(de_start, 1, 1)
    te_start = QtWidgets.QTimeEdit()
    te_start.setDisplayFormat("HH:mm:ss")
    l_start.addWidget(te_start, 1, 2)
    l_start.addWidget(QtWidgets.QLabel("hh:mm:ss UTC"), 1, 3)
    h_epochs.addWidget(gb_start)

    # Epoch Stop
    gb_stop, l_stop = _create_group("Epoch Stop", "grid")
    l_stop.addWidget(QtWidgets.QLabel("GPS:"), 0, 0)
    sb_gps_stop = _small_spin_int(0, 2000000000, 100)
    sb_gps_stop.setValue(1451117047)
    l_stop.addWidget(sb_gps_stop, 0, 1)
    l_stop.addWidget(QtWidgets.QLabel("sec"), 0, 2)
    l_stop.addWidget(QtWidgets.QLabel("Date/Time:"), 1, 0)
    de_stop = QtWidgets.QDateEdit()
    de_stop.setDisplayFormat("dd/MM/yyyy")
    l_stop.addWidget(de_stop, 1, 1)
    te_stop = QtWidgets.QTimeEdit()
    te_stop.setDisplayFormat("HH:mm:ss")
    l_stop.addWidget(te_stop, 1, 2)
    l_stop.addWidget(QtWidgets.QLabel("hh:mm:ss UTC"), 1, 3)
    h_epochs.addWidget(gb_stop)
    l_nds2.addLayout(h_epochs)
    vbox.addWidget(gb_nds2)

    # -- LiDaX Data Source --
    gb_lidx, l_lidx = _create_group("LiDaX Data Source", "grid")
    l_lidx.addWidget(QtWidgets.QLabel("Server:"), 0, 0)
    l_lidx.addWidget(QtWidgets.QComboBox(), 0, 1)
    cb_lfs = QtWidgets.QComboBox()
    cb_lfs.addItems(["Local file system"])
    l_lidx.addWidget(cb_lfs, 0, 2)
    l_lidx.addWidget(QtWidgets.QPushButton("Add..."), 0, 3)
    l_lidx.addWidget(QtWidgets.QLabel("Channels:"), 0, 4)
    l_lidx.addWidget(QtWidgets.QLineEdit(), 0, 5)
    l_lidx.addWidget(QtWidgets.QPushButton("Select..."), 0, 6)
    l_lidx.addWidget(QtWidgets.QLabel("UDN:"), 1, 0)
    l_lidx.addWidget(QtWidgets.QComboBox(), 1, 1, 1, 2)
    l_lidx.addWidget(QtWidgets.QPushButton("More..."), 1, 3)
    l_lidx.addWidget(QtWidgets.QLabel("Keep:"), 1, 4)
    te_keep = QtWidgets.QTimeEdit()
    te_keep.setDisplayFormat("HH:mm")
    l_lidx.addWidget(te_keep, 1, 5)
    l_lidx.addWidget(QtWidgets.QLabel("hh:mm"), 1, 6)
    l_lidx.addWidget(QtWidgets.QPushButton("Staging..."), 1, 7)
    vbox.addWidget(gb_lidx)

    # Hidden or integrated NDS Window control
    nds_win_spin = _small_spin_int(min_val=1, max_val=3600, width=60)
    nds_win_spin.setValue(30)
    nds_win_spin.setVisible(False)
    sim_dur_spin = _small_spin_dbl(1, 60, min_val=0.1, max_val=1e6)
    sim_dur_spin.setValue(10.0)
    sim_dur_spin.setVisible(False)

    vbox.addStretch(1)

    controls: dict[str, Any] = {
        "ds_combo": ds_combo,
        "sim_dur": sim_dur_spin,
        "nds_win": nds_win_spin,
        "nds_server": cb_serv,
        "nds_port": sb_port,
        "nds2_server": cb_serv2,
        "nds2_port": sb_port2,
        "reconnect": chk_reconnect,
        "clear_cache": btn_clear,
        "pcaudio": chk_pcaudio,
    }
    return tab, controls


def create_measurement_tab():
    tab = QtWidgets.QWidget()
    outer = QtWidgets.QVBoxLayout(tab)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(10)

    controls: dict[str, Any] = {}

    # Group: Measurement
    gb_meas, vb = _create_group("Measurement", "v")
    hbox_m = QtWidgets.QHBoxLayout()
    hbox_m.addWidget(QtWidgets.QRadioButton("Fourier Tools", checked=True))
    hbox_m.addWidget(QtWidgets.QRadioButton("Swept Sine Response"))
    hbox_m.addWidget(QtWidgets.QRadioButton("Sine Response"))
    hbox_m.addWidget(QtWidgets.QRadioButton("Triggered Time Response"))
    hbox_m.addStretch(1)
    vb.addLayout(hbox_m)
    outer.addWidget(gb_meas)

    # Group: Measurement Channels
    gb_chan, v_chan = _create_group("Measurement Channels", "v")

    # Radio buttons for channel banks
    hbox_banks = QtWidgets.QHBoxLayout()
    banks = [
        "Channels 0 to 15",
        "Channels 16 to 31",
        "Channels 32 to 47",
        "Channels 48 to 63",
        "Channels 64 to 79",
        "Channels 80 to 95",
    ]
    rb_list = []
    for i, b in enumerate(banks):
        rb = QtWidgets.QRadioButton(b)
        if i == 0:
            rb.setChecked(True)
        hbox_banks.addWidget(rb)
        rb_list.append(rb)
    hbox_banks.addStretch(1)

    btn_browse = QtWidgets.QPushButton("Channel Browser...")
    hbox_banks.addWidget(btn_browse)

    v_chan.addLayout(hbox_banks)

    # Channel State Management (96 channels)
    # Default SIM channels for first 8
    # Default SIM channels Removed
    # channel_states initialized empty
    channel_states = [{"active": False, "name": ""} for _ in range(96)]

    # Callback for external updates (Main Window)
    meas_callback: Optional[Callable[[], Any]] = None

    def on_widget_change():
        # Save current state immediately to model
        # Find which bank is active
        active_bank = 0
        for i, rb in enumerate(rb_list):
            if rb.isChecked():
                active_bank = i
                break

        save_current_bank(active_bank)

        if meas_callback:
            meas_callback()

    def save_current_bank(bank_idx):
        start_ch = bank_idx * 16
        for i in range(16):  # 0-15 in grid
            ch_idx = start_ch + i
            # Grid logic: i=0..7 is col 1,2; i=8..15 is col 4,5
            # We stored them in flat list chan_grid_refs
            lbl, chk, cmb, btn = chan_grid_refs[i]
            channel_states[ch_idx]["active"] = chk.isChecked()
            channel_states[ch_idx]["name"] = cmb.text()

    def load_bank(bank_idx):
        start_ch = bank_idx * 16
        # Block signals to prevent recursion during load
        for ref in chan_grid_refs:
            ref[1].blockSignals(True)
            ref[2].blockSignals(True)

        for i in range(16):
            ch_idx = start_ch + i
            state = channel_states[ch_idx]
            lbl, chk, cmb, btn = chan_grid_refs[i]

            lbl.setText(str(ch_idx))
            chk.setChecked(state["active"])
            cmb.setText(state["name"])

        for ref in chan_grid_refs:
            ref[1].blockSignals(False)
            ref[2].blockSignals(False)

    def set_all_channels(new_channels):
        """
        External method to bulk update channel states.
        new_channels: list of dict {'name': str, 'active': bool}
        """
        nonlocal channel_states

        # Reset all to defaults first
        for s in channel_states:
            s["name"] = ""
            s["active"] = False

        # Update with provided info, up to 96
        count = min(len(new_channels), 96)
        for i in range(count):
            item = new_channels[i]
            channel_states[i]["name"] = item.get("name", "")
            channel_states[i]["active"] = item.get("active", True)

        # Refresh current view
        # We need to know current bank idx.
        # But we don't have easy access to 'active_bank' var here outside on_widget_change.
        # So we iterate radio buttons to find checked common one
        curr_bank = 0
        for i, rb in enumerate(rb_list):
            if rb.isChecked():
                curr_bank = i
                break
        load_bank(curr_bank)

        # Trigger external update if any
        if meas_callback:
            meas_callback()

    # Grid of channels
    c_grid = QtWidgets.QGridLayout()
    c_grid.setContentsMargins(0, 0, 0, 0)
    c_grid.setHorizontalSpacing(15)

    chan_grid_refs = []

    # Create 16 sets of widgets first
    for i in range(16):
        lbl = QtWidgets.QLabel(str(i))
        chk = QtWidgets.QCheckBox()
        cmb = QtWidgets.QLineEdit()
        cmb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn = QtWidgets.QPushButton("▼")
        btn.setFixedWidth(25)
        chk.toggled.connect(on_widget_change)
        cmb.textChanged.connect(on_widget_change)
        chan_grid_refs.append((lbl, chk, cmb, btn))

    # Add to layout
    c_grid.setHorizontalSpacing(4)
    c_grid.setColumnMinimumWidth(3, 20)  # Middle spacer

    for i in range(8):
        # Left: 0-7
        l_widgets = chan_grid_refs[i]
        c_grid.addWidget(l_widgets[0], i, 0)
        c_grid.addWidget(l_widgets[1], i, 1)

        l_box = QtWidgets.QHBoxLayout()
        l_box.setSpacing(0)
        l_box.setContentsMargins(0, 0, 0, 0)
        l_box.addWidget(l_widgets[2])
        l_box.addWidget(l_widgets[3])
        c_grid.addLayout(l_box, i, 2)

        # Right: 8-15
        r_widgets = chan_grid_refs[i + 8]
        c_grid.addWidget(r_widgets[0], i, 4)
        c_grid.addWidget(r_widgets[1], i, 5)

        r_box = QtWidgets.QHBoxLayout()
        r_box.setSpacing(0)
        r_box.setContentsMargins(0, 0, 0, 0)
        r_box.addWidget(r_widgets[2])
        r_box.addWidget(r_widgets[3])
        c_grid.addLayout(r_box, i, 6)

    v_chan.addLayout(c_grid)
    outer.addWidget(gb_chan)

    def update_chan_bank(idx):
        # Save previous bank (Note: checking which radio is checked is tricky if we are in the middle of a switch)
        # Better: Since we only support 1 bank visible, we can just save the *current usage* before loading new.
        # But `load_bank` overwrites widgets.
        # The radio button `toggled` emits for the one turning OFF and the one turning ON.
        # We only want to handle the one turning ON.
        load_bank(idx)

    # Initial load
    load_bank(0)

    # We need to hook up radio buttons to save/load.
    # When button X is clicked:
    # 1. We should have already saved the previous state.
    #    Issue: `toggled` happens. We don't easily know "previous".
    #    Solution: The `on_widget_change` handles saving real-time. So the model `channel_states` is ALWAYS up to date for the *visible* bank.
    #    So we only need `load_bank` when switching.

    def get_bank_offset():
        for i, rb in enumerate(rb_list):
            if rb.isChecked():
                return i * 16
        return 0

    [
        rb.toggled.connect(
            lambda checked, idx=i: update_chan_bank(idx) if checked else None
        )
        for i, rb in enumerate(rb_list)
    ]

    controls["channel_states"] = channel_states

    def set_callback(cb):
        nonlocal meas_callback
        meas_callback = cb

    controls["set_change_callback"] = set_callback
    controls["set_all_channels"] = set_all_channels
    controls["btn_browse"] = btn_browse
    controls["grid_refs"] = chan_grid_refs
    controls["get_bank_offset"] = get_bank_offset

    # Group: Fourier Tools
    gb_fft, g_fft = _create_group("Fourier Tools", "grid")

    def _add_param(row, col, label, spin_widget, unit=None, key=None):
        g_fft.addWidget(QtWidgets.QLabel(label), row, col)
        g_fft.addWidget(spin_widget, row, col + 1)
        if unit:
            g_fft.addWidget(QtWidgets.QLabel(unit), row, col + 2)
        if key:
            controls[key] = spin_widget
        return col + 3

    # Row 0
    c = 0
    c = _add_param(0, c, "Start:", _small_spin_dbl(width=80), "Hz", "start_freq")
    c = _add_param(
        0, c, "Stop:", _small_spin_dbl(width=80, max_val=1e5), "Hz", "stop_freq"
    )
    controls["stop_freq"].setValue(900)
    c = _add_param(0, c, "BW:", _small_spin_dbl(width=80), "Hz", "bw")
    controls["bw"].setValue(1)
    ts_spin = _small_spin_dbl(width=80)
    c = _add_param(0, c, "TimeSpan:", ts_spin, "s")
    controls["time_span"] = ts_spin

    # Settling Time
    sb_settling = _small_spin_dbl(width=80)
    sb_settling.setValue(10.0)
    c = _add_param(0, c, "Settling Time:", sb_settling, "%", "settling_time")

    # Ramp Down
    sb_ramp_down = _small_spin_dbl(width=80)
    sb_ramp_down.setValue(1.0)
    c = _add_param(0, c, "Ramp Down:", sb_ramp_down, "Sec", "ramp_down")

    # Ramp Up
    c = _add_param(0, c, "Ramp Up:", _small_spin_dbl(width=80), "", "ramp_up")

    # Row 1
    g_fft.addWidget(QtWidgets.QLabel("Window:"), 1, 0)
    cb_win = QtWidgets.QComboBox()
    cb_win.addItems(["Hann", "Flattop", "Uniform"])
    controls["window"] = cb_win
    g_fft.addWidget(cb_win, 1, 1, 1, 2)

    g_fft.addWidget(QtWidgets.QLabel("Overlap:"), 1, 3)
    sb_ov = _small_spin_dbl(width=80)
    sb_ov.setValue(50)
    controls["overlap"] = sb_ov
    g_fft.addWidget(sb_ov, 1, 4)
    g_fft.addWidget(QtWidgets.QLabel("%"), 1, 5)

    chk_rm = QtWidgets.QCheckBox("Remove mean", checked=True)
    controls["remove_mean"] = chk_rm
    g_fft.addWidget(chk_rm, 1, 6, 1, 2)

    g_fft.addWidget(QtWidgets.QLabel("Number of A channels:"), 1, 9)
    sb_num_a = _small_spin_int(width=60)
    controls["num_a_channels"] = sb_num_a
    g_fft.addWidget(sb_num_a, 1, 10)

    # Row 2
    g_fft.addWidget(QtWidgets.QLabel("Averages:"), 2, 0)
    sb_avg = _small_spin_int(width=80)
    sb_avg.setValue(10)
    controls["averages"] = sb_avg
    g_fft.addWidget(sb_avg, 2, 1)

    l_avg = QtWidgets.QHBoxLayout()
    l_avg.addWidget(QtWidgets.QLabel("Average Type:"))
    rb_fixed = QtWidgets.QRadioButton("Fixed", checked=True)
    rb_exp = QtWidgets.QRadioButton("Exponential")
    rb_accum = QtWidgets.QRadioButton("Accumulative")
    controls.update(
        {"avg_type_fixed": rb_fixed, "avg_type_exp": rb_exp, "avg_type_accum": rb_accum}
    )
    l_avg.addWidget(rb_fixed)
    l_avg.addWidget(rb_exp)
    l_avg.addWidget(rb_accum)
    l_avg.addStretch(1)
    g_fft.addLayout(l_avg, 2, 3, 1, 6)

    g_fft.addWidget(QtWidgets.QLabel("Burst Noise Quiet Time"), 2, 12)
    sb_burst = _small_spin_dbl(decimals=2, width=80)
    controls["burst_quiet_time"] = sb_burst
    g_fft.addWidget(sb_burst, 2, 13)
    g_fft.addWidget(QtWidgets.QLabel("sec"), 2, 14)

    outer.addWidget(gb_fft)

    # Group: Start Time (Visual Only for now)
    gb_time, g_time = _create_group("Start Time", "grid")
    rb_now = QtWidgets.QRadioButton("Now", checked=True)
    g_time.addWidget(rb_now, 0, 0)
    rb_gps = QtWidgets.QRadioButton("GPS:")
    g_time.addWidget(rb_gps, 1, 0)
    controls["rb_gps"] = rb_gps

    gps_spin = _small_spin_int(width=100, min_val=0, max_val=2000000000)
    controls["start_gps"] = gps_spin
    g_time.addWidget(gps_spin, 1, 1)
    g_time.addWidget(QtWidgets.QLabel("sec"), 1, 2)
    g_time.addWidget(_small_spin_int(width=80, min_val=0), 1, 3)
    g_time.addWidget(QtWidgets.QLabel("nsec"), 1, 4)
    rb_dt = QtWidgets.QRadioButton("Date/time:")
    g_time.addWidget(rb_dt, 2, 0)

    date_edit = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
    time_edit = QtWidgets.QTimeEdit(QtCore.QTime.currentTime())
    controls["start_date"] = date_edit
    controls["start_time"] = time_edit

    # Sync Logic
    def sync_gps_to_dt():
        val = gps_spin.value()
        try:
            from astropy.time import Time

            t = Time(val, format="gps", scale="utc").datetime
            date_edit.blockSignals(True)
            time_edit.blockSignals(True)
            date_edit.setDate(t.date())
            time_edit.setTime(t.time())
            date_edit.blockSignals(False)
            time_edit.blockSignals(False)
        except (TypeError, ValueError):
            pass

    def sync_dt_to_gps():
        d = date_edit.date().toPyDate()
        t = time_edit.time().toPyTime()
        import datetime

        dt = datetime.datetime.combine(d, t)
        try:
            from astropy.time import Time

            gps = int(Time(dt, format="datetime", scale="utc").gps)
            gps_spin.blockSignals(True)
            gps_spin.setValue(gps)
            gps_spin.blockSignals(False)
        except (TypeError, ValueError):
            pass

    gps_spin.valueChanged.connect(sync_gps_to_dt)
    date_edit.dateChanged.connect(sync_dt_to_gps)
    time_edit.timeChanged.connect(sync_dt_to_gps)

    # Initial Sync (GPS to DT, assuming GPS 0 or default is not helpful, so maybe DT to GPS?)
    # Actually, GPS spin defaults to 0. DT defaults to Now.
    # Let's sync DT -> GPS initially so GPS shows Now.
    sync_dt_to_gps()

    g_time.addWidget(date_edit, 2, 1)
    g_time.addWidget(time_edit, 2, 2)
    g_time.addWidget(QtWidgets.QLabel("UTC"), 2, 3)
    g_time.addWidget(QtWidgets.QRadioButton("In the future:"), 0, 6)
    g_time.addWidget(QtWidgets.QTimeEdit(), 0, 7)
    g_time.addWidget(QtWidgets.QLabel("hh:mm:ss"), 0, 8)
    g_time.addWidget(QtWidgets.QRadioButton("In the past:"), 1, 6)
    g_time.addWidget(QtWidgets.QTimeEdit(), 1, 7)
    g_time.addWidget(QtWidgets.QLabel("hh:mm:ss"), 1, 8)
    g_time.addWidget(QtWidgets.QPushButton("Time now"), 2, 6)
    g_time.addWidget(QtWidgets.QPushButton("Lookup..."), 2, 7)
    l_slow = QtWidgets.QHBoxLayout()
    l_slow.addStretch(1)
    l_slow.addWidget(QtWidgets.QLabel("Slow down:"))
    l_slow.addWidget(_small_spin_int(width=60, min_val=0))
    l_slow.addWidget(QtWidgets.QLabel("sec/avrg."))
    g_time.addLayout(l_slow, 2, 9, 1, 4)
    outer.addWidget(gb_time)

    # Group: Measurement Information
    gb_info, g_info = _create_group("Measurement Information", "grid")
    g_info.addWidget(QtWidgets.QLabel("Measurement Time:"), 0, 0)
    meas_time_edit = QtWidgets.QLineEdit("06/01/1980 00:00:00 UTC")
    controls["meas_time_str"] = meas_time_edit
    g_info.addWidget(meas_time_edit, 0, 1)
    g_info.addWidget(QtWidgets.QLabel("Comment / Description:"), 0, 2)
    outer.addWidget(gb_info)
    txt_comment = QtWidgets.QLineEdit()
    txt_comment.setMinimumHeight(30)
    g_info.addWidget(txt_comment, 1, 0, 1, 3)

    outer.addStretch(1)
    return tab, controls


def create_excitation_tab():
    tab = QtWidgets.QWidget()
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    tab_inner = QtWidgets.QWidget()
    scroll.setWidget(tab_inner)

    # Outer layout
    tab_layout = QtWidgets.QVBoxLayout(tab)
    tab_layout.setContentsMargins(0, 0, 0, 0)
    tab_layout.addWidget(scroll)

    outer = QtWidgets.QVBoxLayout(tab_inner)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(10)

    controls: dict[str, Any] = {}
    controls["panels"] = []

    # Waveform items
    waveforms = [
        "None",
        "Sine",
        "Square",
        "Ramp",
        "Triangle",
        "Impulse",
        "Offset",
        "Noise (Gauss)",
        "Noise (Uniform)",
        "Arbitrary",
        "Sweep (linear)",
        "Sweep (log)",
    ]

    # CS Group
    gb_cs, l_cs = _create_group("Channel Selection", "h")
    rb_0_3 = QtWidgets.QRadioButton("Channels 0 to 3", checked=True)
    l_cs.addWidget(rb_0_3)
    l_cs.addWidget(QtWidgets.QRadioButton("Channels 4 to 7"))
    l_cs.addWidget(QtWidgets.QRadioButton("Channels 8 to 11"))
    l_cs.addWidget(QtWidgets.QRadioButton("Channels 12 to 15"))
    l_cs.addWidget(QtWidgets.QRadioButton("Channels 16 to 19"))
    l_cs.addStretch(1)
    outer.addWidget(gb_cs)

    # 4 Channel panels
    panels = []
    for i in range(4):
        gb, gl = _create_group(f"Channel {i}", "grid")

        # Row 0
        chk_active = QtWidgets.QCheckBox("Active")
        gl.addWidget(chk_active, 0, 0)

        gl.addWidget(QtWidgets.QLabel("Excitation Channel:"), 0, 1)
        cb_ex = QtWidgets.QComboBox()
        cb_ex.setEditable(True)
        cb_ex.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        cb_ex.setEditText(f"Excitation-{i}")  # Default unique name
        gl.addWidget(cb_ex, 0, 2, 1, 10)  # Span wide

        # Row 1
        gl.addWidget(QtWidgets.QLabel("Readback Channel:"), 1, 0)

        h_rb = QtWidgets.QHBoxLayout()
        h_rb.addWidget(QtWidgets.QRadioButton("Default", checked=True))
        h_rb.addWidget(QtWidgets.QRadioButton("None"))
        h_rb.addWidget(QtWidgets.QRadioButton("User:"))
        gl.addLayout(h_rb, 1, 1)

        gl.addWidget(QtWidgets.QComboBox(), 1, 2, 1, 10)  # Span wide

        # Row 2
        gl.addWidget(QtWidgets.QLabel("Waveform:"), 2, 0)
        cb_wave = QtWidgets.QComboBox()
        cb_wave.addItems(waveforms)
        gl.addWidget(cb_wave, 2, 1)  # None

        gl.addWidget(QtWidgets.QLabel("Waveform File:"), 2, 2)
        gl.addWidget(QtWidgets.QLineEdit(), 2, 3, 1, 8)
        gl.addWidget(QtWidgets.QPushButton("Choose..."), 2, 11)

        # Row 3
        gl.addWidget(QtWidgets.QLabel("Frequency:"), 3, 0)

        h_vals = QtWidgets.QHBoxLayout()
        sb_freq = _small_spin_dbl(width=80, max_val=1e5)
        sb_freq.setValue(100.0)
        h_vals.addWidget(sb_freq)
        h_vals.addWidget(QtWidgets.QLabel("Hz"))

        h_vals.addWidget(QtWidgets.QLabel("Amplitude:"))
        sb_amp = _small_spin_dbl(width=80, max_val=1e9)
        sb_amp.setValue(1.0)  # Default 1.0 for visibility
        h_vals.addWidget(sb_amp)

        h_vals.addWidget(QtWidgets.QLabel("Offset:"))
        sb_off = _small_spin_dbl(width=80, max_val=1e9)
        h_vals.addWidget(sb_off)

        h_vals.addWidget(QtWidgets.QLabel("Phase:"))
        sb_phase = _small_spin_dbl(width=80)
        h_vals.addWidget(sb_phase)
        h_vals.addWidget(QtWidgets.QLabel("deg"))

        h_vals.addWidget(QtWidgets.QLabel("Ratio:"))
        sb_ratio = _small_spin_dbl(width=80)
        h_vals.addWidget(sb_ratio)
        h_vals.addWidget(QtWidgets.QLabel("%"))
        gl.addLayout(h_vals, 3, 1, 1, 11)

        # Row 4
        gl.addWidget(QtWidgets.QLabel("Freq. Range:"), 4, 0)

        h_bot = QtWidgets.QHBoxLayout()
        sb_fstart = _small_spin_dbl(width=80)
        h_bot.addWidget(sb_fstart)
        h_bot.addWidget(QtWidgets.QLabel("Hz"))

        h_bot.addWidget(QtWidgets.QLabel("Ampl. Range:"))
        sb_arange = _small_spin_dbl(width=80)
        h_bot.addWidget(sb_arange)

        h_bot.addWidget(QtWidgets.QLabel("Filter:"))
        h_bot.addWidget(QtWidgets.QLineEdit())

        gl.addLayout(h_bot, 4, 1, 1, 10)
        gl.addWidget(QtWidgets.QPushButton("Foton..."), 4, 11)

        outer.addWidget(gb)

        # Store panel controls
        panel_ctrl = {
            "group_box": gb,
            "active": chk_active,
            "ex_chan": cb_ex,
            "waveform": cb_wave,
            "freq": sb_freq,
            "amp": sb_amp,
            "offset": sb_off,
            "phase": sb_phase,
            "fstart": sb_fstart,
        }
        panels.append(panel_ctrl)
        controls["panels"].append(panel_ctrl)

    def update_titles(start_idx):
        for i, panel in enumerate(panels):
            panel["group_box"].setTitle(f"Channel {start_idx + i}")
            # We could also potentially load/save state here,
            # but we assume the 4 panels just modify 4 distinct slots if we had full logic.
            # For now, visual only.

    # We only implemented rb_0_3 logic for brevity in diaggui looks, but let's wire it up simply
    rb_0_3.toggled.connect(lambda c: update_titles(0) if c else None)

    outer.addStretch(1)

    # Expose target combo for external update
    controls["target_combos"] = [p["ex_chan"] for p in panels]

    return tab, controls


def create_result_tab(on_import=None):
    tab = QtWidgets.QWidget()
    hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    left_panel = QtWidgets.QWidget()
    left_vbox = QtWidgets.QVBoxLayout(left_panel)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    left_content = QtWidgets.QWidget()
    scroll.setWidget(left_content)
    lv = QtWidgets.QVBoxLayout(left_content)
    lv.setContentsMargins(6, 6, 6, 6)

    right_panel = QtWidgets.QWidget()
    rv = QtWidgets.QVBoxLayout(right_panel)
    rv.setContentsMargins(0, 0, 0, 0)
    plot1 = pg.PlotWidget(title="Graph 1")
    plot2 = pg.PlotWidget(title="Graph 2")
    rv.addWidget(plot1)
    rv.addWidget(plot2)

    traces1 = [
        {
            "curve": plot1.plot(pen="r"),
            "bar": pg.BarGraphItem(x=[0], height=[0], width=0.1, brush="r"),
            "img": pg.ImageItem(),
        }
        for _ in range(8)
    ]
    traces2 = [
        {
            "curve": plot2.plot(pen="b"),
            "bar": pg.BarGraphItem(x=[0], height=[0], width=0.1, brush="b"),
            "img": pg.ImageItem(),
        }
        for _ in range(8)
    ]
    for t in traces1:
        plot1.addItem(t["bar"])
        plot1.addItem(t["img"])
        t["bar"].setVisible(False)
        t["img"].setVisible(False)
    for t in traces2:
        plot2.addItem(t["bar"])
        plot2.addItem(t["img"])
        t["bar"].setVisible(False)
        t["img"].setVisible(False)

    info1_panel = GraphPanel(1, plot1, traces1)
    lv.addWidget(info1_panel)
    info1 = info1_panel.to_graph_info()
    lv.addWidget(QtWidgets.QFrame())
    lv.itemAt(lv.count() - 1).widget().setFrameShape(QtWidgets.QFrame.HLine)
    info2_panel = GraphPanel(2, plot2, traces2)
    lv.addWidget(info2_panel)
    info2 = info2_panel.to_graph_info()
    lv.addStretch(1)
    left_vbox.addWidget(scroll)
    hsplit.addWidget(left_panel)
    hsplit.addWidget(right_panel)
    hsplit.setStretchFactor(1, 1)

    main_layout = QtWidgets.QVBoxLayout(tab)
    main_layout.addWidget(hsplit)

    # --- Phase 2: Zoom/Active/Options state management ---
    # State variables stored in tab widget
    tab._active_pad = 0  # 0 = Graph 1, 1 = Graph 2
    tab._zoomed_pad = -1  # -1 = not zoomed, 0/1 = zoomed pad index
    tab._plots = [plot1, plot2]
    tab._panels = [info1_panel, info2_panel]

    def _update_active_highlight():
        """Update visual highlight for active pad."""
        for idx, plot in enumerate(tab._plots):
            if idx == tab._active_pad:
                plot.setTitle(f"Graph {idx + 1} [Active]")
            else:
                plot.setTitle(f"Graph {idx + 1}")

    def _toggle_zoom():
        """Toggle maximize/restore for the active pad (DTT Zoom behavior)."""
        if tab._zoomed_pad < 0:
            # Not zoomed -> Zoom active pad (hide the other)
            tab._zoomed_pad = tab._active_pad
            for idx, plot in enumerate(tab._plots):
                plot.setVisible(idx == tab._zoomed_pad)
        else:
            # Zoomed -> Restore all
            tab._zoomed_pad = -1
            for plot in tab._plots:
                plot.setVisible(True)

    def _cycle_active():
        """Cycle to the next graph pad (DTT Active behavior)."""
        num_pads = len(tab._plots)
        tab._active_pad = (tab._active_pad + 1) % num_pads
        _update_active_highlight()
        # If zoomed, switch zoom to new active
        if tab._zoomed_pad >= 0:
            tab._zoomed_pad = tab._active_pad
            for idx, plot in enumerate(tab._plots):
                plot.setVisible(idx == tab._zoomed_pad)

    def _show_layout_dialog():
        """Show layout options dialog (DTT Options behavior)."""
        dialog = QtWidgets.QDialog(tab)
        dialog.setWindowTitle("Layout Options")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)

        group = QtWidgets.QGroupBox("Select Layout")
        vbox = QtWidgets.QVBoxLayout(group)

        # Radio buttons for layout options
        rb_1x1 = QtWidgets.QRadioButton("1 x 1 (Single Graph)")
        rb_2x1 = QtWidgets.QRadioButton("2 x 1 (Two Graphs, Vertical)")
        rb_1x2 = QtWidgets.QRadioButton("1 x 2 (Two Graphs, Horizontal)")

        # Current layout: 2x1 (vertical stack)
        rb_2x1.setChecked(True)

        vbox.addWidget(rb_1x1)
        vbox.addWidget(rb_2x1)
        vbox.addWidget(rb_1x2)
        layout.addWidget(group)

        # Buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Apply layout change
            if rb_1x1.isChecked():
                # Show only active pad
                for idx, plot in enumerate(tab._plots):
                    plot.setVisible(idx == tab._active_pad)
                tab._zoomed_pad = tab._active_pad
            elif rb_2x1.isChecked():
                # Vertical stack (default)
                rv.setDirection(QtWidgets.QBoxLayout.TopToBottom)
                for plot in tab._plots:
                    plot.setVisible(True)
                tab._zoomed_pad = -1
            elif rb_1x2.isChecked():
                # Horizontal stack
                rv.setDirection(QtWidgets.QBoxLayout.LeftToRight)
                for plot in tab._plots:
                    plot.setVisible(True)
                tab._zoomed_pad = -1

    # Initialize highlight
    _update_active_highlight()

    # Bottom toolbar - DTT diaggui button order
    bot_toolbar = QtWidgets.QHBoxLayout()

    btn_reset = QtWidgets.QPushButton("Reset")
    btn_reset.clicked.connect(lambda: (plot1.autoRange(), plot2.autoRange()))
    bot_toolbar.addWidget(btn_reset)

    btn_zoom = QtWidgets.QPushButton("Zoom")
    btn_zoom.setToolTip("Maximize/restore the active graph pad")
    btn_zoom.clicked.connect(_toggle_zoom)
    bot_toolbar.addWidget(btn_zoom)

    btn_active = QtWidgets.QPushButton("Active")
    btn_active.setToolTip("Cycle focus to the next graph pad")
    btn_active.clicked.connect(_cycle_active)
    bot_toolbar.addWidget(btn_active)

    btn_new = QtWidgets.QPushButton("New")
    btn_new.setToolTip("Open a new window sharing the same data")
    bot_toolbar.addWidget(btn_new)

    btn_options = QtWidgets.QPushButton("Options...")
    btn_options.setToolTip("Configure graph layout (1x1, 2x1, etc.)")
    btn_options.clicked.connect(_show_layout_dialog)
    bot_toolbar.addWidget(btn_options)

    btn_import = QtWidgets.QPushButton("Import...")
    bot_toolbar.addWidget(btn_import)
    if on_import:
        btn_import.clicked.connect(on_import)

    btn_export = QtWidgets.QPushButton("Export...")
    btn_export.setToolTip("Save plot data to file")
    bot_toolbar.addWidget(btn_export)

    btn_reference = QtWidgets.QPushButton("Reference...")
    btn_reference.setToolTip("Manage reference traces for comparison")
    btn_reference.setEnabled(False)  # Phase 3: Reference
    bot_toolbar.addWidget(btn_reference)

    btn_calibration = QtWidgets.QPushButton("Calibration...")
    btn_calibration.setToolTip("Edit calibration table")
    btn_calibration.setEnabled(False)  # Phase 4
    bot_toolbar.addWidget(btn_calibration)

    bot_toolbar.addStretch(1)
    main_layout.addLayout(bot_toolbar)

    # Store button references for external access
    tab.btn_reset = btn_reset
    tab.btn_zoom = btn_zoom
    tab.btn_active = btn_active
    tab.btn_new = btn_new
    tab.btn_options = btn_options
    tab.btn_import = btn_import
    tab.btn_export = btn_export
    tab.btn_reference = btn_reference
    tab.btn_calibration = btn_calibration

    # Expose internal functions for testing
    tab._toggle_zoom = _toggle_zoom
    tab._cycle_active = _cycle_active
    tab._show_layout_dialog = _show_layout_dialog
    tab._update_active_highlight = _update_active_highlight

    return tab, info1, info2, traces1, traces2
