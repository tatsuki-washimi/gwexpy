# gui_measurement_mock_v3.py
# pip install pyqt5 pyqtgraph
# python gui_measurement_mock_v3.py

import sys
from PyQt5 import QtWidgets, QtCore


def _small_spin_int(minw=90):
    w = QtWidgets.QSpinBox()
    w.setRange(-10**9, 10**9)
    w.setMinimumWidth(minw)
    return w


def _small_spin_dbl(minw=90, decimals=1):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(-1e12, 1e12)
    w.setDecimals(decimals)
    w.setMinimumWidth(minw)
    return w


def create_measurement_tab():
    tab = QtWidgets.QWidget()
    outer = QtWidgets.QVBoxLayout(tab)
    outer.setContentsMargins(10, 10, 10, 10)
    outer.setSpacing(10)

    # ----------------------------
    # Measurement (mode selection)  <-- 横並びに修正
    # ----------------------------
    gb_measurement = QtWidgets.QGroupBox("Measurement")
    vb = QtWidgets.QVBoxLayout(gb_measurement)
    vb.setSpacing(6)

    mode_row = QtWidgets.QHBoxLayout()
    mode_row.setSpacing(16)

    rb_fourier = QtWidgets.QRadioButton("Fourier Tools")
    rb_swept = QtWidgets.QRadioButton("Swept Sine Response")
    rb_sine = QtWidgets.QRadioButton("Sine Response")
    rb_triggered = QtWidgets.QRadioButton("Triggered Time Response")
    rb_fourier.setChecked(True)

    mode_row.addWidget(rb_fourier)
    mode_row.addWidget(rb_swept)
    mode_row.addWidget(rb_sine)
    mode_row.addWidget(rb_triggered)
    mode_row.addStretch(1)

    vb.addLayout(mode_row)
    outer.addWidget(gb_measurement)

    # ----------------------------
    # Measurement Channels
    # ----------------------------
    gb_channels = QtWidgets.QGroupBox("Measurement Channels")
    vb = QtWidgets.QVBoxLayout(gb_channels)
    vb.setSpacing(8)

    range_row = QtWidgets.QHBoxLayout()
    range_buttons = []
    for t in [
        "Channels 0 to 15",
        "Channels 16 to 31",
        "Channels 32 to 47",
        "Channels 48 to 63",
        "Channels 64 to 79",
        "Channels 80 to 95",
    ]:
        rb = QtWidgets.QRadioButton(t)
        range_buttons.append(rb)
        range_row.addWidget(rb)
    range_buttons[0].setChecked(True)
    range_row.addStretch(1)
    vb.addLayout(range_row)

    # Two-column channel list (0-7 left, 8-15 right)
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(12)
    grid.setVerticalSpacing(6)

    for i in range(8):
        idx_l = i
        chk_l = QtWidgets.QCheckBox(str(idx_l))
        combo_l = QtWidgets.QComboBox()
        combo_l.setMinimumWidth(380)
        combo_l.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        idx_r = i + 8
        chk_r = QtWidgets.QCheckBox(str(idx_r))
        combo_r = QtWidgets.QComboBox()
        combo_r.setMinimumWidth(380)
        combo_r.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        grid.addWidget(chk_l, i, 0)
        grid.addWidget(combo_l, i, 1)
        grid.addWidget(chk_r, i, 2)
        grid.addWidget(combo_r, i, 3)

    grid.setColumnStretch(1, 1)
    grid.setColumnStretch(3, 1)
    vb.addLayout(grid)
    outer.addWidget(gb_channels)

    # ----------------------------
    # Fourier Tools
    # ----------------------------
    gb_fft = QtWidgets.QGroupBox("Fourier Tools")
    g = QtWidgets.QGridLayout(gb_fft)
    g.setHorizontalSpacing(10)
    g.setVerticalSpacing(6)

    # Row 0: Start/Stop/BW/TimeSpan/Settling/RampDown/RampUp
    r = 0
    c = 0

    def add_labeled_spin(label, spin, unit, col):
        g.addWidget(QtWidgets.QLabel(label), r, col)
        g.addWidget(spin, r, col + 1)
        g.addWidget(QtWidgets.QLabel(unit), r, col + 2)

    add_labeled_spin("Start:", _small_spin_dbl(), "Hz", c); c += 3
    add_labeled_spin("Stop:", _small_spin_dbl(), "Hz", c); c += 3
    add_labeled_spin("BW:", _small_spin_dbl(), "Hz", c); c += 3
    add_labeled_spin("TimeSpan:", _small_spin_dbl(), "s", c); c += 3
    add_labeled_spin("Settling Time:", _small_spin_dbl(), "%", c); c += 3
    add_labeled_spin("Ramp Down:", _small_spin_dbl(), "Sec", c); c += 3
    add_labeled_spin("Ramp Up:", _small_spin_dbl(), "Sec", c); c += 3

    # Row 1
    r = 1
    g.addWidget(QtWidgets.QLabel("Window:"), r, 0)
    cb_window = QtWidgets.QComboBox()
    cb_window.addItems(["Hanning", "Hamming", "Blackman", "Rectangular"])
    cb_window.setMinimumWidth(140)
    g.addWidget(cb_window, r, 1, 1, 2)

    g.addWidget(QtWidgets.QLabel("Overlap:"), r, 3)
    sp_overlap = _small_spin_dbl()
    g.addWidget(sp_overlap, r, 4)
    g.addWidget(QtWidgets.QLabel("%"), r, 5)

    chk_remove_mean = QtWidgets.QCheckBox("Remove mean")
    g.addWidget(chk_remove_mean, r, 6, 1, 2)

    g.addWidget(QtWidgets.QLabel("Number of A channels:"), r, 8)
    sp_num_a = _small_spin_int(80)
    g.addWidget(sp_num_a, r, 9)

    # Row 2
    r = 2
    g.addWidget(QtWidgets.QLabel("Averages:"), r, 0)
    sp_avg = _small_spin_int(80)
    sp_avg.setValue(10)
    g.addWidget(sp_avg, r, 1)

    g.addWidget(QtWidgets.QLabel("Average Type:"), r, 2)
    avg_type_box = QtWidgets.QHBoxLayout()
    rb_fixed = QtWidgets.QRadioButton("Fixed")
    rb_exp = QtWidgets.QRadioButton("Exponential")
    rb_acc = QtWidgets.QRadioButton("Accumulative")
    rb_fixed.setChecked(True)
    avg_type_box.addWidget(rb_fixed)
    avg_type_box.addWidget(rb_exp)
    avg_type_box.addWidget(rb_acc)
    avg_type_box.addStretch(1)
    g.addLayout(avg_type_box, r, 3, 1, 5)

    g.addWidget(QtWidgets.QLabel("Burst Noise Quiet Time"), r, 8)
    sp_quiet = _small_spin_dbl(90, decimals=2)
    g.addWidget(sp_quiet, r, 9)
    g.addWidget(QtWidgets.QLabel("sec"), r, 10)

    g.setColumnStretch(11, 1)
    outer.addWidget(gb_fft)

    # ----------------------------
    # Start Time
    # ----------------------------
    gb_time = QtWidgets.QGroupBox("Start Time")
    g = QtWidgets.QGridLayout(gb_time)
    g.setHorizontalSpacing(10)
    g.setVerticalSpacing(6)

    rb_now = QtWidgets.QRadioButton("Now")
    rb_now.setChecked(True)
    g.addWidget(rb_now, 0, 0, 1, 1)

    rb_gps = QtWidgets.QRadioButton("GPS:")
    g.addWidget(rb_gps, 1, 0)
    sp_gps_sec = _small_spin_int(140)
    sp_gps_nsec = _small_spin_int(100)
    g.addWidget(sp_gps_sec, 1, 1)
    g.addWidget(QtWidgets.QLabel("sec"), 1, 2)
    g.addWidget(sp_gps_nsec, 1, 3)
    g.addWidget(QtWidgets.QLabel("nsec"), 1, 4)

    rb_dt = QtWidgets.QRadioButton("Date/time:")
    g.addWidget(rb_dt, 2, 0)
    de = QtWidgets.QDateEdit()
    de.setCalendarPopup(True)
    te = QtWidgets.QTimeEdit()
    g.addWidget(de, 2, 1)
    g.addWidget(te, 2, 2)
    g.addWidget(QtWidgets.QLabel("UTC"), 2, 3)

    rb_future = QtWidgets.QRadioButton("In the future:")
    rb_past = QtWidgets.QRadioButton("In the past:")
    g.addWidget(rb_future, 0, 6)
    g.addWidget(rb_past, 1, 6)

    t_future = QtWidgets.QTimeEdit()
    t_future.setDisplayFormat("hh:mm:ss")
    t_past = QtWidgets.QTimeEdit()
    t_past.setDisplayFormat("hh:mm:ss")
    g.addWidget(t_future, 0, 7)
    g.addWidget(QtWidgets.QLabel("hh:mm:ss"), 0, 8)
    g.addWidget(t_past, 1, 7)
    g.addWidget(QtWidgets.QLabel("hh:mm:ss"), 1, 8)

    btn_time_now = QtWidgets.QPushButton("Time now")
    btn_lookup = QtWidgets.QPushButton("Lookup...")
    g.addWidget(btn_time_now, 2, 6)
    g.addWidget(btn_lookup, 2, 7)

    g.addWidget(QtWidgets.QLabel("Slow down:"), 2, 9)
    sp_slow = _small_spin_dbl(80, decimals=0)
    g.addWidget(sp_slow, 2, 10)
    g.addWidget(QtWidgets.QLabel("sec/avrg."), 2, 11)

    g.setColumnStretch(5, 1)
    g.setColumnStretch(12, 1)
    outer.addWidget(gb_time)

    # ----------------------------
    # Measurement Information
    # ----------------------------
    gb_info = QtWidgets.QGroupBox("Measurement Information")
    g = QtWidgets.QGridLayout(gb_info)
    g.setHorizontalSpacing(10)
    g.setVerticalSpacing(6)

    g.addWidget(QtWidgets.QLabel("Measurement Time:"), 0, 0)
    le_mtime = QtWidgets.QLineEdit()
    g.addWidget(le_mtime, 0, 1, 1, 5)

    g.addWidget(QtWidgets.QLabel("Comment / Description:"), 1, 0)
    le_comment = QtWidgets.QLineEdit()
    g.addWidget(le_comment, 1, 1, 1, 5)

    outer.addWidget(gb_info)

    outer.addStretch(1)
    return tab


def create_bottom_bar():
    bar = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(bar)
    h.setContentsMargins(10, 6, 10, 6)
    h.setSpacing(10)

    btn_start = QtWidgets.QPushButton("Start")
    btn_pause = QtWidgets.QPushButton("Pause")
    btn_resume = QtWidgets.QPushButton("Resume")
    btn_abort = QtWidgets.QPushButton("Abort")

    # 見た目だけ合わせる（元の雰囲気）
    btn_pause.setEnabled(False)
    btn_resume.setEnabled(False)

    h.addWidget(btn_start)
    h.addWidget(btn_pause)
    h.addWidget(btn_resume)
    h.addWidget(btn_abort)

    h.addStretch(1)

    # 右下は「文字表示」だけに修正
    lbl_repeat = QtWidgets.QLabel("Repeat")
    lbl_ftools = QtWidgets.QLabel("Fourier tools")
    lbl_repeat.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    lbl_ftools.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    h.addWidget(lbl_repeat)
    h.addSpacing(12)
    h.addWidget(lbl_ftools)

    return bar


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CDS Diagnostic Test Tools")
        self.resize(900, 900)

        self._create_menu()
        self._create_ui()

    def _create_menu(self):
        mb = self.menuBar()
        for name in ["File", "Edit", "Measurement", "Plot", "Window", "Help"]:
            mb.addMenu(name)

    def _create_ui(self):
        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(QtWidgets.QWidget(), "Input")
        self.tabs.addTab(create_measurement_tab(), "Measurement")
        self.tabs.addTab(QtWidgets.QWidget(), "Excitation")
        self.tabs.addTab(QtWidgets.QWidget(), "Result")

        v.addWidget(self.tabs, 1)
        v.addWidget(create_bottom_bar(), 0)

        self.setCentralWidget(central)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
