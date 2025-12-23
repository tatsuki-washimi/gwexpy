import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# --- スタイル設定 (LIGOツールの雰囲気を再現) ---
STYLESHEET = """
QMainWindow {
    background-color: #f0f0f0;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #aaa;
    border-radius: 3px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
    color: #333;
}
QPushButton {
    min-height: 24px;
    font-weight: bold;
}
/* Startボタンは緑系、Abortは赤系 */
QPushButton#btn_start { background-color: #d0e8d0; border: 1px solid #8f8; }
QPushButton#btn_abort { background-color: #e8d0d0; border: 1px solid #f88; }
"""

# --- ヘルパー関数 (修正版: 型キャストを追加) ---
def _create_spin(val=0, min_val=0, max_val=1000000000, decimals=1, suffix="", min_width=80):
    """
    数値入力ボックスを作成するヘルパー関数
    valがfloatならQDoubleSpinBox、intならQSpinBoxを返す
    """
    if isinstance(val, float):
        sb = QtWidgets.QDoubleSpinBox()
        sb.setDecimals(decimals)
        # QDoubleSpinBoxはfloatを受け付ける
        sb.setRange(float(min_val), float(max_val))
    else:
        sb = QtWidgets.QSpinBox()
        # QSpinBoxは厳密にintでなければならないためキャストする
        sb.setRange(int(min_val), int(max_val))
    
    sb.setValue(val)
    sb.setSuffix(suffix)
    sb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons) # 上下ボタンを隠す（本家風）
    sb.setMinimumWidth(min_width)
    return sb

class MeasurementTab(QtWidgets.QWidget):
    """
    Measurementタブ: 左側にモード選択、右側にパラメータ設定
    """
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 全体を左右に分割するレイアウト
        main_hbox = QtWidgets.QHBoxLayout(self)
        main_hbox.setContentsMargins(10, 10, 10, 10)
        main_hbox.setSpacing(10)

        # ------------------------------------------------
        # 1. 左カラム: Measurement Mode (Radio Buttons)
        # ------------------------------------------------
        gb_mode = QtWidgets.QGroupBox("Measurement Mode")
        vbox_mode = QtWidgets.QVBoxLayout(gb_mode)
        vbox_mode.setSpacing(10)
        
        self.rb_fft = QtWidgets.QRadioButton("Fourier Tools (FFT)")
        self.rb_swept = QtWidgets.QRadioButton("Swept Sine Response")
        self.rb_sine = QtWidgets.QRadioButton("Sine Response")
        self.rb_time = QtWidgets.QRadioButton("Time Response")
        
        self.rb_fft.setChecked(True) # デフォルト選択
        
        # ボタンを配置
        vbox_mode.addWidget(self.rb_fft)
        vbox_mode.addWidget(self.rb_swept)
        vbox_mode.addWidget(self.rb_sine)
        vbox_mode.addWidget(self.rb_time)
        vbox_mode.addStretch(1) # 下詰め
        
        main_hbox.addWidget(gb_mode, 1) # 比率1

        # ------------------------------------------------
        # 2. 右カラム: Parameters (Stacked Widget)
        # ------------------------------------------------
        self.param_stack = QtWidgets.QStackedWidget()
        
        # 各モード用のパネルを作成してスタックに追加
        self.param_stack.addWidget(self._create_fft_panel())   # Index 0
        self.param_stack.addWidget(self._create_swept_panel()) # Index 1
        # (他のモードはプレースホルダー)
        self.param_stack.addWidget(QtWidgets.QLabel("Sine Response Parameters..."))
        self.param_stack.addWidget(QtWidgets.QLabel("Time Response Parameters..."))

        main_hbox.addWidget(self.param_stack, 3) # 比率3 (右側を広く)

        # シグナル接続 (モード変更でパネル切り替え)
        self.rb_fft.toggled.connect(lambda: self.param_stack.setCurrentIndex(0))
        self.rb_swept.toggled.connect(lambda: self.param_stack.setCurrentIndex(1))
        self.rb_sine.toggled.connect(lambda: self.param_stack.setCurrentIndex(2))
        self.rb_time.toggled.connect(lambda: self.param_stack.setCurrentIndex(3))

    def _create_fft_panel(self):
        """FFTモード用のパラメータパネル"""
        panel = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)

        # --- Frequency Group ---
        gb_freq = QtWidgets.QGroupBox("Frequency Parameters")
        grid = QtWidgets.QGridLayout(gb_freq)
        grid.setHorizontalSpacing(15)
        grid.setVerticalSpacing(8)

        # Start / Stop
        grid.addWidget(QtWidgets.QLabel("Start Freq:"), 0, 0)
        grid.addWidget(_create_spin(0.0, suffix=" Hz"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Stop Freq:"), 0, 2)
        grid.addWidget(_create_spin(1000.0, suffix=" Hz"), 0, 3)

        # Bandwidth / Points
        grid.addWidget(QtWidgets.QLabel("Bandwidth:"), 1, 0)
        grid.addWidget(_create_spin(1.0, decimals=3, suffix=" Hz"), 1, 1)
        grid.addWidget(QtWidgets.QLabel("Points:"), 1, 2)
        grid.addWidget(_create_spin(401, decimals=0), 1, 3) # ここでエラーが出ていた箇所
        
        # Overlap
        grid.addWidget(QtWidgets.QLabel("Overlap:"), 2, 0)
        grid.addWidget(_create_spin(50, decimals=0, suffix=" %"), 2, 1)

        vbox.addWidget(gb_freq)

        # --- Averages & Windowing (横並び) ---
        hbox_opts = QtWidgets.QHBoxLayout()
        
        # Averages
        gb_avg = QtWidgets.QGroupBox("Averages")
        form_avg = QtWidgets.QFormLayout(gb_avg)
        form_avg.addRow("Count:", _create_spin(10, decimals=0))
        hbox_opts.addWidget(gb_avg)

        # Window
        gb_win = QtWidgets.QGroupBox("Windowing")
        form_win = QtWidgets.QFormLayout(gb_win)
        cb_win = QtWidgets.QComboBox()
        cb_win.addItems(["Hanning", "Flat-top", "Uniform"])
        form_win.addRow("Type:", cb_win)
        hbox_opts.addWidget(gb_win)

        vbox.addLayout(hbox_opts)
        vbox.addStretch(1) # 下詰め
        return panel

    def _create_swept_panel(self):
        """Swept Sineモード用のパラメータパネル"""
        panel = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)

        gb_sweep = QtWidgets.QGroupBox("Sweep Parameters")
        grid = QtWidgets.QGridLayout(gb_sweep)
        
        grid.addWidget(QtWidgets.QLabel("Start Freq:"), 0, 0)
        grid.addWidget(_create_spin(1.0, suffix=" Hz"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Stop Freq:"), 0, 2)
        grid.addWidget(_create_spin(100.0, suffix=" Hz"), 0, 3)
        
        grid.addWidget(QtWidgets.QLabel("Points:"), 1, 0)
        grid.addWidget(_create_spin(101, decimals=0), 1, 1)
        
        vbox.addWidget(gb_sweep)
        
        # Settling
        gb_settle = QtWidgets.QGroupBox("Settling")
        form = QtWidgets.QFormLayout(gb_settle)
        form.addRow("Cycles:", _create_spin(10, decimals=0))
        form.addRow("Delay:", _create_spin(0.1, suffix=" s"))
        vbox.addWidget(gb_settle)
        
        vbox.addStretch(1)
        return panel

class DiagMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CDS Diagnostic Test Tool (Replica)")
        self.resize(1000, 850)
        self.setStyleSheet(STYLESHEET)
        
        self._create_menu()
        self._create_ui()

    def _create_menu(self):
        mb = self.menuBar()
        for name in ["File", "Edit", "Measurement", "Plot", "Window", "Help"]:
            mb.addMenu(name)

    def _create_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)

        # ------------------------------------------------
        # 1. 画面分割 (Splitter)
        # 上: グラフ (Results), 下: 設定 (Tabs)
        # ------------------------------------------------
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # --- 上部: グラフエリア (PyQtGraph) ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('k') # 黒背景 (LIGO標準)
        self.plot_widget.ci.layout.setSpacing(0) # グラフ間の隙間を詰める
        
        # Plot 1: Magnitude
        self.p1 = self.plot_widget.addPlot(row=0, col=0)
        self.p1.setLabel('left', 'Magnitude')
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        self.p1.setLogMode(x=True, y=True) # 両対数
        
        # Plot 2: Phase (下に配置)
        self.p2 = self.plot_widget.addPlot(row=1, col=0)
        self.p2.setLabel('left', 'Phase', units='deg')
        self.p2.setLabel('bottom', 'Frequency', units='Hz')
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        self.p2.setLogMode(x=True, y=False) # 片対数
        self.p2.setXLink(self.p1) # X軸同期
        
        # ダミーデータ描画 (黄色い線)
        x = np.logspace(0, 3, 1000)
        y_mag = 100 / (x + 1) + np.random.normal(0, 0.1, 1000)
        y_phase = -180 * np.arctan(x/10) / np.pi + np.random.normal(0, 5, 1000)
        
        pen = pg.mkPen('#ffeb3b', width=1.5) # 黄色
        self.p1.plot(x, y_mag, pen=pen)
        self.p2.plot(x, y_phase, pen=pen)

        splitter.addWidget(self.plot_widget)

        # --- 下部: 設定タブ ---
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(QtWidgets.QWidget(), "Input")
        self.tabs.addTab(MeasurementTab(), "Measurement") # カスタムタブ
        self.tabs.addTab(QtWidgets.QWidget(), "Excitation")
        self.tabs.addTab(QtWidgets.QWidget(), "Results")
        
        self.tabs.setCurrentIndex(1) # Measurementを初期表示
        splitter.addWidget(self.tabs)
        
        # 初期比率 (グラフ 2 : タブ 1)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # ------------------------------------------------
        # 2. ボトムバー (Buttons & Status)
        # ------------------------------------------------
        bottom_bar = QtWidgets.QFrame()
        hbox_bar = QtWidgets.QHBoxLayout(bottom_bar)
        hbox_bar.setContentsMargins(5, 5, 5, 5)
        
        # ボタン群
        buttons = [
            ("Start", "btn_start"),
            ("Stop", ""),
            ("Pause", ""),
            ("Abort", "btn_abort")
        ]
        
        for text, obj_name in buttons:
            btn = QtWidgets.QPushButton(text)
            btn.setFixedWidth(80)
            if obj_name:
                btn.setObjectName(obj_name)
            hbox_bar.addWidget(btn)
        
        hbox_bar.addStretch(1)
        
        # ステータス表示
        hbox_bar.addWidget(QtWidgets.QLabel("Status: READY"))
        hbox_bar.addSpacing(15)
        hbox_bar.addWidget(QtWidgets.QLabel("GPS: 1234567890"))

        main_layout.addWidget(bottom_bar)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # フォント設定 (見やすく)
    font = app.font()
    font.setPointSize(10)
    font.setFamily("Arial") # またはシステム標準
    app.setFont(font)

    win = DiagMainWindow()
    win.show()
    sys.exit(app.exec_())
