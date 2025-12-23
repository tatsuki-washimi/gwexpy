import sys
from PyQt5 import QtWidgets, QtCore


class DTTMockGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # ===== メインウィンドウ設定 =====
        self.setWindowTitle("CDS Diagnostic Test Tools")
        self.resize(1100, 500)

        # ===== 中央ウィジェット =====
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_vlayout = QtWidgets.QVBoxLayout(central)
        main_vlayout.setContentsMargins(2, 2, 2, 2)
        main_vlayout.setSpacing(2)

        # ===== タブバー =====
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabPosition(QtWidgets.QTabWidget.North)

        for name in ["Input", "Measurement", "Excitation", "Result"]:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            placeholder = QtWidgets.QLabel(" ")
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            tab_layout.addWidget(placeholder)
            self.tabs.addTab(tab, name)

        main_vlayout.addWidget(self.tabs, stretch=1)

        # ===== 下部共通コントロールバー =====
        bottom_bar = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(6, 4, 6, 4)
        bottom_layout.setSpacing(12)

        # --- 左側：制御ボタン（見た目のみ） ---
        for label in ["Start", "Pause", "Resume", "Abort"]:
            btn = QtWidgets.QPushButton(label)
            btn.setFixedWidth(90)
            bottom_layout.addWidget(btn)

        bottom_layout.addStretch(1)

        # --- 右側：状態表示ラベル（DTT準拠） ---
        lbl_repeat = QtWidgets.QLabel("Repeat")
        lbl_tools = QtWidgets.QLabel("Fourier tools")

        for lbl in (lbl_repeat, lbl_tools):
            lbl.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
            lbl.setStyleSheet("""
                QLabel {
                    color: #404040;
                    padding-left: 6px;
                    padding-right: 6px;
                }
            """)
            bottom_layout.addWidget(lbl)

        main_vlayout.addWidget(bottom_bar, stretch=0)

        # ===== メニューバー =====
        menubar = self.menuBar()
        for name in ["File", "Edit", "Measurement", "Plot", "Window", "Help"]:
            menubar.addMenu(name)

        # ===== 全体スタイル =====
        self.apply_style()

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #e6e6e6;
            }
            QTabWidget::pane {
                border: 1px solid #b0b0b0;
            }
            QTabBar::tab {
                background: #dcdcdc;
                padding: 6px 16px;
                border: 1px solid #b0b0b0;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: #f0f0f0;
            }
            QPushButton {
                background-color: #efefef;
                border: 1px solid #9a9a9a;
                padding: 4px 10px;
            }
            QPushButton:pressed {
                background-color: #d6d6d6;
            }
        """)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = DTTMockGUI()
    win.show()
    sys.exit(app.exec_())
