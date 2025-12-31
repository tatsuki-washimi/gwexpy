from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal as Signal
import nds2
try:
    import sounddevice as sd
except ImportError:
    sd = None

class SearchThread(QtCore.QThread):
    finished = Signal(list, str) # results, error_msg

    def __init__(self, server, port, pattern):
        super().__init__()
        self.server = server
        self.port = port
        self.pattern = pattern

    def run(self):
        print(f"DEBUG: SearchThread starting for {self.pattern} on {self.server}:{self.port}")
        try:
            conn = nds2.connection(self.server, self.port)
            print("DEBUG: Connection established.")
            channels = conn.find_channels(self.pattern)
            print(f"DEBUG: find_channels returned {len(channels)} raw records.")
            
            names = set()
            for c in channels:
                name = c.name
                # Filter trends: ndscope logic hint
                if "," in name and ("-trend" in name):
                    continue
                names.add(name)
            
            res = sorted(list(names))
            print(f"DEBUG: Filtered to {len(res)} unique channels.")
            self.finished.emit(res, "")
        except Exception as e:
            print(f"DEBUG: SearchThread encountered error: {e}")
            self.finished.emit([], str(e))

class ChannelBrowserDialog(QtWidgets.QDialog):
    def __init__(self, server, port, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Channel Browser")
        self.resize(700, 500)
        self.server = server
        self.port = port
        self.selected_channels = []
        self.search_thread = None

        layout = QtWidgets.QVBoxLayout(self)

        # Source Selection
        h_source = QtWidgets.QHBoxLayout()
        h_source.addWidget(QtWidgets.QLabel("Source:"))
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["NDS Network", "Local PC Audio"])
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        h_source.addWidget(self.source_combo)
        h_source.addStretch(1)
        layout.addLayout(h_source)

        # Search Area (NDS)
        self.nds_widget = QtWidgets.QWidget()
        nds_layout = QtWidgets.QHBoxLayout(self.nds_widget)
        nds_layout.setContentsMargins(0, 0, 0, 0)
        nds_layout.addWidget(QtWidgets.QLabel("Search Pattern (glob):"))
        self.search_edit = QtWidgets.QLineEdit("K1:CAL-CS_PROC_*")
        # Connection for returnPressed removed to use keyPressEvent override
        nds_layout.addWidget(self.search_edit)
        self.btn_search = QtWidgets.QPushButton("Search")
        self.btn_search.clicked.connect(self.on_search)
        nds_layout.addWidget(self.btn_search)
        layout.addWidget(self.nds_widget)

        # List Area
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        # Mouse double-click still accepts
        self.list_widget.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.list_widget)

        # Status Area
        self.status_label = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status_label)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def on_source_changed(self, text):
        self.list_widget.clear()
        if text == "Local PC Audio":
            self.nds_widget.setVisible(False)
            self.list_local_devices()
        else:
            self.nds_widget.setVisible(True)
            self.status_label.setText("Enter pattern and click Search.")

    def list_local_devices(self):
        if sd is None:
            self.status_label.setText("Error: sounddevice not found.")
            return
        
        try:
            devices = sd.query_devices()
            results = []
            for i, dev in enumerate(devices):
                # We only interest in input devices
                if dev['max_input_channels'] > 0:
                    name = dev['name']
                    n_ch = dev['max_input_channels']
                    for ch in range(n_ch):
                        # Format: PC:MIC:[ID]-CH[N] (Name)
                        results.append(f"PC:MIC:{i}-CH{ch} ({name})")
            
            self.list_widget.addItems(results)
            self.status_label.setText(f"Found {len(results)} local input channels on {len(devices)} devices.")
        except Exception as e:
            self.status_label.setText(f"Error querying devices: {e}")

    def keyPressEvent(self, event):
        """Handle Enter key behavior manually based on focus."""
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if self.search_edit.hasFocus():
                self.on_search()
                event.accept()
                return
            elif self.list_widget.hasFocus():
                self.accept()
                event.accept()
                return
        super().keyPressEvent(event)

    def on_search(self):
        pattern = self.search_edit.text()
        if not pattern:
            return

        if self.search_thread and self.search_thread.isRunning():
            print("DEBUG: Search already in progress.")
            return

        self.btn_search.setEnabled(False)
        self.status_label.setText(f"Searching for '{pattern}'... please wait.")
        self.list_widget.clear()
        
        self.search_thread = SearchThread(self.server, self.port, pattern)
        self.search_thread.finished.connect(self._on_search_finished)
        self.search_thread.start()

    def _on_search_finished(self, results, error):
        self.btn_search.setEnabled(True)
        if error:
            print(f"DEBUG: Search finished with error: {error}")
            QtWidgets.QMessageBox.critical(self, "Search Error", f"Failed: {error}")
            self.status_label.setText("Error occurred.")
            return

        print(f"DEBUG: UI adding {len(results)} items to list...")
        # Safety limit for UI
        display_limit = 10000
        items_to_add = results[:display_limit]
        
        self.list_widget.addItems(items_to_add)
        
        msg = f"Found {len(results)} channels."
        if len(results) > display_limit:
            msg += f" (Showing first {display_limit})"
        self.status_label.setText(msg)
        print("DEBUG: UI update complete.")

    def accept(self):
        selected = []
        for item in self.list_widget.selectedItems():
            text = item.text()
            if text.startswith("PC:"):
                # Strip helper info: "PC:MIC:0-CH0 (Name)" -> "PC:MIC:0-CH0"
                text = text.split(" (")[0]
            selected.append(text)
        self.selected_channels = selected
        super().accept()
