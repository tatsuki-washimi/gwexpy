from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal as Signal
import nds2

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
        self.setWindowTitle(f"NDS Channel Browser - {server}:{port}")
        self.resize(600, 500)
        self.server = server
        self.port = port
        self.selected_channels = []
        self.search_thread = None

        layout = QtWidgets.QVBoxLayout(self)

        # Search Area
        h_search = QtWidgets.QHBoxLayout()
        h_search.addWidget(QtWidgets.QLabel("Search Pattern (glob):"))
        self.search_edit = QtWidgets.QLineEdit("K1:CAL-CS_PROC_*")
        # Connection for returnPressed removed to use keyPressEvent override
        h_search.addWidget(self.search_edit)
        self.btn_search = QtWidgets.QPushButton("Search")
        self.btn_search.clicked.connect(self.on_search)
        h_search.addWidget(self.btn_search)
        layout.addLayout(h_search)

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
        self.selected_channels = [item.text() for item in self.list_widget.selectedItems()]
        super().accept()
