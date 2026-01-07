from PyQt5 import QtWidgets, QtCore, QtGui
import fnmatch
from ..nds.cache import ChannelListCache
from ..nds.nds_thread import ChannelListWorker

try:
    import sounddevice as sd
except ImportError:
    sd = None


class ChannelBrowserDialog(QtWidgets.QDialog):
    def __init__(self, server, port, parent=None, audio_enabled=False, initial_source="NDS"):
        super().__init__(parent)
        self.setWindowTitle("Channel List")
        self.resize(800, 600)
        self.server_host = server
        self.port = port
        self.server_key = f"{server}:{port}"
        self.audio_enabled = audio_enabled
        self.current_source = initial_source
        self.selected_channels = []
        self.full_channel_list = []  # List of (name, rate, type)
        self.worker = None

        # UI Components
        layout = QtWidgets.QVBoxLayout(self)

        # Source Selection (New)
        h_src = QtWidgets.QHBoxLayout()
        h_src.addWidget(QtWidgets.QLabel("Source:"))
        self.cb_source = QtWidgets.QComboBox()
        self.cb_source.addItem(f"NDS ({self.server_key})", "NDS")
        if self.audio_enabled:
            self.cb_source.addItem("Local PC Audio", "AUDIO")
        
        # Set initial selection
        idx = self.cb_source.findData(self.current_source)
        if idx != -1:
            self.cb_source.setCurrentIndex(idx)
        
        self.cb_source.currentIndexChanged.connect(self.on_source_changed)
        h_src.addWidget(self.cb_source)
        h_src.addStretch(1)
        layout.addLayout(h_src)

        # Info Label
        self.lbl_info = QtWidgets.QLabel(f"server: <b>{self.server_key}</b> [Checking cache...]")
        self.lbl_info.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Search
        self.tab_search = QtWidgets.QWidget()
        self.setup_search_tab()
        self.tabs.addTab(self.tab_search, "Search")

        # Tab 2: Tree
        self.tab_tree = QtWidgets.QWidget()
        self.setup_tree_tab()
        self.tabs.addTab(self.tab_tree, "Tree")

        # Filter Controls
        self.setup_bottom_controls(layout)

        # Initial Load
        QtCore.QTimer.singleShot(100, self.reload_channel_list)

    def on_source_changed(self, idx=None):
        self.current_source = self.cb_source.currentData()
        self.reload_channel_list()

    def reload_channel_list(self):
        self.full_channel_list = []
        try:
            self.search_tree.clear()
            self.hier_tree.clear()
        except Exception as e:
            print(f"Error clearing trees: {e}")

        if self.current_source == "AUDIO":
            self.lbl_info.setText("Loading Audio Devices...")
            self.load_audio_devices()
            return

        # NDS Logic
        cache = ChannelListCache()
        data = cache.get_channels(self.server_key)

        if data is not None:
            self.full_channel_list = data
            self.lbl_info.setText(f"server: <b>{self.server_key}</b> [{len(data)} channels (cached)]")
            self.populate_ui()
        else:
            self.lbl_info.setText(f"server: <b>{self.server_key}</b> [Fetching...]")
            self.lbl_status.setText("Fetching channel list from NDS... This may take a while.")
            if self.worker:
                if self.worker.isRunning(): return

            self.worker = ChannelListWorker(self.server_host, self.port, "*")
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.start()

    def load_audio_devices(self):
        if sd is None:
            QtWidgets.QMessageBox.warning(self, "Error", "sounddevice module not found.")
            return

        try:
            results = []
            devices = sd.query_devices()
            # Input devices only? or output too? usually input for analysis.
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                     # Add each channel
                     for ch in range(dev['max_input_channels']):
                         name = f"PC:MIC:{i}-CH{ch}"
                         results.append((name, dev['default_samplerate'], 'audio'))

            self.full_channel_list = results
            self.lbl_info.setText(f"Local PC Audio [{len(results)} channels]")
            self.populate_ui()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    # ... rest of methods ...
    def setup_search_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_search)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search filter (substring or glob match)")
        self.search_edit.textChanged.connect(self.on_filter_changed)
        layout.addWidget(self.search_edit)

        self.search_tree = QtWidgets.QTreeWidget()
        self.search_tree.setHeaderLabels(["name", "rate"])
        self.search_tree.setColumnWidth(0, 400)
        self.search_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.search_tree.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.search_tree)

    def setup_tree_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_tree)
        self.hier_tree = QtWidgets.QTreeWidget()
        self.hier_tree.setHeaderLabels(["name", "rate"])
        self.hier_tree.setColumnWidth(0, 400)
        self.hier_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.hier_tree.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.hier_tree)

    def setup_bottom_controls(self, parent_layout):
        # Filtering Radios
        h_filter = QtWidgets.QHBoxLayout()

        self.rb_all = QtWidgets.QRadioButton("All")
        self.rb_slow = QtWidgets.QRadioButton("Slow Only")
        self.rb_fast = QtWidgets.QRadioButton("Fast Only")
        self.rb_all.setChecked(True)

        self.bg_filter = QtWidgets.QButtonGroup(self)
        self.bg_filter.addButton(self.rb_all)
        self.bg_filter.addButton(self.rb_slow)
        self.bg_filter.addButton(self.rb_fast)

        self.bg_filter.buttonToggled.connect(self.on_filter_changed)

        h_filter.addWidget(self.rb_all)
        h_filter.addWidget(self.rb_slow)
        h_filter.addWidget(self.rb_fast)

        h_filter.addStretch(1)

        # Testpoints label
        lbl_tp = QtWidgets.QLabel("testpoints in blue")
        lbl_tp.setStyleSheet("color: blue;")
        h_filter.addWidget(lbl_tp)

        parent_layout.addLayout(h_filter)

        # Actions
        h_btns = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Ready")
        h_btns.addWidget(self.lbl_status)
        h_btns.addStretch(1)

        btn_close = QtWidgets.QPushButton("Close") # Screenshot says Close
        btn_close.clicked.connect(self.reject)
        # We also need an "Add" or "Ok" logic. Screenshot shows "Drag channels into plot to add." and "Close".
        # But our main window expects a result on exec_.
        # Standard usage: Double click or maybe we add an "Add" button?
        # Original had OK/Cancel.
        # Let's keep standard OK/Cancel for modal behavior, but maybe label it "Add"?
        # Or just support double click.
        # Let's add an explicit "Add Selected" button just in case.
        self.btn_add = QtWidgets.QPushButton("Add Selected")
        self.btn_add.clicked.connect(self.accept)

        h_btns.addWidget(self.btn_add)
        h_btns.addWidget(btn_close)

        parent_layout.addLayout(h_btns)

    def load_data(self):
        cache = ChannelListCache()
        data = cache.get_channels(self.server_key)

        if data is not None:
            self.full_channel_list = data
            self.lbl_info.setText(f"server: <b>{self.server_key}</b> [{len(data)} channels (cached)]")
            self.populate_ui()
        else:
            self.lbl_info.setText(f"server: <b>{self.server_key}</b> [Fetching...]")
            self.lbl_status.setText("Fetching channel list from NDS... This may take a while.")
            self.worker = ChannelListWorker(self.server_host, self.port, "*")
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.start()

    def on_worker_finished(self, results, error):
        if error:
            # Only show error if we are still expecting NDS
            if self.current_source == "NDS":
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to fetch channels: {error}")
                self.lbl_status.setText("Error.")
            return

        # Update Cache (Always)
        ChannelListCache().set_channels(self.server_key, results)

        # Update UI only if still detecting NDS
        if self.current_source == "NDS":
            self.full_channel_list = results
            self.lbl_info.setText(f"server: <b>{self.server_key}</b> [{len(results)} channels]")
            self.lbl_status.setText("Ready")
            self.populate_ui()

    def populate_ui(self):
        # Build Tree
        self.build_hierarchy_tree(self.full_channel_list)
        # Initial Search Filter (All)
        self.apply_filter()

    def on_filter_changed(self):
        self.apply_filter()

    def apply_filter(self):
        pattern = self.search_edit.text()
        mode = "all"
        if self.rb_slow.isChecked(): mode = "slow"
        elif self.rb_fast.isChecked(): mode = "fast"

        # We only filter the Search Tab list. The Tree Tab usually shows everything or could be filtered too.
        # For performance, let's filter the Search List.

        is_glob = "*" in pattern or "?" in pattern

        # Pre-compile glob if needed? fnmatch is ok.

        count = 0
        limit = 5000 # Safety limit for display

        self.search_tree.clear()

        items = []
        for name, rate, ctype in self.full_channel_list:
            # Rate Filter
            if mode == "slow" and rate > 16: continue
            if mode == "fast" and rate <= 16: continue

            # Name Filter
            if pattern:
                if is_glob:
                    if not fnmatch.fnmatch(name, pattern): continue
                else:
                    if pattern not in name: continue

            # Add to list
            item = QtWidgets.QTreeWidgetItem([name, str(rate)])

            # Color logic (mimicking screenshot)
            # 16Hz -> Green?
            # Fast -> Blue?
            if rate <= 16:
                item.setForeground(0, QtGui.QBrush(QtGui.QColor("green")))
                item.setForeground(1, QtGui.QBrush(QtGui.QColor("green")))
            else:
                item.setForeground(0, QtGui.QBrush(QtGui.QColor("blue")))
                item.setForeground(1, QtGui.QBrush(QtGui.QColor("blue")))

            items.append(item)
            count += 1
            if count >= limit:
                break

        self.search_tree.addTopLevelItems(items)

        if count >= limit:
            self.lbl_status.setText(f"Showing first {limit} matches.")
        else:
            self.lbl_status.setText(f"Found {count} matches.")

    def build_hierarchy_tree(self, channels):
        self.hier_tree.clear()

        # A simple tree builder: split by ':' then '-' or '_'
        # But standard NDS names are usually K1:SYS-SUBSYS_...
        # Let's use a standard separator set for splitting.
        # Or just ':' for top level.

        # K1:ADS-DCU_ID
        #  -> K1
        #    -> ADS
        #      -> DCU
        #        -> K1:ADS-DCU_ID

        # We need a recursive dict builder first
        root = {}

        for name, rate, ctype in channels:
            # Heuristic splitting
            parts = name.split(':')
            # Further split the second part if exists?
            # Example: K1:ADS-DCU... -> [K1, ADS, DCU...]
            # Actually, usually subsystem is first 3 chars after :?
            # Let's keep it simple: Split by ':' first.
            if len(parts) > 1:
                # [K1, ADS-...]
                domain = parts[0]
                rest = parts[1]

                # Try to split by '-'
                sub_parts = rest.split('-')
                path = [domain] + sub_parts
            else:
                path = parts

            # Traverse/Build dict
            current = root
            for p in path[:-1]:
                if p not in current:
                    current[p] = {}
                current = current[p]
                if isinstance(current, tuple): # Conflict: node is both leaf and branch?
                     # Should not happen if naming is consistent, but if it does,
                     # we might lose the leaf. NDS names are unique.
                     # But if we have A:B and A:B-C, then B is a leaf in first and node in second?
                     # We'll see.
                     pass

            leaf_key = path[-1]
            # Store data at leaf
            current[leaf_key] = (name, rate)

        # Now convert dict to QTreeItems
        # This is recursive

        def dict_to_items(node_dict):
            items = []
            keys = sorted(node_dict.keys())
            for k in keys:
                val = node_dict[k]
                if isinstance(val, tuple):
                    # It's a leaf channel
                    # Display: Name Rate
                    # Screenshot shows leaf has full name? Or just suffix?
                    # Screenshot: "K1:ADS-DCU_ID 16"
                    c_name, c_rate = val
                    item = QtWidgets.QTreeWidgetItem([c_name, str(c_rate)])
                    # Color
                    if c_rate <= 16:
                         item.setForeground(0, QtGui.QBrush(QtGui.QColor("green")))
                    else:
                         item.setForeground(0, QtGui.QBrush(QtGui.QColor("blue")))
                    items.append(item)
                else:
                    # It's a folder
                    item = QtWidgets.QTreeWidgetItem([k, ""])
                    children = dict_to_items(val)
                    item.addChildren(children)
                    items.append(item)
            return items

        top_items = dict_to_items(root)
        self.hier_tree.addTopLevelItems(top_items)


    def accept(self):
        # Gather selected from ACTIVE tab
        selected = []
        current_widget = self.tabs.currentWidget()

        tree = None
        if current_widget == self.tab_search:
            tree = self.search_tree
        elif current_widget == self.tab_tree:
            tree = self.hier_tree

        if tree:
            for item in tree.selectedItems():
                # Check if it is a channel (has rate)
                if item.childCount() == 0 and item.text(1):
                    selected.append(item.text(0))

        self.selected_channels = selected
        super().accept()
