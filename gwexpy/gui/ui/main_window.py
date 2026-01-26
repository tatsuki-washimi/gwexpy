import logging

import numpy as np
from gwpy.timeseries import TimeSeries
from PyQt5 import QtCore, QtWidgets

logger = logging.getLogger(__name__)

from gwexpy.io.dttxml_common import extract_xml_channels

from ..engine import Engine

# Excitation Module Imports
from ..excitation.generator import SignalGenerator
from ..loaders import loaders, products
from ..nds.cache import NDSDataCache
from ..plotting.normalize import normalize_series
from ..streaming import SpectralAccumulator
from .channel_browser import ChannelBrowserDialog
from .excitation_manager import ExcitationManager
from .graph_panel import GraphPanel
from .plot_renderer import PlotRenderer
from .tabs import (
    create_excitation_tab,
    create_input_tab,
    create_measurement_tab,
    create_result_tab,
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, enable_preload=True, data_backend=None):
        super().__init__()
        self._enable_preload = enable_preload
        self.setWindowTitle("pyaggui : a diaggui-like gwexpy GUI-tool")
        self.resize(1100, 850)

        mb = self.menuBar()
        for m in ["File", "Edit", "Measurement", "Plot", "Window", "Help"]:
            menu = mb.addMenu(m)
            if m == "File":
                op = QtWidgets.QAction("Open...", self)
                op.triggered.connect(self.open_file_dialog)
                menu.addAction(op)
                menu.addAction("Exit").triggered.connect(self.close)

        self.tabs = QtWidgets.QTabWidget()
        input_tab, self.input_controls = create_input_tab()
        self.tabs.addTab(input_tab, "Input")
        meas_tab, self.meas_controls = create_measurement_tab()
        self.tabs.addTab(meas_tab, "Measurement")

        # Create Excitation Tab and capture controls
        exc_tab, self.exc_controls = create_excitation_tab()
        self.tabs.addTab(exc_tab, "Excitation (Simulation)")
        # Enable it now that we have implementation
        self.tabs.setTabEnabled(2, True)

        res_tab, self.graph_info1, self.graph_info2, self.traces1, self.traces2 = (
            create_result_tab(on_import=self.open_file_dialog)
        )
        self.tabs.addTab(res_tab, "Result")
        self.res_tab = res_tab  # Keep reference for button access

        # Connect Export button
        if hasattr(res_tab, "btn_export"):
            res_tab.btn_export.clicked.connect(lambda: self.export_data())

        # Connect New button (Phase 3)
        if hasattr(res_tab, "btn_new"):
            res_tab.btn_new.clicked.connect(self.open_new_result_window)

        # Connect Reference button (Phase 3)
        if hasattr(res_tab, "btn_reference"):
            res_tab.btn_reference.clicked.connect(self.show_reference_dialog)

        # Connect Calibration button (Phase 4)
        if hasattr(res_tab, "btn_calibration"):
            res_tab.btn_calibration.setEnabled(
                False
            )  # Disable until implemented in v0.5+
            # res_tab.btn_calibration.clicked.connect(self.show_calibration_dialog)

        self.engine = Engine()
        self.loaded_products = {}
        self.is_file_mode = False
        self.is_loading_file = False

        # NDS Integration
        self.nds_cache = data_backend if data_backend is not None else NDSDataCache()
        self.nds_cache.signal_data.connect(self.on_nds_data)
        self.nds_latest_raw = None  # Stores latest DataBufferDict
        self.data_source = "SIM"  # SIM, FILE, NDS

        # Signal Generator
        self.sig_gen = SignalGenerator()

        # Streaming Accumulator
        self.accumulator = SpectralAccumulator()
        self.nds_cache.signal_payload.connect(self.on_stream_payload)
        if hasattr(self.nds_cache, "signal_error"):
            self.nds_cache.signal_error.connect(self.on_data_error)

        # Excitation Manager (Phase 2 refactor)
        self.excitation_manager = ExcitationManager(self.sig_gen, self.exc_controls)

        # Plot Renderer (Phase 3 refactor)
        self.plot_renderer = PlotRenderer(self)

        self.input_controls["ds_combo"].currentTextChanged.connect(
            self.on_source_changed
        )

        def connect_trace_combos(graph_info):
            for ctrl in graph_info["traces"]:
                if "chan_a" in ctrl:
                    ctrl["chan_a"].currentTextChanged.connect(
                        self.on_trace_channel_changed
                    )
                if "chan_b" in ctrl:
                    ctrl["chan_b"].currentTextChanged.connect(
                        self.on_trace_channel_changed
                    )
            # Also connect graph type combo
            if "graph_combo" in graph_info:
                graph_info["graph_combo"].currentTextChanged.connect(
                    self.on_trace_channel_changed
                )

        connect_trace_combos(self.graph_info1)
        connect_trace_combos(self.graph_info2)

        # Link X-Axis of Graph 2 to Graph 1 (Synchronization)
        # self.graph_info2["plot"].setXLink(self.graph_info1["plot"]) # Removed unconditional link

        # Connect Graph Combo to update X-Link logic
        self.graph_info1["graph_combo"].currentTextChanged.connect(
            self.update_x_link_logic
        )
        self.graph_info2["graph_combo"].currentTextChanged.connect(
            self.update_x_link_logic
        )

        # Connect Range Mode (Auto/Manual) RadioButtons to update X-Link logic
        p1 = self.graph_info1.get("panel")
        p2 = self.graph_info2.get("panel")
        if p1 and hasattr(p1, "rb_x_auto"):
            p1.rb_x_auto.toggled.connect(self.update_x_link_logic)
        if p2 and hasattr(p2, "rb_x_auto"):
            p2.rb_x_auto.toggled.connect(self.update_x_link_logic)

        # Initial Link Check
        self.update_x_link_logic()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        cv = QtWidgets.QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.tabs)

        bot = QtWidgets.QHBoxLayout()
        central.layout().addLayout(bot)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #A00000;")
        bot.addWidget(self.status_label)
        bot.addStretch(1)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.setMinimumWidth(120)
        self.btn_start.setStyleSheet("font-weight: bold;")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setMinimumWidth(120)
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_resume.setEnabled(False)
        self.btn_resume.setMinimumWidth(120)
        self.btn_abort = QtWidgets.QPushButton("Abort")
        self.btn_abort.setMinimumWidth(120)
        for b in [self.btn_start, self.btn_pause, self.btn_resume, self.btn_abort]:
            bot.addWidget(b)
            if b != self.btn_abort:
                bot.addStretch(1)
        self.btn_start.clicked.connect(self.start_animation)
        self.btn_pause.clicked.connect(self.pause_animation)
        self.btn_resume.clicked.connect(self.resume_animation)
        self.btn_abort.clicked.connect(self.stop_animation)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graphs)
        self.time_counter = 0.0

        # Connect Measurement Channel Updates
        self.meas_controls["set_change_callback"](self.on_measurement_channel_changed)
        self.meas_controls["btn_browse"].clicked.connect(
            lambda: self.show_channel_browser()
        )

        # Connect Individual Channel Browse buttons
        for i, row in enumerate(self.meas_controls["grid_refs"]):
            # Each row is (label, checkbox, combobox, browse_button)
            btn = row[3]
            btn.clicked.connect(
                lambda checked, idx=i: self.show_channel_browser(start_slot=idx)
            )

        # Connect Excitation Channel Updates to trigger channel list refresh
        if self.exc_controls and "panels" in self.exc_controls:
            for p in self.exc_controls["panels"]:
                # When Active toggled or Channel name changed
                p["active"].toggled.connect(self.on_measurement_channel_changed)
                p["ex_chan"].editTextChanged.connect(
                    self.on_measurement_channel_changed
                )
                p["ex_chan"].currentIndexChanged.connect(
                    self.on_measurement_channel_changed
                )

        # Initial sync
        self.on_measurement_channel_changed()

        if self._enable_preload:
            # Startup NDS Cache Pre-loading (like ndscope)
            QtCore.QTimer.singleShot(0, self.preload_nds_channels)
            # Trigger update when switching away from Input tab (if settings changed)
            self.tabs.currentChanged.connect(lambda idx: self.preload_nds_channels())

    def preload_nds_channels(self):
        """Fetch NDS channels in background for the default server."""
        if not self._enable_preload:
            return

        # Prevention: Do not start another worker if one is already running
        if getattr(self, "_preload_worker", None) is not None:
            return

        try:
            # Determine default server/port (Same logic as show_channel_browser)
            ds_mode = self.input_controls["ds_combo"].currentText()
            # If not in NDS mode, do not preload
            if ds_mode not in ["NDS", "NDS2"]:
                return

            if ds_mode == "NDS2":
                server = self.input_controls["nds2_server"].currentText()
                port = self.input_controls["nds2_port"].value()
            else:
                server = self.input_controls["nds_server"].currentText()
                port = self.input_controls["nds_port"].value()

            from ..nds.cache import ChannelListCache
            from ..nds.nds_thread import ChannelListWorker

            key = f"{server}:{port}"
            if ChannelListCache().get_channels(key) is not None:
                return  # Already cached (unlikely at startup, but safe)

            logger.debug(f"Pre-loading NDS channels for {key}")
            self._preload_worker = ChannelListWorker(server, port)
            self._preload_worker.finished.connect(
                lambda results, err: self._on_preload_finished(key, results, err)
            )
            self._preload_worker.start()
        except (AttributeError, KeyError, RuntimeError, TypeError) as e:
            logger.error(f"Preload Error: {e}")

    def _on_preload_finished(self, key, results, error):
        from ..nds.cache import ChannelListCache

        if error:
            logger.error(f"Pre-load failed for {key}: {error}")
        else:
            ChannelListCache().set_channels(key, results)
            logger.info(f"Pre-load complete for {key}: {len(results)} channels cached.")
        self._preload_worker = None  # Release ref

    def update_x_link_logic(self):
        """
        Dynamically link/unlink X-axes based on graph types AND range mode.
        - Time-based graphs (Time Series, Spectrogram) should be linked.
        - Frequency-based graphs (ASD, Coherence, TF) should be linked.
        - Mixed types should NOT be linked.
        - Only link when BOTH are in "Auto" range mode.
        """
        g1 = self.graph_info1["graph_combo"].currentText()
        g2 = self.graph_info2["graph_combo"].currentText()

        time_types = ["Time Series", "Spectrogram"]

        is_g1_time = g1 in time_types
        is_g2_time = g2 in time_types

        same_type = is_g1_time == is_g2_time

        # Check Auto Range mode
        p1 = self.graph_info1.get("panel")
        p2 = self.graph_info2.get("panel")

        is_g1_auto = True
        is_g2_auto = True
        if p1 and hasattr(p1, "rb_x_auto"):
            is_g1_auto = p1.rb_x_auto.isChecked()
        if p2 and hasattr(p2, "rb_x_auto"):
            is_g2_auto = p2.rb_x_auto.isChecked()

        should_link = same_type and is_g1_auto and is_g2_auto

        # Cache previous state to avoid redundant calls (prevents flicker)
        prev_linked = getattr(self, "_x_axes_linked", None)

        if should_link and prev_linked is not True:
            self.graph_info2["plot"].setXLink(self.graph_info1["plot"])
            self._x_axes_linked = True
        elif not should_link and prev_linked is not False:
            self.graph_info2["plot"].setXLink(None)
            self._x_axes_linked = False

    def on_measurement_channel_changed(self):
        # Update Result tab comboboxes based on active Measurement channels
        active_channels = []
        states = self.meas_controls["channel_states"]
        for s in states:
            if s["active"] and s["name"]:
                active_channels.append(s["name"])

        # Add generated channels if known?
        # Or should they always be available?
        # Let's add standard "Excitation" if enabled?
        # Better: Do it dynamically or just rely on them being added when active.
        # Ideally, user should see "Excitation" in the list even if stopped?
        # For now, let's keep it simple: Measurement channels define the majority.
        # We can append "Exictation" and "Sum:..." implicitly if we want persistent selection.
        # But if we clear the combo, we lose the selection if it's not in the list.
        # Let's add 'Excitation' to the list if the module is theoretically available.
        # if "Excitation" not in active_channels:
        #      active_channels.append("Excitation")

        # Feature: Add channels defined in Excitation Tab (active ones) to the list
        if hasattr(self, "exc_controls") and "panels" in self.exc_controls:
            for p in self.exc_controls["panels"]:
                if p["active"].isChecked():
                    # Channel used as injection TARGET
                    tgt = p["ex_chan"].currentText()
                    if tgt and tgt not in active_channels:
                        active_channels.append(tgt)
                    # Also Add Sum:{tgt} if it's injection?
                    # For now, let's blindly add the Target name.
                    # If user typed a NEW name (not in measurement), it's a pure sim channel.
                    # If it's existing, it's already in active_channels (measurement).

        # Also, update the Target Channel list in Excitation Tab (New Multi-panel support)
        if hasattr(self, "exc_controls") and "target_combos" in self.exc_controls:
            for cb_target in self.exc_controls["target_combos"]:
                curr = cb_target.currentText()
                cb_target.blockSignals(True)
                cb_target.clear()
                cb_target.addItems(active_channels)
                # Restore
                idx = cb_target.findText(curr)
                if idx != -1:
                    cb_target.setCurrentIndex(idx)
                else:
                    # If custom name (e.g. default Excitation-0) is not in list, preserve it visually
                    cb_target.setEditText(curr)

                cb_target.blockSignals(False)
        # Fallback for old key if mixed versions (optional, can remove)
        elif hasattr(self, "exc_controls") and "ex_target" in self.exc_controls:
            cb_target = self.exc_controls["ex_target"]
            curr = cb_target.currentText()
            cb_target.blockSignals(True)
            cb_target.clear()
            cb_target.addItems(active_channels)
            idx = cb_target.findText(curr)
            if idx != -1:
                cb_target.setCurrentIndex(idx)
            cb_target.blockSignals(False)

        # Update comboboxes in Result tab
        # Note: We should preserve current selection if it is still valid
        for info in [self.graph_info1, self.graph_info2]:
            for ctrl in info["traces"]:
                for c_key in ["chan_a", "chan_b"]:
                    if c_key in ctrl:
                        combo = ctrl[c_key]
                        current = combo.currentText()
                        combo.blockSignals(True)
                        combo.clear()
                        combo.addItems(active_channels)

                        # Restore previous selection if possible, otherwise default or empty
                        idx = combo.findText(current)
                        if idx != -1:
                            combo.setCurrentIndex(idx)
                        elif active_channels:
                            combo.setCurrentIndex(0)

                        combo.blockSignals(False)

    def show_channel_browser(self, start_slot=None):
        # Determine which server/port to use based on current selection in Input tab
        use_pc_audio = self.input_controls["pcaudio"].isChecked()
        ds_mode = self.input_controls["ds_combo"].currentText()
        if ds_mode == "NDS2":
            server = self.input_controls["nds2_server"].currentText()
            port = self.input_controls["nds2_port"].value()
        else:
            server = self.input_controls["nds_server"].currentText()
            port = self.input_controls["nds_port"].value()

        initial_source = "NDS"
        if use_pc_audio and ds_mode not in ["NDS", "NDS2"]:
            # If we are in SIM/FILE mode but want to browse, default to Audio if enabled
            initial_source = "AUDIO"

        dlg = ChannelBrowserDialog(
            server,
            port,
            self,
            audio_enabled=use_pc_audio,
            initial_source=initial_source,
        )
        if dlg.exec_():
            chans = dlg.selected_channels
            if not chans:
                return

            # Get current display offset (0, 16, 32, ...)
            offset = self.meas_controls["get_bank_offset"]()
            states = self.meas_controls["channel_states"]
            new_states = [dict(s) for s in states]

            if start_slot is not None:
                # 1. Individual row triggered (starts at start_slot + offset)
                idx = offset + start_slot

                # Replace the current one with the first selected channel
                new_states[idx]["name"] = chans[0]
                new_states[idx]["active"] = True

                # Fill subsequent selected channels into empty slots after idx
                chan_idx = 1
                if len(chans) > 1:
                    for j in range(idx + 1, len(new_states)):
                        if chan_idx >= len(chans):
                            break
                        if not new_states[j]["name"]:
                            new_states[j]["name"] = chans[chan_idx]
                            new_states[j]["active"] = True
                            chan_idx += 1
                count = chan_idx
            else:
                # 2. Master "Channel Browser..." button (find any empty slots from start)
                count = 0
                for i in range(len(new_states)):
                    if not new_states[i]["name"]:
                        new_states[i]["name"] = chans[count]
                        new_states[i]["active"] = True
                        count += 1
                        if count >= len(chans):
                            break

            # Update UI to reflect model
            self.meas_controls["set_all_channels"](new_states)
            logger.info(f"Added {count} channels from {server}:{port}")

    def on_source_changed(self, text):
        self.data_source = text
        if text == "NDS":
            self.nds_latest_raw = None

    def on_stream_payload(self, payload):
        """Handle incremental NDS data."""
        self.accumulator.add_chunk(payload)

    def on_nds_data(self, buffers):
        print(f"DEBUG: MainWindow received NDS data with {len(buffers)} channels")
        self.nds_latest_raw = buffers

    def on_data_error(self, message):
        logger.warning("Data backend error: %s", message)
        if hasattr(self, "status_label"):
            self.status_label.setText(message)

    def start_animation(self):
        # Prevention: Do not start if already running
        if (
            self.timer.isActive()
            or self.btn_abort.isEnabled()
            and not self.btn_start.isEnabled()
        ):
            logger.warning("start_animation called while already active. Ignoring.")
            return

        if hasattr(self, "status_label"):
            self.status_label.setText("")
        # Validate Excitation Channels (uniqueness)
        if self.exc_controls and "panels" in self.exc_controls:
            active_names = []
            for i, p in enumerate(self.exc_controls["panels"]):
                if p["active"].isChecked():
                    name = p["ex_chan"].currentText()
                    if name in active_names:
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Configuration Error",
                            f"Duplicate Excitation Channel name detected: '{name}'.\nPlease assign unique names for active excitation channels.",
                        )
                        return
                    active_names.append(name)

        self.tabs.setCurrentIndex(3)
        self.is_file_mode = False

        # Read Data Source directly from UI to avoid signal issues
        self.data_source = self.input_controls["ds_combo"].currentText()
        logger.info(f"start_animation called. DataSource: {self.data_source}")

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_abort.setEnabled(True)
        # self.tabs.setTabEnabled(0, False) # Allow switching to Input tab even during operation

        use_pc_audio = self.input_controls["pcaudio"].isChecked()
        channels = []

        if self.data_source in ["NDS", "NDS2"] or use_pc_audio:
            # Collect channels
            sim_channels = [
                "HF_sine",
                "LF_sine",
                "beating_sine",
                "white_noise",
                "sine_plus_noise",
                "square_wave",
                "sawtooth_wave",
                "random_walk",
            ]
            states = self.meas_controls["channel_states"]
            for s in states:
                if s["active"] and s["name"]:
                    if s["name"] in sim_channels:
                        continue

                    if s["name"].startswith("PC:"):
                        if use_pc_audio:
                            channels.append(s["name"])
                    elif self.data_source in ["NDS", "NDS2"]:
                        channels.append(s["name"])

            if channels:
                # Determine which NDS server/port to use (ignored by PC Audio channels anyway)
                ds_mode = self.input_controls["ds_combo"].currentText()
                if ds_mode == "NDS2":
                    server = self.input_controls["nds2_server"].currentText()
                    port = self.input_controls["nds2_port"].value()
                else:
                    server = self.input_controls["nds_server"].currentText()
                    port = self.input_controls["nds_port"].value()

                # If we have NDS channels, configure server. Otherwise skip to allow PC Audio only.
                # Filter strictly NDS channels (exclude PC: and Excitation specific if any)
                real_nds_channels = [
                    c for c in channels if not c.startswith("PC:") and c != "Excitation"
                ]

                if real_nds_channels:
                    self.nds_cache.set_server(f"{server}:{port}")
                    self.nds_cache.set_channels(channels)
                    win_sec = self.input_controls["nds_win"].value()
                    self.nds_cache.online_start(lookback=win_sec)
                elif any(c.startswith("PC:") for c in channels):
                    self.nds_cache.set_channels(channels)
                    win_sec = self.input_controls["nds_win"].value()
                    self.nds_cache.online_start(lookback=win_sec)

        elif self.data_source == "Simulation":
            # Simulation mode
            logger.debug("Starting Simulation mode")
            # For simulation, we can just use the provided channels from meas_controls
            states = self.meas_controls["channel_states"]
            for s in states:
                if s["active"] and s["name"]:
                    channels.append(s["name"])

            if not channels:
                # Default if none: just use white_noise
                channels = ["white_noise"]

            self.nds_cache.set_channels(channels)
            win_sec = self.input_controls["nds_win"].value()
            self.nds_cache.sim_start(lookback=win_sec)

        # Configure Spectral Accumulator (moved outside of specific source blocks to cover all)
        params = self.get_ui_params()
        all_traces = []
        for info in [self.graph_info1, self.graph_info2]:
            g_type = info["graph_combo"].currentText()
            for c in info["traces"]:
                all_traces.append(
                    {
                        "active": c["active"].isChecked(),
                        "ch_a": c["chan_a"].currentText(),
                        "ch_b": c["chan_b"].currentText(),
                        "gain": c.get("gain").value() if c.get("gain") else 1.0,
                        "graph_type": g_type,
                    }
                )
        self.accumulator.configure(params, all_traces, available_channels=channels)

        self.meas_start_gps = None  # Initialize measurement start time
        self.timer.start(50)

    def pause_animation(self):
        self.timer.stop()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(True)
        self.btn_abort.setEnabled(True)
        self.nds_cache.online_stop()

    def resume_animation(self):
        self.timer.start(50)
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_abort.setEnabled(True)
        win = self.input_controls["nds_win"].value()
        self.nds_cache.online_start(lookback=win)

    def stop_animation(self):
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.btn_resume.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.nds_cache.reset()
        if hasattr(self, "status_label"):
            self.status_label.setText("")
        self.accumulator.reset()  # Reset accumulator
        self.nds_latest_raw = None
        self.meas_start_gps = None  # Reset measurement start
        for t_list in [self.traces1, self.traces2]:
            for t in t_list:
                t["curve"].setData([], [])
                t["bar"].setOpts(x=[0], height=[0])
                t["img"].clear()
        self.time_counter = 0.0

    def get_ui_params(self):
        p = {}
        c = self.meas_controls
        p.update(
            {
                "start_freq": c["start_freq"].value(),
                "stop_freq": c["stop_freq"].value(),
                "bw": c["bw"].value(),
                "averages": c["averages"].value(),
                "overlap": c["overlap"].value() / 100.0,
                "window": "boxcar"
                if c["window"].currentText().lower() == "uniform"
                else c["window"].currentText().lower(),
            }
        )
        p["avg_type"] = (
            "fixed"
            if c["avg_type_fixed"].isChecked()
            else ("exponential" if c["avg_type_exp"].isChecked() else "accumulative")
        )
        return p

    def update_graphs(self):
        """
        Main update loop for streaming mode.

        This method orchestrates the data pipeline:
        1. Collect data from source (NDS/Simulation)
        2. Apply excitation signals via ExcitationManager
        3. Check stop conditions (Fixed Averaging)
        4. Render results via PlotRenderer
        """
        if self.is_file_mode:
            return

        # Stage 1: Collect data
        data_map, current_times, current_fs = self._collect_data_map()

        if data_map is None and current_times is None:
            return  # No data and no fallback

        if data_map is None:
            data_map = {}

        # Stage 2: Apply excitation signals
        total_excitation = self.excitation_manager.inject_signals(
            data_map, current_times, current_fs
        )
        self.excitation_manager.publish_excitation_channel(
            data_map, total_excitation, current_times, current_fs
        )

        # Stage 3: Apply time cropping and check stop conditions
        self._apply_time_cropping(data_map, current_times)
        self._check_stop_condition(data_map)

        # Stage 4: Render graphs
        try:
            self._render_graphs(data_map, current_times)
        except (RuntimeError, TypeError, ValueError):
            logger.exception("Error rendering graphs")

    def _collect_data_map(self):
        """
        Collect data from the current source into a data_map.

        Returns
        -------
        tuple
            (data_map, current_times, current_fs) where data_map may be empty
            or None if no data is available.
        """
        data_map = {}
        current_times = None
        current_fs = 16384  # Fallback

        # Check if we have Active Excitation for fallback timebase
        has_active_excitation = self.excitation_manager.has_active_excitation()

        if (
            self.data_source in ["NDS", "NDS2", "Simulation"]
            or self.input_controls["pcaudio"].isChecked()
        ):
            # Try to get data from NDS buffers
            if self.nds_latest_raw:
                for ch_name, buf in self.nds_latest_raw.items():
                    y_vals = buf.data_map.get("raw")
                    if (
                        y_vals is not None
                        and len(y_vals) > 0
                        and len(buf.tarray) == len(y_vals)
                    ):
                        ts = TimeSeries(
                            y_vals, t0=buf.tarray[0], dt=buf.step, name=ch_name
                        )
                        data_map[ch_name] = ts
                        if current_times is None:
                            current_times = buf.tarray
                            current_fs = 1.0 / buf.step

            # Fallback if no data but we want to see Excitation (only in NDS mode)
            if (
                not data_map
                and has_active_excitation
                and self.data_source in ["NDS", "NDS2"]
            ):
                self.time_counter += 0.05
                params = self.get_ui_params()
                duration = max(10.0, 2.0 / params.get("bw", 1.0))
                fs = 16384
                current_fs = fs
                t0 = self.time_counter
                n = int(fs * duration)
                current_times = np.linspace(t0, t0 + duration, n, endpoint=False)

            if (
                not data_map
                and current_times is None
                and self.data_source in ["NDS", "NDS2", "Simulation"]
            ):
                return None, None, current_fs  # No data and no fallback

        # SIM Logic (Legacy/Fallback)
        elif self.data_source == "SIM":
            self.time_counter += 0.05
            params = self.get_ui_params()
            self.engine.configure(params)
            duration = max(10.0, 2.0 / params.get("bw", 1.0))
            fs = 16384
            current_fs = fs
            t0 = self.time_counter
            n = int(fs * duration)
            current_times = np.linspace(t0, t0 + duration, n, endpoint=False)

        return data_map, current_times, current_fs

    def _apply_time_cropping(self, data_map, current_times):
        """
        Crop data to measurement start time and remove stale data.

        Parameters
        ----------
        data_map : dict
            Dictionary of channel data. Modified in place.
        current_times : array_like or None
            Current time array (used to initialize meas_start_gps if needed).
        """
        if not data_map:
            return

        # Determine measurement start if not set
        if self.meas_start_gps is None:
            if current_times is not None and len(current_times) > 0:
                self.meas_start_gps = current_times[0]
                logger.debug(f"Measurement Start GPS set to {self.meas_start_gps}")

        # Apply cropping
        if self.meas_start_gps is not None:
            for k, ts in list(data_map.items()):
                # Check for entirely old data
                if ts.t0.value + ts.duration.value <= self.meas_start_gps:
                    del data_map[k]
                    continue

                # Crop to measurement start
                if ts.t0.value < self.meas_start_gps:
                    try:
                        data_map[k] = ts.crop(start=self.meas_start_gps)
                    except (RuntimeError, ValueError):
                        del data_map[k]

    def _check_stop_condition(self, data_map):
        """
        Check if Fixed Averaging stop condition is met.

        Parameters
        ----------
        data_map : dict
            Dictionary of channel data.
        """
        ui_params = self.get_ui_params()

        if ui_params["avg_type"] != "fixed" or not data_map:
            return

        # Calculate required duration
        fft_len = 1.0 / max(ui_params["bw"], 1e-9)
        overlap_pct = ui_params["overlap"]
        overlap_sec = fft_len * overlap_pct
        stride = fft_len - overlap_sec

        target_avgs = max(ui_params["averages"], 1)
        req_duration = fft_len + (target_avgs - 1) * stride

        # Check collected duration
        try:
            ts_check = next(iter(data_map.values()))
            collected = ts_check.duration.value

            if collected >= req_duration:
                logger.debug(
                    f"Fixed Averaging Reached. Collected: {collected:.2f}s, "
                    f"Req: {req_duration:.2f}s"
                )
                self.pause_animation()
        except (AttributeError, StopIteration, TypeError, ValueError) as e:
            logger.debug(f"Error checking stop condition: {e}")

    def _render_graphs(self, data_map, current_times):
        """
        Render analysis results to both graph panels.

        Parameters
        ----------
        data_map : dict
            Dictionary of channel data.
        current_times : array_like or None
            Current time array.
        """
        for plot_idx, info_root in enumerate([self.graph_info1, self.graph_info2]):
            # Update meta info in Param tab
            self._update_panel_meta(info_root, data_map)

            try:
                g_type = info_root["graph_combo"].currentText()
                results = self._get_results(plot_idx, info_root, data_map, g_type)

                # Determine if X axis is Time
                is_time_axis = g_type in ["Time Series", "Spectrogram"]
                start_time_gps = self._get_start_time_gps(
                    is_time_axis, current_times, results
                )

                # Update axis labels
                if is_time_axis and start_time_gps is not None:
                    start_time_utc = self._gps_to_utc(start_time_gps)
                    self.plot_renderer.update_axis_labels(
                        info_root, start_time_gps, start_time_utc
                    )

                # Render traces
                self.plot_renderer.render_panel(
                    plot_idx,
                    info_root,
                    results,
                    g_type,
                    start_time_gps=start_time_gps,
                    is_time_axis=is_time_axis,
                )

                # Stabilize range during streaming
                is_streaming = self.data_source in ["NDS", "NDS2", "Simulation"]
                nds_window = getattr(self, "nds_window", 30.0)
                self.plot_renderer.stabilize_streaming_range(
                    info_root, is_streaming, is_time_axis, nds_window
                )

            except (RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"Error in update_graphs for Graph {plot_idx + 1}: {e}")

    def _update_panel_meta(self, info_root, data_map):
        """Update panel metadata display."""
        ui_p = self.get_ui_params()
        if data_map:
            first_ts = next(iter(data_map.values()), None)
            if first_ts is not None and hasattr(first_ts, "t0"):
                info_root["panel"].meta_info["start_time"] = first_ts.t0.value
                info_root["panel"].meta_info["avgs"] = ui_p.get("averages", 1)
                info_root["panel"].meta_info["bw"] = ui_p.get("bw", 0)
                info_root["panel"].update_params_display()

    def _get_results(self, plot_idx, info_root, data_map, g_type):
        """
        Get analysis results from accumulator or engine.

        Parameters
        ----------
        plot_idx : int
            Index of the graph panel.
        info_root : dict
            Graph info dictionary.
        data_map : dict
            Dictionary of channel data.
        g_type : str
            Graph type string.

        Returns
        -------
        list
            List of results for each trace.
        """
        # Decide source of results
        use_accumulator = (
            self.data_source in ["NDS", "NDS2", "Simulation"]
            or self.input_controls["pcaudio"].isChecked()
        )

        # Fallback to engine if NDS selected but no data arrived
        if (
            use_accumulator
            and self.nds_latest_raw is None
            and not self.input_controls["pcaudio"].isChecked()
            and data_map
        ):
            use_accumulator = False

        if use_accumulator:
            # Use Accumulator
            offset = 0
            if plot_idx == 1:
                offset = len(self.graph_info1["traces"])

            all_results = self.accumulator.get_results()
            logger.debug(f"Accumulator returned {len(all_results)} results")

            n_traces = len(info_root["traces"])
            results = all_results[offset : offset + n_traces]
        else:
            # Legacy Engine (SIM/FILE)
            results = self.engine.compute(
                data_map,
                g_type,
                [
                    {
                        "active": c["active"].isChecked(),
                        "ch_a": c["chan_a"].currentText(),
                        "ch_b": c["chan_b"].currentText(),
                        "gain": c.get("gain").value() if c.get("gain") else 1.0,
                    }
                    for c in info_root["traces"]
                ],
            )

        return results

    def _get_start_time_gps(self, is_time_axis, current_times, results):
        """
        Determine the GPS start time for time-axis graphs.

        Parameters
        ----------
        is_time_axis : bool
            Whether the X axis is time-based.
        current_times : array_like or None
            Current time array.
        results : list
            Analysis results.

        Returns
        -------
        float or None
            GPS start time.
        """
        if not is_time_axis:
            return None

        # Use fixed measurement start if available
        if self.meas_start_gps is not None:
            return self.meas_start_gps

        # Fallback to data start
        if current_times is not None and len(current_times) > 0:
            return current_times[0]

        # Try to find from results
        if results:
            for r in results:
                if r is not None:
                    if isinstance(r, dict) and "times" in r:
                        return r["times"][0]
                    elif isinstance(r, tuple) and len(r) == 2 and len(r[0]) > 0:
                        return r[0][0]

        return None

    def _gps_to_utc(self, gps_time):
        """Convert GPS time to UTC string."""
        try:
            from astropy.time import Time

            t = Time(gps_time, format="gps", scale="utc")
            return t.isot.replace("T", " ")
        except (TypeError, ValueError):
            return "?"

    def on_trace_channel_changed(self):
        if not self.is_loading_file and self.is_file_mode:
            self.update_file_plot()

    def update_file_plot(self):
        if not self.loaded_products:
            return
        for graph_idx in [0, 1]:
            info, traces = (
                (self.graph_info1, self.traces1)
                if graph_idx == 0
                else (self.graph_info2, self.traces2)
            )
            # Update t0 from loaded products
            for p_type in ["TS", "ASD", "PSD", "CSD", "Spectrogram"]:
                items = self.loaded_products.get(p_type)
                if items:
                    first_val = next(iter(items.values()))
                    if hasattr(first_val, "t0"):
                        info["panel"].meta_info["start_time"] = first_val.t0.value
                        info["panel"].update_params_display()
                        break
            try:
                g_type = info["graph_combo"].currentText()
                p_name = "TS"
                if g_type == "Amplitude Spectral Density":
                    p_name = "ASD"
                elif g_type == "Cross Spectral Density":
                    p_name = "CSD"
                elif g_type == "Coherence":
                    p_name = "COH"
                elif g_type == "Transfer Function":
                    p_name = "TF"
                elif g_type == "Spectrogram":
                    p_name = "Spectrogram"

                items = (
                    self.loaded_products.get(p_name)
                    or (self.loaded_products.get("PSD") if p_name == "ASD" else None)
                    or (
                        self.loaded_products.get("Spectrogram")
                        if p_name == "Spectrogram"
                        else None
                    )
                )
                if not items:
                    continue

                for t_idx, ctrl in enumerate(info["traces"]):
                    try:
                        tr = traces[t_idx]
                        print(f"DEBUG: Processing Trace {t_idx}")
                        curve, _bar, img = tr["curve"], tr["bar"], tr["img"]

                        if not ctrl["active"].isChecked():
                            curve.setData([], [])
                            curve.setVisible(False)
                            img.clear()
                            img.setVisible(False)
                            continue

                        ch_a, ch_b = (
                            ctrl["chan_a"].currentText(),
                            ctrl["chan_b"].currentText(),
                        )
                        key = (
                            ch_a
                            if p_name in ["TS", "ASD", "PSD", "Spectrogram"]
                            else (ch_b, ch_a)
                        )
                        val = items.get(key)

                        # Handle Spectrogram (2D)
                        is_spectrogram = (
                            hasattr(val, "ndim")
                            and val.ndim == 2
                            and hasattr(val, "value")
                            and hasattr(val, "times")
                            and hasattr(val, "frequencies")
                        )

                        if is_spectrogram:
                            data = val.value
                            times = val.times.value
                            freqs = val.frequencies.value
                            disp = info.get("units", {}).get("display_y").currentText()

                            if disp == "dB":
                                data = 10 * np.log10(np.abs(data) + 1e-20)
                            elif disp == "Phase":
                                data = (
                                    np.angle(data, deg=True)
                                    if np.iscomplexobj(data)
                                    else np.zeros_like(data)
                                )
                            elif disp == "Magnitude":
                                data = np.abs(data)

                            img.setImage(data, autoLevels=False)
                            img.setLevels([np.min(data), np.max(data)])

                            if len(times) > 1 and len(freqs) > 1:
                                dt = times[1] - times[0]
                                df = freqs[1] - freqs[0]
                                img.setRect(
                                    QtCore.QRectF(
                                        times[0],
                                        freqs[0],
                                        dt * len(times),
                                        df * len(freqs),
                                    )
                                )
                                img.setVisible(True)
                                curve.setVisible(False)
                            else:
                                img.clear()
                            continue

                        # Handle 1D Series
                        res = normalize_series(val) if val is not None else None
                        if res:
                            x, d = res
                            disp = info.get("units", {}).get("display_y").currentText()
                            if disp == "dB":
                                d = (
                                    10
                                    if "Power" in g_type or "Squared" in g_type
                                    else 20
                                ) * np.log10(np.abs(d) + 1e-20)
                            elif disp == "Phase":
                                d = (
                                    np.angle(d, deg=True)
                                    if np.iscomplexobj(d)
                                    else np.zeros_like(d)
                                )
                            elif disp == "Magnitude":
                                d = np.abs(d)
                            elif disp == "Deg Unwrapped":
                                d = (
                                    np.degrees(np.unwrap(np.angle(d)))
                                    if np.iscomplexobj(d)
                                    else np.zeros_like(d)
                                )

                            curve.setData(x, d)
                            curve.setVisible(True)
                            img.setVisible(False)
                        else:
                            curve.setVisible(False)
                            img.setVisible(False)
                    except (RuntimeError, TypeError, ValueError) as e:
                        print(
                            f"Error updating File Plot Graph {graph_idx + 1} Trace {t_idx}: {e}"
                        )
            except (RuntimeError, TypeError, ValueError) as e:
                print(f"Error in update_file_plot for Graph {graph_idx + 1}: {e}")

    def export_data(self):
        """Export current plot data to file (HDF5/CSV/GWF)."""
        filters = [
            "HDF5 (*.h5 *.hdf5)",
            "CSV (*.csv *.txt)",
            "GWF Frame (*.gwf)",
            "All Files (*)",
        ]
        filename, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data", "", ";;".join(filters)
        )
        if not filename:
            return

        # Determine format from filter or extension
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        if "h5" in selected_filter.lower() or ext in ("h5", "hdf5"):
            fmt = "hdf5"
            if not filename.endswith((".h5", ".hdf5")):
                filename += ".h5"
        elif "gwf" in selected_filter.lower() or ext == "gwf":
            fmt = "gwf"
            if not filename.endswith(".gwf"):
                filename += ".gwf"
        else:
            fmt = "csv"
            if not filename.endswith((".csv", ".txt")):
                filename += ".csv"

        try:
            # Collect exportable data from loaded products or engine
            export_items = []
            if self.loaded_products:
                for key, prod in self.loaded_products.items():
                    if hasattr(prod, "write"):
                        export_items.append((key, prod))
            elif self.accumulator and hasattr(self.accumulator, "latest_result"):
                result = self.accumulator.latest_result
                if result and hasattr(result, "write"):
                    export_items.append(("accumulator_result", result))

            if not export_items:
                QtWidgets.QMessageBox.warning(
                    self, "Export", "No data available to export."
                )
                return

            # Write data
            for name, data in export_items:
                if fmt == "hdf5":
                    data.write(filename, format="hdf5", overwrite=True)
                elif fmt == "gwf":
                    data.write(filename, format="gwf")
                elif fmt == "csv":
                    data.write(filename, format="csv")
                break  # Export first item for now (TODO: export all)

            QtWidgets.QMessageBox.information(
                self, "Export", f"Data exported to:\n{filename}"
            )
        except (OSError, RuntimeError, ValueError) as e:
            logger.exception("Export failed")
            QtWidgets.QMessageBox.critical(
                self, "Export Error", f"Failed to export data:\n{e}"
            )

    def open_new_result_window(self):
        """Open a new window sharing the same data (DTT New behavior)."""
        try:
            from .result_window import ResultWindow

            new_win = ResultWindow(loaded_products=self.loaded_products, parent=None)
            new_win.show()
            # Keep reference to prevent garbage collection
            if not hasattr(self, "_child_windows"):
                self._child_windows = []
            self._child_windows.append(new_win)
        except (ImportError, ModuleNotFoundError):
            # ResultWindow not implemented yet - show placeholder message
            QtWidgets.QMessageBox.information(
                self,
                "New Window",
                "New Result Window feature:\n"
                "Opens a separate window sharing the same data.\n\n"
                "(Full implementation pending)",
            )

    def show_reference_dialog(self):
        """Show reference trace management dialog (DTT Reference behavior)."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Reference Traces")
        dialog.setMinimumSize(500, 400)
        layout = QtWidgets.QVBoxLayout(dialog)

        # Instructions
        label = QtWidgets.QLabel(
            "Reference traces allow you to overlay previously saved data\n"
            "onto the current plot for comparison."
        )
        layout.addWidget(label)

        # Reference list
        group = QtWidgets.QGroupBox("Loaded Reference Traces")
        group_layout = QtWidgets.QVBoxLayout(group)

        self._ref_list_widget = QtWidgets.QListWidget()
        # Populate with any existing references
        if hasattr(self, "_reference_traces"):
            for name in self._reference_traces.keys():
                self._ref_list_widget.addItem(name)
        group_layout.addWidget(self._ref_list_widget)

        # Buttons for reference management
        btn_layout = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("Add from File...")
        btn_remove = QtWidgets.QPushButton("Remove")
        btn_toggle = QtWidgets.QPushButton("Show/Hide")
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_remove)
        btn_layout.addWidget(btn_toggle)
        group_layout.addLayout(btn_layout)
        layout.addWidget(group)

        def add_reference():
            filters = ["Data Files (*.xml *.gwf *.h5 *.hdf5)", "All Files (*)"]
            f, _ = QtWidgets.QFileDialog.getOpenFileName(
                dialog, "Load Reference Data", "", ";;".join(filters)
            )
            if f:
                import os

                name = os.path.basename(f)
                if not hasattr(self, "_reference_traces"):
                    self._reference_traces = {}
                # Placeholder: actual loading would use gwpy/loaders
                self._reference_traces[name] = {
                    "file": f,
                    "visible": True,
                    "data": None,
                }
                self._ref_list_widget.addItem(name)

        def remove_reference():
            item = self._ref_list_widget.currentItem()
            if item:
                name = item.text()
                if (
                    hasattr(self, "_reference_traces")
                    and name in self._reference_traces
                ):
                    del self._reference_traces[name]
                self._ref_list_widget.takeItem(self._ref_list_widget.row(item))

        btn_add.clicked.connect(add_reference)
        btn_remove.clicked.connect(remove_reference)

        # Close button
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)

        dialog.exec_()

    def show_calibration_dialog(self):
        """Show calibration table editor dialog (DTT Calibration behavior)."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Calibration Table Editor")
        dialog.setMinimumSize(600, 500)
        layout = QtWidgets.QVBoxLayout(dialog)

        # Instructions
        label = QtWidgets.QLabel(
            "Calibration tables define transfer functions to convert\n"
            "raw ADC/DAC values to physical units (e.g., m/s, strain).\n\n"
            "Use Zeros/Poles or FIR filter coefficients."
        )
        layout.addWidget(label)

        # Tabs for different calibration methods
        cal_tabs = QtWidgets.QTabWidget()

        # --- Tab 1: Channel List ---
        channel_tab = QtWidgets.QWidget()
        ch_layout = QtWidgets.QVBoxLayout(channel_tab)

        ch_table = QtWidgets.QTableWidget()
        ch_table.setColumnCount(4)
        ch_table.setHorizontalHeaderLabels(["Channel", "Gain", "Units", "Filter"])
        ch_table.horizontalHeader().setStretchLastSection(True)

        # Populate with active channels
        active_channels = []
        if hasattr(self, "meas_controls"):
            for state in self.meas_controls.get("channel_states", []):
                if state.get("active"):
                    active_channels.append(state.get("name", ""))

        ch_table.setRowCount(len(active_channels) if active_channels else 2)
        for i, ch_name in enumerate(active_channels[:10]):  # Limit to 10
            ch_table.setItem(i, 0, QtWidgets.QTableWidgetItem(ch_name))
            ch_table.setItem(i, 1, QtWidgets.QTableWidgetItem("1.0"))
            ch_table.setItem(i, 2, QtWidgets.QTableWidgetItem("counts"))
            ch_table.setItem(i, 3, QtWidgets.QTableWidgetItem("None"))

        ch_layout.addWidget(ch_table)
        cal_tabs.addTab(channel_tab, "Channels")

        # --- Tab 2: Zeros/Poles Editor ---
        zp_tab = QtWidgets.QWidget()
        zp_layout = QtWidgets.QFormLayout(zp_tab)

        zp_layout.addRow("Gain:", QtWidgets.QDoubleSpinBox())
        zp_layout.addRow("Zeros (comma sep):", QtWidgets.QLineEdit(""))
        zp_layout.addRow("Poles (comma sep):", QtWidgets.QLineEdit(""))

        cal_tabs.addTab(zp_tab, "Zeros/Poles")

        # --- Tab 3: Import/Export ---
        io_tab = QtWidgets.QWidget()
        io_layout = QtWidgets.QVBoxLayout(io_tab)

        btn_import_cal = QtWidgets.QPushButton("Import Calibration File...")
        btn_export_cal = QtWidgets.QPushButton("Export Calibration File...")
        io_layout.addWidget(btn_import_cal)
        io_layout.addWidget(btn_export_cal)
        io_layout.addStretch(1)

        def import_calibration():
            f, _ = QtWidgets.QFileDialog.getOpenFileName(
                dialog,
                "Import Calibration",
                "",
                "Calibration Files (*.cal *.xml *.txt);;All Files (*)",
            )
            if f:
                QtWidgets.QMessageBox.information(
                    dialog, "Import", f"Loaded calibration from:\n{f}"
                )

        def export_calibration():
            f, _ = QtWidgets.QFileDialog.getSaveFileName(
                dialog,
                "Export Calibration",
                "",
                "Calibration Files (*.cal *.xml);;All Files (*)",
            )
            if f:
                QtWidgets.QMessageBox.information(
                    dialog, "Export", f"Exported calibration to:\n{f}"
                )

        btn_import_cal.clicked.connect(import_calibration)
        btn_export_cal.clicked.connect(export_calibration)

        cal_tabs.addTab(io_tab, "Import/Export")

        layout.addWidget(cal_tabs)

        # Dialog buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Store calibration settings
            if not hasattr(self, "_calibration_table"):
                self._calibration_table = {}
            # Collect data from table
            for row in range(ch_table.rowCount()):
                ch_item = ch_table.item(row, 0)
                gain_item = ch_table.item(row, 1)
                unit_item = ch_table.item(row, 2)
                if ch_item and ch_item.text():
                    self._calibration_table[ch_item.text()] = {
                        "gain": float(gain_item.text()) if gain_item else 1.0,
                        "units": unit_item.text() if unit_item else "counts",
                    }

    def open_file_dialog(self):
        filters = ["Data Files (*.xml *.gwf *.h5 *.hdf5 *.csv *.txt)", "All Files (*)"]
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Data File", "", ";;".join(filters)
        )
        if f:
            self.open_file(f)

    def reset_state(self):
        """Reset the entire GUI and engine state to defaults."""
        self.is_loading_file = True

        # 1. Reset Engine
        self.engine.reset()
        self.loaded_products = {}
        self.is_file_mode = False
        self.time_counter = 0.0

        # 2. Reset Graphs
        # We need to find the GraphPanel instances.
        # In create_result_tab, they were not explicitly stored in self but are children of result_tab.
        # However, we can find them or use the info objects if they have refs.
        # Let's find them from the tab widget if possible, or assume they are accessible.
        # Actually, in MainWindow.__init__, we have self.graph_info1/2 which are from create_result_tab.
        # We should probably store the GraphPanel instances themselves.
        # For now, let's look for them in the Result tab.
        res_tab = self.tabs.widget(3)  # Result tab is index 3
        if res_tab:
            panels = res_tab.findChildren(GraphPanel)
            for p in panels:
                p.reset()

        # 3. Reset Measurement Tab
        self.meas_controls["set_all_channels"]([])

        # 4. Clear NDS status
        self.nds_latest_raw = None

        self.is_loading_file = False

    def open_file(self, filename):
        try:
            # Prepare fresh state
            self.reset_state()
            self.is_loading_file = True

            products_dict = loaders.load_products(filename)
            if not products_dict:
                QtWidgets.QMessageBox.warning(self, "Error", "No products loaded")
                self.is_loading_file = False
                return

            self.loaded_products = products_dict
            self.is_file_mode = True
            self.timer.stop()
            # Extract common channels
            cols = products.extract_channels(products_dict)

            # --- Channel Population Logic ---
            new_states = []
            if filename.lower().endswith(".xml"):
                # Use local XML parser to get Active states (preserves DTT enabled/disabled state)
                xml_channels = extract_xml_channels(filename)
                if xml_channels:
                    new_states = xml_channels

            # Fallback / Non-XML: Use all loaded columns as active channels
            if not new_states:
                new_states = [{"name": c, "active": True} for c in cols]

            self.meas_controls["set_all_channels"](new_states)
            # --- End Channel Population ---

            # --- Update Time / Duration Info ---
            try:
                # Find the first available product to get time info
                first_prod_dict = next(iter(products_dict.values()))
                first_series = next(iter(first_prod_dict.values()))

                t0 = None
                duration = None

                # Check for TimeSeries
                if hasattr(first_series, "t0"):
                    t0 = (
                        first_series.t0.value
                        if hasattr(first_series.t0, "value")
                        else first_series.t0
                    )
                    if hasattr(first_series, "duration"):
                        duration = (
                            first_series.duration.value
                            if hasattr(first_series.duration, "value")
                            else first_series.duration
                        )
                    elif hasattr(first_series, "dt") and hasattr(first_series, "size"):
                        dt = (
                            first_series.dt.value
                            if hasattr(first_series.dt, "value")
                            else first_series.dt
                        )
                        duration = dt * first_series.size

                # Check for FrequencySeries (epoch)
                elif hasattr(first_series, "epoch"):
                    t0 = (
                        first_series.epoch.value
                        if hasattr(first_series.epoch, "value")
                        else first_series.epoch
                    )
                    # Duration for ASD? Maybe unavailable.

                if t0 is not None:
                    # Update Start Time GPS
                    if "start_gps" in self.meas_controls:
                        self.meas_controls["start_gps"].setValue(int(t0))
                    if "rb_gps" in self.meas_controls:
                        self.meas_controls["rb_gps"].setChecked(True)

                    # Update Measurement Time String
                    # Format: MM/DD/YYYY HH:MM:SS UTC
                    import astropy.time

                    t_obj = astropy.time.Time(t0, format="gps", scale="utc")
                    # DTT format example: 06/01/1980 00:00:00 UTC
                    # strftime format: %m/%d/%Y %H:%M:%S UTC
                    t_str = t_obj.strftime("%m/%d/%Y %H:%M:%S UTC")
                    if "meas_time_str" in self.meas_controls:
                        self.meas_controls["meas_time_str"].setText(t_str)

                if duration is not None:
                    if "time_span" in self.meas_controls:
                        self.meas_controls["time_span"].setValue(float(duration))

            except (TypeError, ValueError) as e:
                print(f"Time info extraction failed: {e}")
            # --- End Time Info Update ---

            for g_idx in [0, 1]:
                info = self.graph_info1 if g_idx == 0 else self.graph_info2
                p_name = list(products_dict.keys())[min(g_idx, len(products_dict) - 1)]
                ctype = "Time Series"
                if p_name in ["ASD", "PSD"]:
                    ctype = "Amplitude Spectral Density"
                elif p_name == "CSD":
                    ctype = "Cross Spectral Density"
                elif p_name == "COH":
                    ctype = "Coherence"
                elif p_name in ["TF", "STF"]:
                    ctype = "Transfer Function"
                elif p_name == "Spectrogram":
                    ctype = "Spectrogram"
                info["graph_combo"].blockSignals(True)
                info["graph_combo"].setCurrentText(ctype)
                info["graph_combo"].blockSignals(False)

                # For XML, we rely on the combo boxes being already populated by on_measurement_channel_changed.
                # For non-XML (File mode legacy), we populate them directly here.
                is_xml = filename.lower().endswith(".xml")

                # Auto-assign the first few channels to traces for visualization
                available_keys = list(products_dict[p_name].keys())[:8]
                for t_idx, item_key in enumerate(available_keys):
                    ctrl = info["traces"][t_idx]

                    if not is_xml:
                        # Legacy behavior for non-XML: Populate combos directly
                        for c in ["chan_a", "chan_b"]:
                            if c in ctrl:
                                ctrl[c].clear()
                                ctrl[c].addItems(cols)

                    # Set selection
                    ca, cb = (
                        (str(item_key), "")
                        if not isinstance(item_key, tuple)
                        else (str(item_key[1]), str(item_key[0]))
                    )

                    # For XML, check if these items exist in the combo (they should if populated from Meas tab)
                    if "chan_a" in ctrl:
                        if ctrl["chan_a"].findText(ca) == -1:
                            ctrl["chan_a"].addItem(ca)  # Fallback
                        ctrl["chan_a"].setCurrentText(ca)
                    if "chan_b" in ctrl:
                        if ctrl["chan_b"].findText(cb) == -1:
                            ctrl["chan_b"].addItem(cb)  # Fallback
                        ctrl["chan_b"].setCurrentText(cb)

                    ctrl["active"].setChecked(True)

                # Trigger axis scale update based on graph type (since blockSignals was used)
                if "range_updater" in info:
                    # Set appropriate axis scales based on graph type
                    res_tab = self.tabs.widget(3)
                    if res_tab:
                        panels = res_tab.findChildren(GraphPanel)
                        for p in panels:
                            # Trigger the graph type change handler to set correct axis scales
                            p.graph_combo.currentIndexChanged.emit(
                                p.graph_combo.currentIndex()
                            )

            self.is_loading_file = False
            self.update_file_plot()
        except (OSError, RuntimeError, ValueError) as e:
            self.is_loading_file = False
            import traceback

            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed: {e}")
