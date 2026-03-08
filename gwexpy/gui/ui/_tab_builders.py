"""Extracted tab-builder helpers for GraphPanel._init_ui.

Each function takes a ``panel`` (GraphPanel instance) as its first argument
and sets the required attributes on it.  All functions return the
top-level QWidget for the tab (or, for the setup helpers, nothing).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from gwexpy.gui.ui.graph_panel import _small_spin_dbl, _small_spin_int

if TYPE_CHECKING:
    from gwexpy.gui.ui.graph_panel import GraphPanel

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

COLORS = [
    ("Red", "#FF0000"),
    ("Blue", "#0000FF"),
    ("Green", "#00FF00"),
    ("Black", "#000000"),
    ("Magenta", "#FF00FF"),
    ("Cyan", "#00FFFF"),
    ("Yellow", "#FFFF00"),
    ("Orange", "#FFA500"),
]

COLORS_HEX_LIST = [c[1] for c in COLORS]

LINE_STYLES = [
    ("Solid", Qt.SolidLine),  # type: ignore
    ("Dash", Qt.DashLine),  # type: ignore
    ("Dot", Qt.DotLine),  # type: ignore
]

SYMBOLS = [("Circle", "o"), ("Square", "s"), ("Triangle", "t")]

FILL_PATTERNS = [
    ("Solid", Qt.SolidPattern),  # type: ignore
    ("Dense", Qt.Dense3Pattern),  # type: ignore
]

CHANNEL_NAMES = [
    "HF_sine",
    "LF_sine",
    "beating_sine",
    "white_noise",
    "sine_plus_noise",
    "square_wave",
    "sawtooth_wave",
    "random_walk",
]

# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------


def _clr_box() -> QtWidgets.QComboBox:
    """Create a colour-swatch combo box."""
    lc = QtWidgets.QComboBox()
    lc.setFixedWidth(40)
    for j, c in enumerate(COLORS):
        lc.addItem("")
        lc.setItemData(j, QtGui.QColor(c[1]), QtCore.Qt.BackgroundRole)  # type: ignore
    return lc


# ---------------------------------------------------------------------------
# Tab-bar / stack setup
# ---------------------------------------------------------------------------


def _build_tab_bar_setup(panel: GraphPanel) -> None:
    """Create the two QTabBar rows, the QStackedWidget, and the switching
    logic.  Sets ``panel.tab_row1``, ``panel.tab_row2``, ``panel.main_stack``,
    and returns the outer QVBoxLayout ``pv`` so the caller can keep a
    reference to it.
    """
    pv = QtWidgets.QVBoxLayout(panel)
    pv.setContentsMargins(0, 0, 0, 0)
    pv.setSpacing(0)

    panel.tab_row1 = QtWidgets.QTabBar()
    panel.tab_row1.addTab("Style")
    panel.tab_row1.addTab("X-axis")
    panel.tab_row1.addTab("Y-axis")
    panel.tab_row1.addTab("Legend")
    panel.tab_row1.addTab("Param")

    panel.tab_row2 = QtWidgets.QTabBar()
    panel.tab_row2.addTab("Traces")
    panel.tab_row2.addTab("Range")
    panel.tab_row2.addTab("Units")
    panel.tab_row2.addTab("Cursor")
    panel.tab_row2.addTab("Config")

    BASE_TAB_STYLE = "QTabBar::tab { height: 25px; padding: 2px; border: 1px solid #C0C0C0; border-bottom: none; min-width: 55px; background: #E0E0E0; font-weight: normal; }"
    ACTIVE_TAB_STYLE = (
        BASE_TAB_STYLE
        + " QTabBar::tab:selected { font-weight: bold; background: #FFFFFF; border-bottom: 2px solid #FFFFFF; }"
    )

    panel.tab_row1.setStyleSheet(BASE_TAB_STYLE)
    panel.tab_row2.setStyleSheet(ACTIVE_TAB_STYLE)

    pv.addWidget(panel.tab_row1)
    pv.addWidget(panel.tab_row2)

    panel.main_stack = QtWidgets.QStackedWidget()
    pv.addWidget(panel.main_stack)

    # Tab Switching Logic
    def row1_changed(idx):
        if idx == -1:
            return
        panel.tab_row1.setStyleSheet(ACTIVE_TAB_STYLE)
        panel.tab_row2.blockSignals(True)
        panel.tab_row2.setStyleSheet(BASE_TAB_STYLE)
        panel.tab_row2.setCurrentIndex(-1)
        panel.tab_row2.blockSignals(False)
        panel.main_stack.setCurrentIndex(idx + 5)

    def row2_changed(idx):
        if idx == -1:
            return
        panel.tab_row2.setStyleSheet(ACTIVE_TAB_STYLE)
        panel.tab_row1.blockSignals(True)
        panel.tab_row1.setStyleSheet(BASE_TAB_STYLE)
        panel.tab_row1.setCurrentIndex(-1)
        panel.tab_row1.blockSignals(False)
        panel.main_stack.setCurrentIndex(idx)

    panel.tab_row1.currentChanged.connect(row1_changed)
    panel.tab_row2.currentChanged.connect(row2_changed)

    return pv


# ---------------------------------------------------------------------------
# Traces tab
# ---------------------------------------------------------------------------


def _build_traces_tab(panel: GraphPanel) -> QtWidgets.QWidget:
    """Build the Traces tab with 8 trace pages.

    Sets ``panel.trace_controls``, ``panel.graph_combo``,
    ``panel.trace_tab_ctrl``, and ``panel._update_style``.

    Returns the tab widget.
    """
    traces_tab_widget = QtWidgets.QWidget()
    traces_vbox = QtWidgets.QVBoxLayout(traces_tab_widget)
    traces_vbox.setContentsMargins(2, 2, 2, 2)

    r_graph = QtWidgets.QHBoxLayout()
    r_graph.addWidget(QtWidgets.QLabel("Graph:"))
    panel.graph_combo = QtWidgets.QComboBox()
    panel.graph_combo.addItems(
        [
            "Time Series",
            "Amplitude Spectral Density",
            "Cross Spectral Density",
            "Coherence",
            "Squared Coherence",
            "Transfer Function",
            "Spectrogram",
        ]
    )
    r_graph.addWidget(panel.graph_combo)
    traces_vbox.addLayout(r_graph)

    panel.trace_tab_ctrl = QtWidgets.QTabWidget()
    panel.trace_tab_ctrl.setStyleSheet("""
        QTabBar::tab { height: 25px; width: 32px; margin: 0; padding: 0; background: #E0E0E0; }
        QTabBar::tab:selected { font-weight: bold; background: #FFFFFF; }
    """)
    panel.trace_tab_ctrl.setUsesScrollButtons(False)
    traces_vbox.addWidget(panel.trace_tab_ctrl)

    def update_style(t_idx):
        ctrl = panel.trace_controls[t_idx]
        target_curve = panel.traces_items[t_idx]["curve"]
        target_bar = panel.traces_items[t_idx]["bar"]
        target_img = panel.traces_items[t_idx]["img"]

        g_type = panel.graph_combo.currentText()
        is_active = ctrl["active"].isChecked()
        is_spec_gram = g_type == "Spectrogram"

        dual = g_type in [
            "Cross Spectral Density",
            "Coherence",
            "Squared Coherence",
            "Transfer Function",
        ]
        ctrl["chan_b"].setEnabled(dual)
        ctrl["g_style"].setEnabled(not is_spec_gram)

        if not is_active:
            target_curve.setPen(None)
            target_curve.setSymbol(None)
            target_bar.setVisible(False)
            target_img.setVisible(False)
            return

        if is_spec_gram:
            target_curve.setPen(None)
            target_curve.setSymbol(None)
            target_bar.setVisible(False)
            target_img.setVisible(True)
            if not hasattr(target_img, "_lut_set"):
                try:
                    lut = pg.colormap.get("viridis").getLookupTable()
                    target_img.setLookupTable(lut)
                    target_img._lut_set = True
                except (AttributeError, KeyError, ValueError):
                    pass
            return

        if ctrl["line_chk"].isChecked():
            c_hex = COLORS[ctrl["line_c"].currentIndex()][1]
            ctrl["line_c"].setStyleSheet(f"background-color: {c_hex};")
            pen = pg.mkPen(
                color=c_hex,
                width=ctrl["line_w"].value(),
                style=ctrl["line_s"].itemData(ctrl["line_s"].currentIndex()),
            )
            target_curve.setPen(pen)
            target_bar.setVisible(False)
            target_img.setVisible(False)
        elif ctrl["bar_chk"].isChecked():
            c_hex = COLORS[ctrl["bar_c"].currentIndex()][1]
            ctrl["bar_c"].setStyleSheet(f"background-color: {c_hex};")
            target_curve.setPen(None)
            target_bar.setVisible(True)
            target_img.setVisible(False)
            brush = QtGui.QBrush(
                QtGui.QColor(c_hex),
                ctrl["bar_s"].itemData(ctrl["bar_s"].currentIndex()),
            )
            target_bar.setOpts(brush=brush, width=ctrl["bar_w"].value())
        else:
            target_curve.setPen(None)
            target_bar.setVisible(False)
            target_img.setVisible(False)

        if ctrl["sym_chk"].isChecked():
            c_hex = COLORS[ctrl["sym_c"].currentIndex()][1]
            ctrl["sym_c"].setStyleSheet(f"background-color: {c_hex};")
            target_curve.setSymbol(
                ctrl["sym_s"].itemData(ctrl["sym_s"].currentIndex())
            )
            target_curve.setSymbolBrush(c_hex)
            target_curve.setSymbolPen(c_hex)
            target_curve.setSymbolSize(ctrl["sym_w"].value())
        else:
            target_curve.setSymbol(None)

        if hasattr(panel, "update_legend"):
            panel.update_legend()

        if hasattr(panel, "update_cursor_visibility"):
            panel.update_cursor_visibility()

    # Expose so other tabs can call it
    panel._update_style = update_style

    for i in range(8):
        _build_trace_page(panel, i, update_style)

    return traces_tab_widget


def _build_trace_page(panel: GraphPanel, i: int, update_style: Callable[[int], None]) -> QtWidgets.QWidget:
    """Build a single trace page and append the control set to
    ``panel.trace_controls``.
    """
    page = QtWidgets.QWidget()
    pl = QtWidgets.QVBoxLayout(page)
    pl.setContentsMargins(4, 4, 4, 4)
    pl.setSpacing(2)
    active_chk = QtWidgets.QCheckBox("Active")
    active_chk.setChecked(i == 0)
    pl.addWidget(active_chk)
    gc = QtWidgets.QGroupBox("Channels")
    gl = QtWidgets.QGridLayout(gc)
    gl.setContentsMargins(4, 4, 4, 4)
    gl.addWidget(QtWidgets.QLabel("A:"), 0, 0)
    ca = QtWidgets.QComboBox()
    ca.setEditable(True)
    ca.addItems(CHANNEL_NAMES)
    gl.addWidget(ca, 0, 1)
    gl.addWidget(QtWidgets.QLabel("B:"), 1, 0)
    cb = QtWidgets.QComboBox()
    cb.setEditable(True)
    cb.addItems(CHANNEL_NAMES)
    gl.addWidget(cb, 1, 1)

    gl.addWidget(QtWidgets.QLabel("Gain:"), 2, 0)
    sb_gain = _small_spin_dbl(decimals=3, width=80, step=0.1)
    sb_gain.setValue(1.0)
    gl.addWidget(sb_gain, 2, 1)
    pl.addWidget(gc)
    gs = QtWidgets.QGroupBox("Style")
    gls = QtWidgets.QGridLayout(gs)
    gls.setContentsMargins(4, 4, 4, 4)
    lchk = QtWidgets.QCheckBox("Line")
    lchk.setChecked(True)
    gls.addWidget(lchk, 0, 0)

    lc = _clr_box()
    lc.setCurrentIndex(i % len(COLORS))
    gls.addWidget(lc, 0, 1)
    ls = QtWidgets.QComboBox()
    [ls.addItem(n, v) for n, v in LINE_STYLES]
    gls.addWidget(ls, 0, 2)
    lw = _small_spin_int(1, 10, 35)
    gls.addWidget(lw, 0, 3)
    schk = QtWidgets.QCheckBox("Symbol")
    gls.addWidget(schk, 1, 0)
    sc = _clr_box()
    sc.setCurrentIndex(i % len(COLORS))
    gls.addWidget(sc, 1, 1)
    ss = QtWidgets.QComboBox()
    [ss.addItem(n, v) for n, v in SYMBOLS]
    gls.addWidget(ss, 1, 2)
    sw = _small_spin_dbl(1, 35, 1, 50)
    sw.setValue(5)
    gls.addWidget(sw, 1, 3)
    bchk = QtWidgets.QCheckBox("Bar")
    gls.addWidget(bchk, 2, 0)
    bc = _clr_box()
    bc.setCurrentIndex(i % len(COLORS))
    gls.addWidget(bc, 2, 1)
    bs = QtWidgets.QComboBox()
    [bs.addItem(n, v) for n, v in FILL_PATTERNS]
    gls.addWidget(bs, 2, 2)
    bw = _small_spin_dbl(2, 35, 0.01, 10, 0.01)
    bw.setValue(0.04)
    gls.addWidget(bw, 2, 3)
    pl.addWidget(gs)
    panel.trace_tab_ctrl.addTab(page, str(i))
    ctrl_set = {
        "active": active_chk,
        "chan_a": ca,
        "chan_b": cb,
        "gain": sb_gain,
        "g_style": gs,
        "line_chk": lchk,
        "line_c": lc,
        "line_s": ls,
        "line_w": lw,
        "sym_chk": schk,
        "sym_c": sc,
        "sym_s": ss,
        "sym_w": sw,
        "bar_chk": bchk,
        "bar_c": bc,
        "bar_s": bs,
        "bar_w": bw,
    }
    panel.trace_controls.append(ctrl_set)

    active_chk.toggled.connect(lambda _, x=i: update_style(x))
    ca.currentIndexChanged.connect(lambda _, x=i: update_style(x))
    cb.currentIndexChanged.connect(lambda _, x=i: update_style(x))

    def _handle_lchk(checked, x=i, b=bchk):
        if checked:
            b.setChecked(False)
        update_style(x)

    lchk.toggled.connect(_handle_lchk)

    def _handle_bchk(checked, x=i, line_chk=lchk):
        if checked:
            line_chk.setChecked(False)
        update_style(x)

    bchk.toggled.connect(_handle_bchk)
    for w in [lc, ls, lw, schk, sc, ss, sw, bc, bs, bw, sb_gain]:
        if isinstance(w, QtWidgets.QComboBox):
            w.currentIndexChanged.connect(lambda _, x=i: update_style(x))
        else:
            w.valueChanged.connect(lambda _, x=i: update_style(x)) if hasattr(
                w, "valueChanged"
            ) else w.toggled.connect(lambda _, x=i: update_style(x))


# ---------------------------------------------------------------------------
# Range tab
# ---------------------------------------------------------------------------


def _build_range_tab(panel: GraphPanel) -> QtWidgets.QWidget:
    """Build the Range tab (Y/X axis ranges).

    Sets ``panel.update_range_logic``, ``panel.rb_y_lin``, ``panel.rb_y_log``,
    ``panel.rb_y_auto``, ``panel.rb_x_lin``, ``panel.rb_x_log``,
    ``panel.rb_x_auto``, and range spin-box references used by
    ``update_axis_labels``.

    Returns the tab widget.
    """
    range_tab_widget = QtWidgets.QWidget()
    range_vbox = QtWidgets.QVBoxLayout(range_tab_widget)
    range_vbox.setContentsMargins(4, 4, 4, 4)

    # --- Y axis group ---
    gy = QtWidgets.QGroupBox("Y axis")
    gly = QtWidgets.QGridLayout(gy)
    gly.addWidget(QtWidgets.QLabel("Scale:"), 0, 0)
    rb_y_lin = QtWidgets.QRadioButton("linear")
    rb_y_log = QtWidgets.QRadioButton("log")
    rb_y_lin.setChecked(True)
    panel.rb_y_lin = rb_y_lin
    panel.rb_y_log = rb_y_log
    bg_y_scale = QtWidgets.QButtonGroup(panel)
    bg_y_scale.addButton(rb_y_lin)
    bg_y_scale.addButton(rb_y_log)
    hly1 = QtWidgets.QHBoxLayout()
    hly1.addWidget(rb_y_lin)
    hly1.addWidget(rb_y_log)
    gly.addLayout(hly1, 0, 1)
    gly.addWidget(QtWidgets.QLabel("Range:"), 1, 0)
    rb_y_auto = QtWidgets.QRadioButton("automatic")
    rb_y_man = QtWidgets.QRadioButton("manual")
    rb_y_auto.setChecked(True)
    panel.rb_y_auto = rb_y_auto
    bg_y_range = QtWidgets.QButtonGroup(panel)
    bg_y_range.addButton(rb_y_auto)
    bg_y_range.addButton(rb_y_man)
    hly2 = QtWidgets.QHBoxLayout()
    hly2.addWidget(rb_y_auto)
    hly2.addWidget(rb_y_man)
    gly.addLayout(hly2, 1, 1)
    gly.addWidget(QtWidgets.QLabel("From"), 2, 0)
    sb_y_from = _small_spin_dbl(2, 60)
    gly.addWidget(sb_y_from, 2, 1)
    gly.addWidget(QtWidgets.QLabel("To"), 2, 2)
    sb_y_to = _small_spin_dbl(2, 60, max_val=1e12)
    sb_y_to.setValue(1.1)
    gly.addWidget(sb_y_to, 2, 3)
    range_vbox.addWidget(gy)

    # --- X axis group ---
    gx = QtWidgets.QGroupBox("X axis")
    glx = QtWidgets.QGridLayout(gx)
    glx.addWidget(QtWidgets.QLabel("Scale:"), 0, 0)
    rb_x_lin = QtWidgets.QRadioButton("linear")
    rb_x_log = QtWidgets.QRadioButton("log")
    rb_x_lin.setChecked(True)
    panel.rb_x_lin = rb_x_lin
    panel.rb_x_log = rb_x_log
    bg_x_scale = QtWidgets.QButtonGroup(panel)
    bg_x_scale.addButton(rb_x_lin)
    bg_x_scale.addButton(rb_x_log)
    hlx1 = QtWidgets.QHBoxLayout()
    hlx1.addWidget(rb_x_lin)
    hlx1.addWidget(rb_x_log)
    glx.addLayout(hlx1, 0, 1)
    glx.addWidget(QtWidgets.QLabel("Range:"), 1, 0)
    rb_x_auto = QtWidgets.QRadioButton("automatic")
    rb_x_man = QtWidgets.QRadioButton("manual")
    rb_x_auto.setChecked(True)
    panel.rb_x_auto = rb_x_auto
    bg_x_range = QtWidgets.QButtonGroup(panel)
    bg_x_range.addButton(rb_x_auto)
    bg_x_range.addButton(rb_x_man)
    hlx2 = QtWidgets.QHBoxLayout()
    hlx2.addWidget(rb_x_auto)
    hlx2.addWidget(rb_x_man)
    glx.addLayout(hlx2, 1, 1)
    glx.addWidget(QtWidgets.QLabel("From"), 2, 0)
    sb_x_from = _small_spin_dbl(2, 60)
    glx.addWidget(sb_x_from, 2, 1)
    glx.addWidget(QtWidgets.QLabel("To"), 2, 2)
    sb_x_to = _small_spin_dbl(2, 60, max_val=1e12)
    sb_x_to.setValue(10)
    glx.addWidget(sb_x_to, 2, 3)
    range_vbox.addWidget(gx)

    def update_range_logic():
        y_log = rb_y_log.isChecked()
        y_auto = rb_y_auto.isChecked()
        panel.target_plot.setLogMode(y=y_log)
        if y_auto:
            panel.target_plot.enableAutoRange(axis="y")
        else:
            panel.target_plot.setYRange(
                sb_y_from.value(), sb_y_to.value(), padding=0
            )
        x_log = rb_x_log.isChecked()
        x_auto = rb_x_auto.isChecked()
        panel.target_plot.setLogMode(x=x_log)
        if x_auto:
            panel.target_plot.enableAutoRange(axis="x", enable=True)
        else:
            panel.target_plot.enableAutoRange(axis="x", enable=False)
            xmin, xmax = sb_x_from.value(), sb_x_to.value()
            if xmin < xmax:
                panel.target_plot.setXRange(xmin, xmax, padding=0)

    panel.update_range_logic = update_range_logic

    # Store range spin boxes for cross-tab access (update_axis_labels)
    panel._range_sb_y_from = sb_y_from
    panel._range_sb_y_to = sb_y_to
    panel._range_rb_y_man = rb_y_man

    for w in [
        rb_y_lin,
        rb_y_log,
        rb_y_auto,
        rb_y_man,
        sb_y_from,
        sb_y_to,
        rb_x_lin,
        rb_x_log,
        rb_x_auto,
        rb_x_man,
        sb_x_from,
        sb_x_to,
    ]:
        if isinstance(w, QtWidgets.QRadioButton):
            w.toggled.connect(update_range_logic)
        else:
            w.valueChanged.connect(update_range_logic)

    return range_tab_widget


# ---------------------------------------------------------------------------
# Units tab
# ---------------------------------------------------------------------------


def _build_units_tab(panel: GraphPanel) -> QtWidgets.QWidget:
    """Build the Units tab.  Sets ``panel.display_y_combo``.

    Returns the tab widget.
    """
    units_tab_widget = QtWidgets.QWidget()
    ul = QtWidgets.QVBoxLayout(units_tab_widget)
    ul.setContentsMargins(4, 4, 4, 4)
    ug = QtWidgets.QGroupBox("Units")
    ugl = QtWidgets.QGridLayout(ug)
    ugl.addWidget(QtWidgets.QLabel("X:"), 0, 0)
    uxa = QtWidgets.QComboBox()
    uxa.addItems(["-", "s", "Hz"])
    ugl.addWidget(uxa, 0, 1)
    ugl.addWidget(QtWidgets.QLabel("Y:"), 1, 0)
    uya = QtWidgets.QComboBox()
    uya.addItems(["-", "m", "V", "pk/rtHz"])
    ugl.addWidget(uya, 1, 1)
    ul.addWidget(ug)
    dg = QtWidgets.QGroupBox("Display")
    dgl = QtWidgets.QGridLayout(dg)
    dgl.addWidget(QtWidgets.QLabel("Y:"), 1, 0)
    panel.display_y_combo = QtWidgets.QComboBox()
    panel.display_y_combo.addItems(["None", "Magnitude", "Phase", "dB"])
    dgl.addWidget(panel.display_y_combo, 1, 1)
    ul.addWidget(dg)

    return units_tab_widget


# ---------------------------------------------------------------------------
# Cursor tab
# ---------------------------------------------------------------------------


def _build_cursor_tab(panel: GraphPanel) -> QtWidgets.QWidget:
    """Build the Cursor tab.

    Sets ``panel.cursor_trace_tab``, ``panel.cursor_states``,
    ``panel.update_cursor_visibility``.

    Returns the tab widget.
    """
    cursor_tab_widget = QtWidgets.QWidget()
    cl = QtWidgets.QVBoxLayout(cursor_tab_widget)
    cl.setContentsMargins(4, 4, 4, 4)
    cl.addWidget(QtWidgets.QLabel("Trace:"))
    ctabs = QtWidgets.QTabBar()
    ctabs.setExpanding(False)
    [ctabs.addTab(str(i)) for i in range(8)]
    cl.addWidget(ctabs)
    panel.cursor_trace_tab = ctabs

    r_curs = QtWidgets.QHBoxLayout()
    ag = QtWidgets.QGroupBox("Active")
    agl = QtWidgets.QVBoxLayout(ag)
    cur_act1 = QtWidgets.QCheckBox("1")
    cur_act2 = QtWidgets.QCheckBox("2")
    agl.addWidget(cur_act1)
    agl.addWidget(cur_act2)
    r_curs.addWidget(ag)
    stg = QtWidgets.QGroupBox("Style")
    stgl = QtWidgets.QGridLayout(stg)
    rb_style_none = QtWidgets.QRadioButton("None")
    rb_style_vert = QtWidgets.QRadioButton("Vert.")
    rb_style_cross = QtWidgets.QRadioButton("Cross")
    rb_style_horiz = QtWidgets.QRadioButton("Horiz.")
    rb_style_none.setChecked(True)
    stgl.addWidget(rb_style_none, 0, 0)
    stgl.addWidget(rb_style_vert, 0, 1)
    stgl.addWidget(rb_style_cross, 1, 0)
    stgl.addWidget(rb_style_horiz, 1, 1)
    r_curs.addWidget(stg)
    tyg = QtWidgets.QGroupBox("Type")
    tygl = QtWidgets.QVBoxLayout(tyg)
    rb_type_abs = QtWidgets.QRadioButton("Abs.")
    rb_type_delta = QtWidgets.QRadioButton("Delta")
    rb_type_abs.setChecked(True)
    tygl.addWidget(rb_type_abs)
    tygl.addWidget(rb_type_delta)
    r_curs.addWidget(tyg)
    cl.addLayout(r_curs)

    vg = QtWidgets.QGroupBox("Values")
    vgl = QtWidgets.QGridLayout(vg)
    txt_x1 = QtWidgets.QLineEdit()
    txt_y1 = QtWidgets.QLineEdit()
    txt_x2 = QtWidgets.QLineEdit()
    txt_y2 = QtWidgets.QLineEdit()
    lab_x1 = QtWidgets.QLabel("X1:")
    lab_y1 = QtWidgets.QLabel("Y1:")
    lab_x2 = QtWidgets.QLabel("X2:")
    lab_y2 = QtWidgets.QLabel("Y2:")
    vgl.addWidget(lab_x1, 0, 0)
    vgl.addWidget(txt_x1, 0, 1)
    vgl.addWidget(lab_y1, 0, 2)
    vgl.addWidget(txt_y1, 0, 3)
    vgl.addWidget(lab_x2, 1, 0)
    vgl.addWidget(txt_x2, 1, 1)
    vgl.addWidget(lab_y2, 1, 2)
    vgl.addWidget(txt_y2, 1, 3)
    cl.addWidget(vg)

    # Cursor Logic
    cursors = {
        "c1_v": pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("g", width=1.5)
        ),
        "c1_h": pg.InfiniteLine(
            angle=0, movable=True, pen=pg.mkPen("g", width=1.5)
        ),
        "c2_v": pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("y", width=1.5)
        ),
        "c2_h": pg.InfiniteLine(
            angle=0, movable=True, pen=pg.mkPen("y", width=1.5)
        ),
    }

    for line in cursors.values():
        panel.target_plot.addItem(line)
        line.setVisible(False)
        line.setZValue(2000)

    panel.cursor_states = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}

    def update_cursor_values():
        # 1. Determine active trace data
        t_idx = ctabs.currentIndex()
        curve = panel.traces_items[t_idx]["curve"]
        try:
            x_data, y_data = curve.getData()
        except (RuntimeError, TypeError):
            x_data, y_data = None, None

        # Snap logic for cursors (nearest data point)
        def get_snapped_val(val, vertical=True):
            if x_data is None or len(x_data) == 0:
                return val, 0.0
            if vertical:
                idx = np.searchsorted(x_data, val)
                idx = np.clip(idx, 0, len(x_data) - 1)
                if idx > 0 and abs(x_data[idx - 1] - val) < abs(x_data[idx] - val):
                    idx -= 1
                return x_data[idx], y_data[idx]
            else:
                return val, val

        is_delta = rb_type_delta.isChecked()
        lab_x2.setText("DX:" if is_delta else "X2:")
        lab_y2.setText("DY:" if is_delta else "Y2:")

        # Update C1
        x1_snap, y1_snap = get_snapped_val(cursors["c1_v"].value())
        panel.cursor_states["x1"] = x1_snap
        panel.cursor_states["y1"] = (
            y1_snap if rb_style_cross.isChecked() else cursors["c1_h"].value()
        )
        if rb_style_cross.isChecked():
            cursors["c1_h"].setValue(y1_snap)

        # Update C2
        x2_snap, y2_snap = get_snapped_val(cursors["c2_v"].value())
        panel.cursor_states["x2"] = x2_snap
        panel.cursor_states["y2"] = (
            y2_snap if rb_style_cross.isChecked() else cursors["c2_h"].value()
        )
        if rb_style_cross.isChecked():
            cursors["c2_h"].setValue(y2_snap)

        # Update Labels (avoid infinite loop with manual edit)
        txt_x1.setText(f"{panel.cursor_states['x1']:.6g}")
        txt_y1.setText(f"{panel.cursor_states['y1']:.6g}")
        if is_delta:
            txt_x2.setText(
                f"{(panel.cursor_states['x2'] - panel.cursor_states['x1']):.6g}"
            )
            txt_y2.setText(
                f"{(panel.cursor_states['y2'] - panel.cursor_states['y1']):.6g}"
            )
        else:
            txt_x2.setText(f"{panel.cursor_states['x2']:.6g}")
            txt_y2.setText(f"{panel.cursor_states['y2']:.6g}")

    for k in ["c1_v", "c1_h", "c2_v", "c2_h"]:
        cursors[k].sigPositionChanged.connect(lambda: update_cursor_values())

    def on_cursor_manual_edited():
        try:
            cursors["c1_v"].setValue(float(txt_x1.text()))
            cursors["c1_h"].setValue(float(txt_y1.text()))
            if not rb_type_delta.isChecked():
                cursors["c2_v"].setValue(float(txt_x2.text()))
                cursors["c2_h"].setValue(float(txt_y2.text()))
        except (TypeError, ValueError):
            pass

    for t in [txt_x1, txt_y1, txt_x2, txt_y2]:
        t.editingFinished.connect(on_cursor_manual_edited)

    def update_cursor_visibility():
        style = "None"
        if rb_style_vert.isChecked():
            style = "Vert"
        elif rb_style_cross.isChecked():
            style = "Cross"
        elif rb_style_horiz.isChecked():
            style = "Horiz"
        act1, act2 = cur_act1.isChecked(), cur_act2.isChecked()

        # --- COLOR SYNC ---
        t_idx = ctabs.currentIndex()
        ctrl = panel.trace_controls[t_idx]
        # Use color from Trace settings
        if ctrl["line_chk"].isChecked():
            c_idx = ctrl["line_c"].currentIndex()
        elif ctrl["bar_chk"].isChecked():
            c_idx = ctrl["bar_c"].currentIndex()
        elif ctrl["sym_chk"].isChecked():
            c_idx = ctrl["sym_c"].currentIndex()
        else:
            c_idx = 0  # Fallback Red

        c_hex = COLORS[c_idx][1]
        pen1 = pg.mkPen(color=c_hex, width=1.5)
        # Second cursor same color (per request)
        pen2 = pg.mkPen(color=c_hex, width=1.5, style=Qt.DashLine)  # type: ignore

        cursors["c1_v"].setPen(pen1)
        cursors["c1_h"].setPen(pen1)
        cursors["c2_v"].setPen(pen2)
        cursors["c2_h"].setPen(pen2)
        # -----------------

        cursors["c1_v"].setVisible(act1 and style in ["Vert", "Cross"])
        cursors["c1_h"].setVisible(act1 and style in ["Cross", "Horiz"])
        cursors["c2_v"].setVisible(act2 and style in ["Vert", "Cross"])
        cursors["c2_h"].setVisible(act2 and style in ["Cross", "Horiz"])
        update_cursor_values()

    panel.update_cursor_visibility = update_cursor_visibility
    for w in [
        cur_act1,
        cur_act2,
        rb_style_none,
        rb_style_vert,
        rb_style_cross,
        rb_style_horiz,
        rb_type_abs,
        rb_type_delta,
    ]:
        w.toggled.connect(update_cursor_visibility)
    ctabs.currentChanged.connect(update_cursor_visibility)
    update_cursor_visibility()

    return cursor_tab_widget


# ---------------------------------------------------------------------------
# Style tab
# ---------------------------------------------------------------------------


def _build_style_tab(panel: GraphPanel) -> QtWidgets.QWidget:
    """Build the Style tab (title, margins).

    Sets ``panel._style_title_edit``, ``panel._apply_title_style``,
    ``panel._apply_margins``.

    Returns the tab widget.
    """
    style_tab_widget = QtWidgets.QWidget()
    sl = QtWidgets.QVBoxLayout(style_tab_widget)
    sl.setContentsMargins(4, 4, 4, 4)
    sl.setSpacing(6)

    # Title Group
    tg = QtWidgets.QGroupBox("Title")
    tgl = QtWidgets.QGridLayout(tg)
    tgl.setContentsMargins(6, 8, 6, 6)
    tgl.setSpacing(4)

    style_title_edit = QtWidgets.QLineEdit("Time series")
    tgl.addWidget(style_title_edit, 0, 0, 1, 5)

    # Font settings row
    cb_font_fam = QtWidgets.QComboBox()
    cb_font_fam.addItems(["Helvetica", "Times", "Courier", "Arial"])
    tgl.addWidget(cb_font_fam, 1, 2)

    cb_font_weight = QtWidgets.QComboBox()
    cb_font_weight.addItems(["normal", "bold", "italic"])
    cb_font_weight.setCurrentText("bold")
    tgl.addWidget(cb_font_weight, 1, 3)

    sb_font_size = _small_spin_dbl(3, 65, 0.001, 1.0, 0.005)
    sb_font_size.setValue(0.100)
    tgl.addWidget(sb_font_size, 1, 4)

    # Alignment and color row
    align_layout = QtWidgets.QHBoxLayout()
    rb_left = QtWidgets.QRadioButton("Left")
    rb_center = QtWidgets.QRadioButton("Center")
    rb_right = QtWidgets.QRadioButton("Right")
    rb_center.setChecked(True)
    align_layout.addWidget(rb_left)
    align_layout.addWidget(rb_center)
    align_layout.addWidget(rb_right)
    tgl.addLayout(align_layout, 2, 0, 1, 3)

    title_clr = _clr_box()
    title_clr.setFixedWidth(60)
    title_clr.setCurrentIndex(3)  # Default to Black
    tgl.addWidget(title_clr, 2, 4, Qt.AlignRight)  # type: ignore

    sl.addWidget(tg)

    # Margins Group
    mg = QtWidgets.QGroupBox("Margins")
    mgl = QtWidgets.QHBoxLayout(mg)
    mgl.setContentsMargins(6, 6, 6, 6)

    margin_sbs = {}
    for label_txt, val in [("L", 0.05), ("R", 0.05), ("T", 0.05), ("B", 0.05)]:
        mgl.addWidget(QtWidgets.QLabel(label_txt))
        m_sb = _small_spin_dbl(2, 60, 0.0, 1.0, 0.01)
        m_sb.setValue(val)
        mgl.addWidget(m_sb)
        margin_sbs[label_txt] = m_sb
        mgl.addSpacing(4)
    mgl.addStretch(1)
    sl.addWidget(mg)

    sl.addStretch(1)

    # Signal Connections
    def apply_title_style():
        text = style_title_edit.text()
        if not text:
            panel.target_plot.setTitle(None)
            return

        # Map values
        c_idx = title_clr.currentIndex()
        color = COLORS_HEX_LIST[c_idx] if c_idx < len(COLORS_HEX_LIST) else "#000000"

        family = cb_font_fam.currentText()
        size = f"{sb_font_size.value() * 100}pt"  # Scale normalized value to points
        is_bold = cb_font_weight.currentText() == "bold"
        is_italic = cb_font_weight.currentText() == "italic"

        # Alignment
        justify = "center"
        if rb_left.isChecked():
            justify = "left"
        elif rb_right.isChecked():
            justify = "right"

        panel.target_plot.setTitle(
            text,
            color=color,
            size=size,
            bold=is_bold,
            italic=is_italic,
            justify=justify,
            family=family,
        )

    def apply_margins():
        # diaggui margins are often normalized. Here we estimate pixels for pyqtgraph setContentsMargins.
        # Usually these are 0.0 - 1.0. Let's multiply by a factor (e.g. 200) for visual effect.
        left = margin_sbs["L"].value() * 200
        r = margin_sbs["R"].value() * 200
        t = margin_sbs["T"].value() * 200
        b = margin_sbs["B"].value() * 200
        panel.target_plot.getPlotItem().setContentsMargins(left, t, r, b)

    # Connect title signals
    style_title_edit.textChanged.connect(apply_title_style)
    cb_font_fam.currentIndexChanged.connect(apply_title_style)
    cb_font_weight.currentIndexChanged.connect(apply_title_style)
    sb_font_size.valueChanged.connect(apply_title_style)
    rb_left.toggled.connect(apply_title_style)
    rb_center.toggled.connect(apply_title_style)
    rb_right.toggled.connect(apply_title_style)
    title_clr.currentIndexChanged.connect(apply_title_style)

    # Connect margin signals
    for label_txt, sb in margin_sbs.items():
        sb.valueChanged.connect(apply_margins)

    # Expose for cross-tab access
    panel._style_title_edit = style_title_edit
    panel._apply_title_style = apply_title_style
    panel._apply_margins = apply_margins

    return style_tab_widget


# ---------------------------------------------------------------------------
# Axis tab (parameterized for X / Y)
# ---------------------------------------------------------------------------


def _build_axis_tab(panel: GraphPanel, axis: str) -> QtWidgets.QWidget:
    """Build an axis configuration tab.

    *axis* must be ``"x"`` or ``"y"``.

    Sets ``panel.xaxis_tab_widget`` / ``panel.yaxis_tab_widget`` and
    ``panel.xaxis_ctrls`` / ``panel.yaxis_ctrls``.

    Returns ``(tab_widget, ctrls_dict)``.
    """
    default_title = "Time" if axis == "x" else "Signal"

    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(2)

    # Title Group
    tg = QtWidgets.QGroupBox("Title")
    tgl = QtWidgets.QGridLayout(tg)
    tgl.setContentsMargins(4, 6, 4, 4)
    tgl.setSpacing(2)
    title_edit = QtWidgets.QLineEdit(default_title)
    tgl.addWidget(title_edit, 0, 0, 1, 6)
    tgl.addWidget(QtWidgets.QLabel("Size:"), 1, 0)
    t_size = _small_spin_dbl(3, 60, 0, 1, 0.001)
    t_size.setValue(0.042)
    tgl.addWidget(t_size, 1, 1)
    tgl.addWidget(QtWidgets.QLabel("Offset:"), 1, 2)
    t_off = _small_spin_dbl(2, 60, -10, 10, 0.1)
    t_off.setValue(1.40)
    tgl.addWidget(t_off, 1, 3)
    t_clr = _clr_box()
    t_clr.setCurrentIndex(3)
    tgl.addWidget(t_clr, 1, 5, QtCore.Qt.AlignRight)  # type: ignore
    layout.addWidget(tg)

    # Ticks/Axis Group
    tag = QtWidgets.QGroupBox("Ticks/Axis")
    tagl = QtWidgets.QGridLayout(tag)
    tagl.setContentsMargins(4, 6, 4, 4)
    tagl.setSpacing(2)
    tagl.addWidget(QtWidgets.QLabel("Length:"), 0, 0)
    tk_len = _small_spin_dbl(3, 60, 0, 1, 0.001)
    tk_len.setValue(0.030)
    tagl.addWidget(tk_len, 0, 1)
    both_chk = QtWidgets.QCheckBox("Both sides")
    both_chk.setChecked(True)
    tagl.addWidget(both_chk, 0, 3)
    grid_chk = QtWidgets.QCheckBox("Grid")
    grid_chk.setChecked(True)
    tagl.addWidget(grid_chk, 0, 4)
    tagl.addWidget(QtWidgets.QLabel("Divisions:"), 1, 0)
    div1 = _small_spin_int(0, 100, 40)
    div1.setValue(10)
    tagl.addWidget(div1, 1, 1)
    div2 = _small_spin_int(0, 100, 40)
    div2.setValue(5)
    tagl.addWidget(div2, 1, 2)
    div3 = _small_spin_int(0, 100, 40)
    div3.setValue(0)
    tagl.addWidget(div3, 1, 3)
    tk_clr = _clr_box()
    tk_clr.setCurrentIndex(3)
    tagl.addWidget(tk_clr, 1, 5, QtCore.Qt.AlignRight)  # type: ignore
    layout.addWidget(tag)

    # Labels Group
    lg = QtWidgets.QGroupBox("Labels")
    lgl = QtWidgets.QGridLayout(lg)
    lgl.setContentsMargins(4, 6, 4, 4)
    lgl.setSpacing(2)
    lgl.addWidget(QtWidgets.QLabel("Size:"), 0, 0)
    l_size = _small_spin_dbl(3, 60, 0, 1, 0.001)
    l_size.setValue(0.040)
    lgl.addWidget(l_size, 0, 1)
    lgl.addWidget(QtWidgets.QLabel("Offset:"), 0, 2)
    l_off = _small_spin_dbl(3, 60, -1, 1, 0.001)
    l_off.setValue(0.005)
    lgl.addWidget(l_off, 0, 3)
    l_clr = _clr_box()
    l_clr.setCurrentIndex(3)
    lgl.addWidget(l_clr, 0, 5, QtCore.Qt.AlignRight)  # type: ignore
    layout.addWidget(lg)

    # Font Group
    fg = QtWidgets.QGroupBox("Font")
    fgl = QtWidgets.QHBoxLayout(fg)
    fgl.setContentsMargins(4, 6, 4, 4)
    fgl.setSpacing(4)
    font_fam = QtWidgets.QComboBox()
    font_fam.addItems(["Helvetica", "Times", "Courier", "Arial"])
    font_weight = QtWidgets.QComboBox()
    font_weight.addItems(["normal", "bold", "italic"])
    font_weight.setCurrentText("bold")
    fgl.addWidget(font_fam)
    fgl.addWidget(font_weight)
    fgl.addStretch(1)
    center_chk = QtWidgets.QCheckBox("Center")
    center_chk.setChecked(True)
    fgl.addWidget(center_chk)
    layout.addWidget(fg)
    layout.addStretch(1)

    ctrls = {
        "title": title_edit,
        "t_size": t_size,
        "t_off": t_off,
        "t_clr": t_clr,
        "tk_len": tk_len,
        "both": both_chk,
        "grid": grid_chk,
        "divs": [div1, div2, div3],
        "tk_clr": tk_clr,
        "l_size": l_size,
        "l_off": l_off,
        "l_clr": l_clr,
        "font_fam": font_fam,
        "font_weight": font_weight,
        "center": center_chk,
    }

    # Store on panel
    if axis == "x":
        panel.xaxis_tab_widget = tab
        panel.xaxis_ctrls = ctrls
    else:
        panel.yaxis_tab_widget = tab
        panel.yaxis_ctrls = ctrls

    return tab, ctrls


def _connect_axis_signals(panel: GraphPanel) -> None:
    """Wire up axis-style application logic for both X and Y axis tabs.

    Must be called after both ``_build_axis_tab`` calls.

    Sets ``panel._update_xaxis`` and ``panel._update_yaxis``.
    """

    def apply_axis_style(axis_pos):
        # axis_pos: 'bottom' or 'left'
        ctrls = panel.xaxis_ctrls if axis_pos == "bottom" else panel.yaxis_ctrls
        axis = panel.target_plot.getAxis(axis_pos)
        opposite_pos = "top" if axis_pos == "bottom" else "right"

        # 1. Title & Labels Style
        t_clr = COLORS_HEX_LIST[ctrls["t_clr"].currentIndex()]
        l_clr = COLORS_HEX_LIST[ctrls["l_clr"].currentIndex()]
        family = ctrls["font_fam"].currentText()
        weight = ctrls["font_weight"].currentText()

        # Title Label Style (Apply via css-like args)
        t_px = (
            f"{int(ctrls['t_size'].value() * 300)}pt"  # Scale normalized to points
        )
        label_style = {
            "color": t_clr,
            "font-size": t_px,
            "font-family": family,
            "font-weight": weight,
        }
        axis.setLabel(text=ctrls["title"].text(), **label_style)

        # Tick Labels Style
        f"{int(ctrls['l_size'].value() * 300)}pt"
        axis.setStyle(tickTextOffset=int(ctrls["l_off"].value() * 1000))
        axis.setTextPen(l_clr)
        # Apply font to tick labels via dummy QFont (pyqtgraph axis uses it)
        font = QtGui.QFont(family)
        if weight == "bold":
            font.setBold(True)
        elif weight == "italic":
            font.setItalic(True)
        font.setPointSizeF(ctrls["l_size"].value() * 300)
        axis.setTickFont(font)

        # 2. Ticks & Axis Line
        tk_clr = COLORS_HEX_LIST[ctrls["tk_clr"].currentIndex()]
        axis.setPen(tk_clr)
        axis.setStyle(tickLength=int(ctrls["tk_len"].value() * 100))

        # Grid
        grid_key = "x" if axis_pos == "bottom" else "y"
        panel.target_plot.showGrid(**{grid_key: ctrls["grid"].isChecked()})

        # Both sides
        show_opp = ctrls["both"].isChecked()
        panel.target_plot.showAxis(opposite_pos, show_opp)
        if show_opp:
            opp_axis = panel.target_plot.getAxis(opposite_pos)
            opp_axis.setPen(tk_clr)
            opp_axis.setTicks(
                []
            )  # Usually opposite side just has the line or minor ticks in DTT

    def update_xaxis():
        apply_axis_style("bottom")

    def update_yaxis():
        apply_axis_style("left")

    panel._update_xaxis = update_xaxis
    panel._update_yaxis = update_yaxis

    # Connect all signals
    for ctrls, upd in [
        (panel.xaxis_ctrls, update_xaxis),
        (panel.yaxis_ctrls, update_yaxis),
    ]:
        ctrls["title"].textChanged.connect(upd)
        ctrls["t_size"].valueChanged.connect(upd)
        ctrls["t_off"].valueChanged.connect(upd)
        ctrls["t_clr"].currentIndexChanged.connect(upd)
        ctrls["tk_len"].valueChanged.connect(upd)
        ctrls["both"].toggled.connect(upd)
        ctrls["grid"].toggled.connect(upd)
        ctrls["tk_clr"].currentIndexChanged.connect(upd)
        ctrls["l_size"].valueChanged.connect(upd)
        ctrls["l_off"].valueChanged.connect(upd)
        ctrls["l_clr"].currentIndexChanged.connect(upd)
        ctrls["font_fam"].currentIndexChanged.connect(upd)
        ctrls["font_weight"].currentIndexChanged.connect(upd)
        ctrls["center"].toggled.connect(upd)
        for d in ctrls["divs"]:
            d.valueChanged.connect(upd)


# ---------------------------------------------------------------------------
# Legend tab
# ---------------------------------------------------------------------------


def _build_legend_tab(panel: GraphPanel) -> QtWidgets.QWidget:
    """Build the Legend tab.

    Sets ``panel.update_legend``, ``panel.user_labels``.

    Returns the tab widget.
    """
    legend_tab_widget = QtWidgets.QWidget()
    ll = QtWidgets.QVBoxLayout(legend_tab_widget)
    ll.setContentsMargins(4, 4, 4, 4)
    ll.setSpacing(2)

    legend_show_chk = QtWidgets.QCheckBox("Show")
    legend_show_chk.setChecked(True)
    ll.addWidget(legend_show_chk)

    # Placement Group
    pg_group = QtWidgets.QGroupBox("Placement")
    pgl = QtWidgets.QGridLayout(pg_group)
    pgl.setContentsMargins(4, 6, 4, 4)
    pgl.setSpacing(2)
    rb_tl = QtWidgets.QRadioButton("Top left")
    rb_tr = QtWidgets.QRadioButton("Top Right")
    rb_bl = QtWidgets.QRadioButton("Bottom left")
    rb_br = QtWidgets.QRadioButton("Bottom Right")
    rb_tr.setChecked(True)
    pgl.addWidget(rb_tl, 0, 0)
    pgl.addWidget(rb_tr, 0, 1)
    pgl.addWidget(rb_bl, 1, 0)
    pgl.addWidget(rb_br, 1, 1)

    pos_ly = QtWidgets.QHBoxLayout()
    pos_ly.addWidget(QtWidgets.QLabel("X:"))
    l_x = _small_spin_dbl(2, 60, -10, 10, 0.01)
    pos_ly.addWidget(l_x)
    pos_ly.addSpacing(10)
    pos_ly.addWidget(QtWidgets.QLabel("Y:"))
    l_y = _small_spin_dbl(2, 60, -10, 10, 0.01)
    pos_ly.addWidget(l_y)
    pos_ly.addSpacing(10)
    pos_ly.addWidget(QtWidgets.QLabel("Size:"))
    l_size = _small_spin_dbl(1, 60, 0.1, 10, 0.1)
    l_size.setValue(1.0)
    pos_ly.addWidget(l_size)
    pos_ly.addStretch(1)
    pgl.addLayout(pos_ly, 2, 0, 1, 2)
    ll.addWidget(pg_group)

    # Symbol style Group
    ss_group = QtWidgets.QGroupBox("Symbol style")
    ssl = QtWidgets.QHBoxLayout(ss_group)
    ssl.setContentsMargins(4, 6, 4, 4)
    rb_sym_same = QtWidgets.QRadioButton("Same as trace")
    rb_sym_none = QtWidgets.QRadioButton("None")
    rb_sym_same.setChecked(True)
    ssl.addWidget(rb_sym_same)
    ssl.addWidget(rb_sym_none)
    ssl.addStretch(1)
    ll.addWidget(ss_group)

    # Text Group
    txt_group = QtWidgets.QGroupBox("Text")
    txl = QtWidgets.QVBoxLayout(txt_group)
    txl.setContentsMargins(4, 6, 4, 4)
    txl.setSpacing(4)
    tx_modes = QtWidgets.QHBoxLayout()
    rb_txt_auto = QtWidgets.QRadioButton("Auto")
    rb_txt_user = QtWidgets.QRadioButton("User")
    rb_txt_auto.setChecked(True)
    tx_modes.addWidget(rb_txt_auto)
    tx_modes.addWidget(rb_txt_user)
    tx_modes.addStretch(1)
    txl.addLayout(tx_modes)

    trace_sel_tab = QtWidgets.QTabBar()
    trace_sel_tab.setExpanding(False)
    for i in range(8):
        trace_sel_tab.addTab(str(i))
    txl.addWidget(trace_sel_tab)

    user_legend_edit = QtWidgets.QLineEdit()
    txl.addWidget(user_legend_edit)
    ll.addWidget(txt_group)
    ll.addStretch(1)

    # --- Legend Logic ---
    target_legend = panel.target_plot.addLegend()
    target_legend.show()
    # Set opaque white background and black border
    target_legend.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 255)))
    target_legend.setPen(pg.mkPen("k"))
    target_legend.setZValue(1000)  # Force to very front

    # Store user-defined labels
    panel.user_labels = ["" for _ in range(8)]

    def update_legend():
        if not legend_show_chk.isChecked():
            target_legend.hide()
            return
        target_legend.show()

        # Anchor placement
        anchor = (1, 0)  # TR
        if rb_tl.isChecked():
            anchor = (0, 0)
        elif rb_bl.isChecked():
            anchor = (0, 1)
        elif rb_br.isChecked():
            anchor = (1, 1)

        # Correct positioning for pyqtgraph LegendItem
        # anchor(itemPos, parentPos, offset)
        target_legend.anchor(
            itemPos=anchor,
            parentPos=anchor,
            offset=(l_x.value() * 20, l_y.value() * 20),
        )

        # Refresh items based on mode
        target_legend.clear()
        for i, ctrl in enumerate(panel.trace_controls):
            if not ctrl["active"].isChecked():
                continue

            label = ""
            if rb_txt_auto.isChecked():
                label = ctrl["chan_a"].currentText()
            else:
                label = panel.user_labels[i] or f"Trace {i}"

            # Spectrograms don't have a simple curve to show in legend usually
            if panel.graph_combo.currentText() == "Spectrogram":
                target_legend.addItem(pg.PlotDataItem(pen=None), label)
                continue

            item = panel.traces_items[i]["curve"]
            if ctrl["bar_chk"].isChecked():
                item = panel.traces_items[i]["bar"]

            # Symbol style
            if rb_sym_none.isChecked():
                target_legend.addItem(pg.PlotDataItem(pen=None), label)
            else:
                target_legend.addItem(item, label)

        # Apply Size (Font size) and Color (Black) to all labels in legend
        f_size = f"{int(l_size.value() * 10)}pt"
        for label_pair in target_legend.items:
            # LegendItem.items contains (SampleItem, LabelItem)
            sample, label_widget = label_pair
            if isinstance(label_widget, pg.LabelItem):
                label_widget.setText(label_widget.text, size=f_size, color="k")

    panel.update_legend = update_legend

    # Connect signals
    legend_show_chk.toggled.connect(update_legend)
    for rb in [
        rb_tl,
        rb_tr,
        rb_bl,
        rb_br,
        rb_sym_same,
        rb_sym_none,
        rb_txt_auto,
        rb_txt_user,
    ]:
        rb.toggled.connect(update_legend)
    for sb in [l_x, l_y, l_size]:
        sb.valueChanged.connect(update_legend)

    def on_trace_sel_changed(idx):
        user_legend_edit.blockSignals(True)
        user_legend_edit.setText(panel.user_labels[idx])
        user_legend_edit.blockSignals(False)

    def on_user_txt_edited():
        idx = trace_sel_tab.currentIndex()
        panel.user_labels[idx] = user_legend_edit.text()
        if rb_txt_user.isChecked():
            update_legend()

    trace_sel_tab.currentChanged.connect(on_trace_sel_changed)
    user_legend_edit.textChanged.connect(on_user_txt_edited)

    # Initial state
    on_trace_sel_changed(0)

    return legend_tab_widget


# ---------------------------------------------------------------------------
# Param tab
# ---------------------------------------------------------------------------


def _build_param_tab(panel: GraphPanel) -> QtWidgets.QWidget:
    """Build the Param tab.

    Sets ``panel.update_params_display``, ``panel.param_text``,
    ``panel.meta_info``.

    Returns the tab widget.
    """
    param_tab_widget = QtWidgets.QWidget()
    pl = QtWidgets.QVBoxLayout(param_tab_widget)
    pl.setContentsMargins(4, 4, 4, 4)
    pl.setSpacing(2)

    param_show_chk = QtWidgets.QCheckBox("Show")
    pl.addWidget(param_show_chk)

    # Time format Group
    tf_group = QtWidgets.QGroupBox("Time format")
    tfl = QtWidgets.QHBoxLayout(tf_group)
    tfl.setContentsMargins(4, 6, 4, 4)
    rb_tf_utc = QtWidgets.QRadioButton("Date/time UTC")
    rb_tf_gps = QtWidgets.QRadioButton("GPS seconds")
    rb_tf_utc.setChecked(True)
    tfl.addWidget(rb_tf_utc)
    tfl.addWidget(rb_tf_gps)
    tfl.addStretch(1)
    pl.addWidget(tf_group)

    # Variable Group
    var_group = QtWidgets.QGroupBox("Variable")
    varl = QtWidgets.QVBoxLayout(var_group)
    varl.setContentsMargins(4, 6, 4, 4)
    varl.setSpacing(2)
    v_start = QtWidgets.QCheckBox("Start time")
    v_start.setChecked(True)
    v_avgs = QtWidgets.QCheckBox("Number of averages")
    v_avgs.setChecked(True)
    v_third = QtWidgets.QCheckBox("Third parameter")
    v_third.setChecked(True)
    v_stats = QtWidgets.QCheckBox("Statistics")
    v_stats.setChecked(True)
    v_hist = QtWidgets.QCheckBox("Histogram Under/Overflow")
    v_hist.setChecked(True)
    for w in [v_start, v_avgs, v_third, v_stats, v_hist]:
        varl.addWidget(w)
    pl.addWidget(var_group)
    pl.addStretch(1)

    # Metadata Label on plot (fixed in view coordinates)
    # diaggui parameters are usually in a white-ish box at bottom-left
    panel.param_text = pg.TextItem(
        "", color="k", anchor=(0, 1), border="k", fill=(255, 255, 255, 220)
    )
    panel.target_plot.addItem(panel.param_text, ignoreBounds=True)
    panel.param_text.setVisible(False)
    panel.param_text.setZValue(1000)  # Ensure it's in front

    panel.meta_info = {
        "start_time": 0,
        "avgs": 1,
        "bw": 0,
    }  # To be updated by engine/main

    def update_params_display():
        if not param_show_chk.isChecked():
            panel.param_text.setVisible(False)
            return
        panel.param_text.setVisible(True)

        # Build HTML content for multi-line display + Table
        html = "<div style='font-family: Arial; font-size: 8pt; color: black;'>"

        # Measurement-wide params on top lines
        header_lines = []
        if v_start.isChecked():
            t0 = panel.meta_info.get("start_time", 0)
            if rb_tf_utc.isChecked():
                import datetime

                ts_str = datetime.datetime.fromtimestamp(t0, datetime.UTC).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
                header_lines.append(f"<b>Start:</b> {ts_str}")
            else:
                header_lines.append(f"<b>Start:</b> {t0:.3f}")

        if v_avgs.isChecked():
            header_lines.append(f"<b>Avg:</b> {panel.meta_info.get('avgs', 1)}")

        if v_third.isChecked():
            g_type = panel.graph_combo.currentText()
            if any(
                x in g_type for x in ["Series", "Density", "Coherence", "Function"]
            ):
                val = panel.meta_info.get("bw", 0)
                if val:
                    header_lines.append(f"<b>BW:</b> {val:.3g} Hz")

        if header_lines:
            html += " ".join(header_lines) + "<br>"

        # Statistics Table
        if v_stats.isChecked():
            # Filter only active traces with curve data
            rows = []
            for i, ctrl in enumerate(panel.trace_controls):
                if not ctrl["active"].isChecked():
                    continue
                item = panel.traces_items[i]["curve"]
                x, y = item.getData()
                if x is not None and len(x) > 0:
                    ch_name = ctrl["chan_a"].currentText()[:12]
                    mean = np.mean(y)
                    rms = np.sqrt(np.mean(y**2))
                    pk_pk = np.ptp(y)
                    rows.append(
                        f"<tr><td>{ch_name}</td><td align='right'>{mean:.2e}</td><td align='right'>{rms:.2e}</td><td align='right'>{pk_pk:.2e}</td></tr>"
                    )

            if rows:
                html += (
                    "<table border='0' cellspacing='5' style='margin-top: 2px;'>"
                )
                html += "<tr><th align='left'>Ch</th><th>Mean</th><th>RMS</th><th>Pk-Pk</th></tr>"
                html += "".join(rows) + "</table>"

        html += "</div>"
        panel.param_text.setHtml(html)

        # Position at bottom-left of viewport
        view_box = panel.target_plot.getViewBox()
        tr = view_box.viewRange()
        panel.param_text.setPos(tr[0][0], tr[1][0])

    panel.update_params_display = update_params_display

    # Connect signals
    param_show_chk.toggled.connect(update_params_display)
    rb_tf_utc.toggled.connect(update_params_display)
    rb_tf_gps.toggled.connect(update_params_display)
    for chk in [v_start, v_avgs, v_third, v_stats, v_hist]:
        chk.toggled.connect(update_params_display)

    # Also re-position on range change
    panel.target_plot.getViewBox().sigRangeChanged.connect(update_params_display)

    return param_tab_widget


# ---------------------------------------------------------------------------
# Config tab
# ---------------------------------------------------------------------------


def _build_config_tab(panel: GraphPanel) -> QtWidgets.QWidget:
    """Build the Config tab.

    Sets ``panel.cfg_plot_settings``, ``panel.cfg_respect_user``,
    ``panel.cfg_axes_title``, ``panel.cfg_bin``, ``panel.cfg_time_adjust``.

    Returns the tab widget.
    """
    config_tab_widget = QtWidgets.QWidget()
    cfl = QtWidgets.QVBoxLayout(config_tab_widget)
    cfl.setContentsMargins(4, 4, 4, 4)
    cfl.setSpacing(10)

    # Auto configuration Group
    ac_group = QtWidgets.QGroupBox("Auto configuration")
    acl = QtWidgets.QVBoxLayout(ac_group)
    acl.setContentsMargins(10, 15, 10, 10)
    acl.setSpacing(2)

    panel.cfg_plot_settings = QtWidgets.QCheckBox("Plot settings")
    panel.cfg_plot_settings.setChecked(True)
    # Indented checkbox
    panel.cfg_respect_user = QtWidgets.QCheckBox("Respect user selection")
    panel.cfg_respect_user.setChecked(True)
    panel.cfg_respect_user.setContentsMargins(20, 0, 0, 0)

    panel.cfg_axes_title = QtWidgets.QCheckBox("Axes title")
    panel.cfg_axes_title.setChecked(True)
    panel.cfg_bin = QtWidgets.QCheckBox("Bin")
    panel.cfg_bin.setChecked(True)
    panel.cfg_time_adjust = QtWidgets.QCheckBox("Time Adjust")
    panel.cfg_time_adjust.setChecked(True)

    for w in [
        panel.cfg_plot_settings,
        panel.cfg_respect_user,
        panel.cfg_axes_title,
        panel.cfg_bin,
        panel.cfg_time_adjust,
    ]:
        acl.addWidget(w)
    cfl.addWidget(ac_group)

    # Buttons Row
    btn_layout = QtWidgets.QHBoxLayout()
    btn_layout.addStretch(1)
    btn_store = QtWidgets.QPushButton("Store...")
    btn_restore = QtWidgets.QPushButton("Restore...")
    btn_layout.addWidget(btn_store)
    btn_layout.addWidget(btn_restore)
    cfl.addLayout(btn_layout)
    cfl.addStretch(1)

    # Connect indent visibility
    panel.cfg_plot_settings.toggled.connect(panel.cfg_respect_user.setEnabled)

    return config_tab_widget


# ---------------------------------------------------------------------------
# Stack assembly + final wiring
# ---------------------------------------------------------------------------


def _assemble_stack(panel: GraphPanel, tabs: dict[str, QtWidgets.QWidget]) -> None:
    """Add all tab widgets to the stack in the correct order and wire up
    the cross-tab ``update_axis_labels`` logic.

    *tabs* is a dict mapping tab names to their widgets.
    """
    panel.main_stack.addWidget(tabs["traces"])      # 0
    panel.main_stack.addWidget(tabs["range"])        # 1
    panel.main_stack.addWidget(tabs["units"])        # 2
    panel.main_stack.addWidget(tabs["cursor"])       # 3
    panel.main_stack.addWidget(tabs["config"])       # 4
    panel.main_stack.addWidget(tabs["style"])        # 5
    panel.main_stack.addWidget(panel.xaxis_tab_widget)   # 6
    panel.main_stack.addWidget(panel.yaxis_tab_widget)   # 7
    panel.main_stack.addWidget(tabs["legend"])       # 8
    panel.main_stack.addWidget(tabs["param"])        # 9

    panel.tab_row2.setCurrentIndex(0)

    # Cross-tab wiring: update_axis_labels
    def update_axis_labels():
        if not panel.cfg_axes_title.isChecked():
            for i in range(8):
                panel._update_style(i)
            panel.update_range_logic()
            return

        txt = panel.graph_combo.currentText()
        panel._style_title_edit.setText(txt)
        if txt == "Time Series":
            panel.xaxis_ctrls["title"].setText("Time")
            panel.yaxis_ctrls["title"].setText("Amplitude")
            panel.rb_x_lin.setChecked(True)
            panel.rb_y_lin.setChecked(True)
            panel.display_y_combo.setCurrentText("None")
        elif txt == "Spectrogram":
            panel.xaxis_ctrls["title"].setText("Time")
            panel.yaxis_ctrls["title"].setText("Frequency")
            panel.rb_x_lin.setChecked(True)
            panel.rb_y_log.setChecked(True)
            panel.display_y_combo.setCurrentText("Magnitude")
        elif "Coherence" in txt:
            panel.xaxis_ctrls["title"].setText("Frequency")
            panel.yaxis_ctrls["title"].setText(
                "|Coherence|" if "Squared" not in txt else "Coherence^2"
            )
            panel.rb_x_log.setChecked(True)
            panel.rb_y_lin.setChecked(True)
            panel._range_sb_y_from.setValue(0)
            panel._range_sb_y_to.setValue(1)
            panel._range_rb_y_man.setChecked(True)
        else:
            panel.xaxis_ctrls["title"].setText("Frequency")
            panel.yaxis_ctrls["title"].setText(txt)
            panel.rb_x_log.setChecked(True)
            panel.rb_y_log.setChecked(True)
        for i in range(8):
            panel._update_style(i)
        panel.update_range_logic()

    panel.graph_combo.currentIndexChanged.connect(update_axis_labels)
    update_axis_labels()

    # Initial apply (move to end to avoid UnboundLocalError)
    panel._apply_title_style()
    panel._apply_margins()

    # Set background to white and foreground to black
    panel.target_plot.setBackground("w")
    panel.target_plot.getAxis("left").setPen("k")
    panel.target_plot.getAxis("bottom").setPen("k")
    panel.target_plot.getAxis("left").setTextPen("k")
    panel.target_plot.getAxis("bottom").setTextPen("k")

    # Force axes (and grid) to the back
    panel.target_plot.getAxis("left").setZValue(-100)
    panel.target_plot.getAxis("bottom").setZValue(-100)

    # Enable grid by default
    panel._update_xaxis()
    panel._update_yaxis()
