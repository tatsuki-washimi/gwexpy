from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

def _small_spin_int(min_val=-1000000000, max_val=1000000000, width=None):
    w = QtWidgets.QSpinBox()
    w.setRange(min_val, max_val)
    if width: w.setFixedWidth(width)
    return w

def _small_spin_dbl(decimals=1, width=None, min_val=-1e12, max_val=1e12, step=0.1):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(min_val, max_val)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    if width: w.setFixedWidth(width)
    return w

class GraphPanel(QtWidgets.QFrame):
    def __init__(self, plot_idx, target_plot, traces_items, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plot_idx = plot_idx
        self.target_plot = target_plot
        self.traces_items = traces_items # List of {'curve': ..., 'bar': ..., 'img': ...}
        
        self.trace_controls = []
        self.graph_combo = None
        self.display_y_combo = None
        
        self._init_ui()

    def _init_ui(self):
        pv = QtWidgets.QVBoxLayout(self)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)

        # Tabs Setup
        self.tab_row1 = QtWidgets.QTabBar()
        self.tab_row1.addTab("Style"); self.tab_row1.addTab("X-axis"); self.tab_row1.addTab("Y-axis"); self.tab_row1.addTab("Legend"); self.tab_row1.addTab("Param")

        self.tab_row2 = QtWidgets.QTabBar()
        self.tab_row2.addTab("Traces"); self.tab_row2.addTab("Range"); self.tab_row2.addTab("Units"); self.tab_row2.addTab("Cursor"); self.tab_row2.addTab("Config")

        BASE_TAB_STYLE = "QTabBar::tab { height: 25px; padding: 2px; border: 1px solid #C0C0C0; border-bottom: none; min-width: 55px; background: #E0E0E0; font-weight: normal; }"
        ACTIVE_TAB_STYLE = BASE_TAB_STYLE + " QTabBar::tab:selected { font-weight: bold; background: #FFFFFF; border-bottom: 2px solid #FFFFFF; }"

        self.tab_row1.setStyleSheet(BASE_TAB_STYLE)
        self.tab_row2.setStyleSheet(ACTIVE_TAB_STYLE)

        pv.addWidget(self.tab_row1)
        pv.addWidget(self.tab_row2)

        self.main_stack = QtWidgets.QStackedWidget()
        pv.addWidget(self.main_stack)

        # Tab Switching Logic
        def row1_changed(idx):
            if idx == -1: return
            self.tab_row1.setStyleSheet(ACTIVE_TAB_STYLE)
            self.tab_row2.blockSignals(True)
            self.tab_row2.setStyleSheet(BASE_TAB_STYLE)
            self.tab_row2.setCurrentIndex(-1)
            self.tab_row2.blockSignals(False)
            self.main_stack.setCurrentIndex(idx + 5)

        def row2_changed(idx):
            if idx == -1: return
            self.tab_row2.setStyleSheet(ACTIVE_TAB_STYLE)
            self.tab_row1.blockSignals(True)
            self.tab_row1.setStyleSheet(BASE_TAB_STYLE)
            self.tab_row1.setCurrentIndex(-1)
            self.tab_row1.blockSignals(False)
            self.main_stack.setCurrentIndex(idx)

        self.tab_row1.currentChanged.connect(row1_changed)
        self.tab_row2.currentChanged.connect(row2_changed)

        # --- Traces Tab ---
        traces_tab_widget = QtWidgets.QWidget()
        traces_vbox = QtWidgets.QVBoxLayout(traces_tab_widget)
        traces_vbox.setContentsMargins(2, 2, 2, 2)

        r_graph = QtWidgets.QHBoxLayout()
        r_graph.addWidget(QtWidgets.QLabel("Graph:"))
        self.graph_combo = QtWidgets.QComboBox()
        self.graph_combo.addItems([
            "Time Series", "Amplitude Spectral Density",
            "Cross Spectral Density", "Coherence", "Squared Coherence",
            "Transfer Function", "Spectrogram"
        ])
        r_graph.addWidget(self.graph_combo)
        traces_vbox.addLayout(r_graph)

        self.trace_tab_ctrl = QtWidgets.QTabWidget()
        self.trace_tab_ctrl.setStyleSheet("""
            QTabBar::tab { height: 25px; width: 32px; margin: 0; padding: 0; background: #E0E0E0; }
            QTabBar::tab:selected { font-weight: bold; background: #FFFFFF; }
        """)
        self.trace_tab_ctrl.setUsesScrollButtons(False)
        traces_vbox.addWidget(self.trace_tab_ctrl)

        colors = [('Red', '#FF0000'), ('Blue', '#0000FF'), ('Green', '#00FF00'), ('Black', '#000000'), ('Magenta', '#FF00FF'), ('Cyan', '#00FFFF'), ('Yellow', '#FFFF00'), ('Orange', '#FFA500')]
        line_styles = [("Solid", QtCore.Qt.SolidLine), ("Dash", QtCore.Qt.DashLine), ("Dot", QtCore.Qt.DotLine)]
        symbols = [("Circle", "o"), ("Square", "s"), ("Triangle", "t")]
        fill_patterns = [("Solid", QtCore.Qt.SolidPattern), ("Dense", QtCore.Qt.Dense3Pattern)]

        def update_style(t_idx):
            ctrl = self.trace_controls[t_idx]
            target_curve = self.traces_items[t_idx]['curve']
            target_bar = self.traces_items[t_idx]['bar']
            target_img = self.traces_items[t_idx]['img']

            g_type = self.graph_combo.currentText()
            is_active = ctrl['active'].isChecked()
            is_spec_gram = g_type == "Spectrogram"

            dual = g_type in ["Cross Spectral Density", "Coherence", "Squared Coherence", "Transfer Function"]
            ctrl['chan_b'].setEnabled(dual)
            ctrl['g_style'].setEnabled(not is_spec_gram)

            if not is_active:
                target_curve.setPen(None); target_curve.setSymbol(None); target_bar.setVisible(False); target_img.setVisible(False)
                return

            if is_spec_gram:
                target_curve.setPen(None); target_curve.setSymbol(None); target_bar.setVisible(False); target_img.setVisible(True)
                if not hasattr(target_img, '_lut_set'):
                    try:
                        lut = pg.colormap.get('viridis').getLookupTable()
                        target_img.setLookupTable(lut); target_img._lut_set = True
                    except (AttributeError, KeyError, ValueError):
                        pass
                return

            if ctrl['line_chk'].isChecked():
                c_hex = colors[ctrl['line_c'].currentIndex()][1]
                ctrl['line_c'].setStyleSheet(f"background-color: {c_hex};")
                pen = pg.mkPen(color=c_hex, width=ctrl['line_w'].value(), style=ctrl['line_s'].itemData(ctrl['line_s'].currentIndex()))
                target_curve.setPen(pen); target_bar.setVisible(False); target_img.setVisible(False)
            elif ctrl['bar_chk'].isChecked():
                c_hex = colors[ctrl['bar_c'].currentIndex()][1]
                ctrl['bar_c'].setStyleSheet(f"background-color: {c_hex};")
                target_curve.setPen(None); target_bar.setVisible(True); target_img.setVisible(False)
                brush = QtGui.QBrush(QtGui.QColor(c_hex), ctrl['bar_s'].itemData(ctrl['bar_s'].currentIndex()))
                target_bar.setOpts(brush=brush, width=ctrl['bar_w'].value())
            else:
                target_curve.setPen(None); target_bar.setVisible(False); target_img.setVisible(False)

            if ctrl['sym_chk'].isChecked():
                c_hex = colors[ctrl['sym_c'].currentIndex()][1]
                ctrl['sym_c'].setStyleSheet(f"background-color: {c_hex};")
                target_curve.setSymbol(ctrl['sym_s'].itemData(ctrl['sym_s'].currentIndex()))
                target_curve.setSymbolBrush(c_hex); target_curve.setSymbolPen(c_hex); target_curve.setSymbolSize(ctrl['sym_w'].value())
            else:
                target_curve.setSymbol(None)

        channel_names = ["HF_sine", "LF_sine", "beating_sine", "white_noise", "sine_plus_noise", "square_wave", "sawtooth_wave", "random_walk"]
        for i in range(8):
            page = QtWidgets.QWidget()
            pl = QtWidgets.QVBoxLayout(page)
            pl.setContentsMargins(4,4,4,4); pl.setSpacing(2)
            active_chk = QtWidgets.QCheckBox("Active"); active_chk.setChecked(i==0)
            pl.addWidget(active_chk)
            gc = QtWidgets.QGroupBox("Channels"); gl = QtWidgets.QGridLayout(gc); gl.setContentsMargins(4,4,4,4)
            gl.addWidget(QtWidgets.QLabel("A:"), 0, 0); ca = QtWidgets.QComboBox(); ca.setEditable(True); ca.addItems(channel_names); gl.addWidget(ca, 0, 1)
            gl.addWidget(QtWidgets.QLabel("B:"), 1, 0); cb = QtWidgets.QComboBox(); cb.setEditable(True); cb.addItems(channel_names); gl.addWidget(cb, 1, 1)
            pl.addWidget(gc)
            gs = QtWidgets.QGroupBox("Style"); gls = QtWidgets.QGridLayout(gs); gls.setContentsMargins(4,4,4,4)
            lchk = QtWidgets.QCheckBox("Line"); lchk.setChecked(True); gls.addWidget(lchk, 0, 0)
            def _clr_box():
                lc = QtWidgets.QComboBox(); lc.setFixedWidth(40)
                [lc.addItem("") or lc.setItemData(j, QtGui.QColor(c[1]), QtCore.Qt.BackgroundRole) for j,c in enumerate(colors)]
                return lc
            lc = _clr_box(); lc.setCurrentIndex(i % len(colors)); gls.addWidget(lc, 0, 1)
            ls = QtWidgets.QComboBox(); [ls.addItem(n, v) for n,v in line_styles]; gls.addWidget(ls, 0, 2)
            lw = _small_spin_int(1,10, 35); gls.addWidget(lw, 0, 3)
            schk = QtWidgets.QCheckBox("Symbol"); gls.addWidget(schk, 1, 0)
            sc = _clr_box(); sc.setCurrentIndex(i % len(colors)); gls.addWidget(sc, 1, 1)
            ss = QtWidgets.QComboBox(); [ss.addItem(n, v) for n,v in symbols]; gls.addWidget(ss, 1, 2)
            sw = _small_spin_dbl(1, 35, 1, 50); sw.setValue(5); gls.addWidget(sw, 1, 3)
            bchk = QtWidgets.QCheckBox("Bar"); gls.addWidget(bchk, 2, 0)
            bc = _clr_box(); bc.setCurrentIndex(i % len(colors)); gls.addWidget(bc, 2, 1)
            bs = QtWidgets.QComboBox(); [bs.addItem(n, v) for n,v in fill_patterns]; gls.addWidget(bs, 2, 2)
            bw = _small_spin_dbl(2, 35, 0.01, 10, 0.01); bw.setValue(0.04); gls.addWidget(bw, 2, 3)
            pl.addWidget(gs)
            self.trace_tab_ctrl.addTab(page, str(i))
            ctrl_set = {'active': active_chk, 'chan_a': ca, 'chan_b': cb, 'g_style': gs, 'line_chk': lchk, 'line_c': lc, 'line_s': ls, 'line_w': lw, 'sym_chk': schk, 'sym_c': sc, 'sym_s': ss, 'sym_w': sw, 'bar_chk': bchk, 'bar_c': bc, 'bar_s': bs, 'bar_w': bw}
            self.trace_controls.append(ctrl_set)

            active_chk.toggled.connect(lambda _, x=i: update_style(x))
            ca.currentIndexChanged.connect(lambda _, x=i: update_style(x))
            cb.currentIndexChanged.connect(lambda _, x=i: update_style(x))
            lchk.toggled.connect(lambda checked, x=i, b=bchk: (b.setChecked(False) if checked else None, update_style(x)))
            bchk.toggled.connect(lambda checked, x=i, l=lchk: (l.setChecked(False) if checked else None, update_style(x)))
            for w in [lc, ls, lw, schk, sc, ss, sw, bc, bs, bw]:
                if isinstance(w, QtWidgets.QComboBox): w.currentIndexChanged.connect(lambda _, x=i: update_style(x))
                else: w.valueChanged.connect(lambda _, x=i: update_style(x)) if hasattr(w, 'valueChanged') else w.toggled.connect(lambda _, x=i: update_style(x))

        # --- Range Tab ---
        range_tab_widget = QtWidgets.QWidget()
        range_vbox = QtWidgets.QVBoxLayout(range_tab_widget)
        range_vbox.setContentsMargins(4,4,4,4)

        def update_range_logic():
            y_log = rb_y_log.isChecked(); y_auto = rb_y_auto.isChecked()
            self.target_plot.setLogMode(y=y_log)
            if y_auto: self.target_plot.enableAutoRange(axis='y')
            else: self.target_plot.setYRange(sb_y_from.value(), sb_y_to.value(), padding=0)
            x_log = rb_x_log.isChecked(); x_auto = rb_x_auto.isChecked()
            self.target_plot.setLogMode(x=x_log)
            if x_auto: self.target_plot.enableAutoRange(axis='x', enable=True)
            else:
                self.target_plot.enableAutoRange(axis='x', enable=False)
                xmin, xmax = sb_x_from.value(), sb_x_to.value()
                if xmin < xmax: self.target_plot.setXRange(xmin, xmax, padding=0)
        self.update_range_logic = update_range_logic

        gy = QtWidgets.QGroupBox("Y axis"); gly = QtWidgets.QGridLayout(gy)
        gly.addWidget(QtWidgets.QLabel("Scale:"), 0, 0); rb_y_lin = QtWidgets.QRadioButton("linear"); rb_y_log = QtWidgets.QRadioButton("log"); rb_y_lin.setChecked(True)
        hly1 = QtWidgets.QHBoxLayout(); hly1.addWidget(rb_y_lin); hly1.addWidget(rb_y_log); gly.addLayout(hly1, 0, 1)
        gly.addWidget(QtWidgets.QLabel("Range:"), 1, 0); rb_y_auto = QtWidgets.QRadioButton("automatic"); rb_y_man = QtWidgets.QRadioButton("manual"); rb_y_auto.setChecked(True)
        hly2 = QtWidgets.QHBoxLayout(); hly2.addWidget(rb_y_auto); hly2.addWidget(rb_y_man); gly.addLayout(hly2, 1, 1)
        gly.addWidget(QtWidgets.QLabel("From"), 2, 0); sb_y_from = _small_spin_dbl(2, 60); gly.addWidget(sb_y_from, 2, 1)
        gly.addWidget(QtWidgets.QLabel("To"), 2, 2); sb_y_to = _small_spin_dbl(2, 60, max_val=1e12); sb_y_to.setValue(1.1); gly.addWidget(sb_y_to, 2, 3)
        range_vbox.addWidget(gy)

        gx = QtWidgets.QGroupBox("X axis"); glx = QtWidgets.QGridLayout(gx)
        glx.addWidget(QtWidgets.QLabel("Scale:"), 0, 0); rb_x_lin = QtWidgets.QRadioButton("linear"); rb_x_log = QtWidgets.QRadioButton("log"); rb_x_lin.setChecked(True)
        hlx1 = QtWidgets.QHBoxLayout(); hlx1.addWidget(rb_x_lin); hlx1.addWidget(rb_x_log); glx.addLayout(hlx1, 0, 1)
        glx.addWidget(QtWidgets.QLabel("Range:"), 1, 0); rb_x_auto = QtWidgets.QRadioButton("automatic"); rb_x_man = QtWidgets.QRadioButton("manual"); rb_x_auto.setChecked(True)
        hlx2 = QtWidgets.QHBoxLayout(); hlx2.addWidget(rb_x_auto); hlx2.addWidget(rb_x_man); glx.addLayout(hlx2, 1, 1)
        glx.addWidget(QtWidgets.QLabel("From"), 2, 0); sb_x_from = _small_spin_dbl(2, 60); glx.addWidget(sb_x_from, 2, 1)
        glx.addWidget(QtWidgets.QLabel("To"), 2, 2); sb_x_to = _small_spin_dbl(2, 60, max_val=1e12); sb_x_to.setValue(10); glx.addWidget(sb_x_to, 2, 3)
        range_vbox.addWidget(gx)

        for w in [rb_y_lin, rb_y_log, rb_y_auto, rb_y_man, sb_y_from, sb_y_to, rb_x_lin, rb_x_log, rb_x_auto, rb_x_man, sb_x_from, sb_x_to]:
            if isinstance(w, QtWidgets.QRadioButton): w.toggled.connect(update_range_logic)
            else: w.valueChanged.connect(update_range_logic)

        def update_axis_labels():
            txt = self.graph_combo.currentText()
            style_title_edit.setText(txt)
            if txt == "Time Series":
                xaxis_title_edit.setText("Time"); yaxis_title_edit.setText("Amplitude")
                rb_x_log.setChecked(False); rb_y_log.setChecked(False)
                self.display_y_combo.setCurrentText("None")
            elif txt == "Spectrogram":
                xaxis_title_edit.setText("Time"); yaxis_title_edit.setText("Frequency")
                rb_x_log.setChecked(False); rb_y_log.setChecked(False)
                self.display_y_combo.setCurrentText("Magnitude")
            elif "Coherence" in txt:
                xaxis_title_edit.setText("Frequency"); yaxis_title_edit.setText("|Coherence|" if "Squared" not in txt else "Coherence^2")
                rb_x_log.setChecked(True); rb_y_log.setChecked(False)
                sb_y_from.setValue(0); sb_y_to.setValue(1); rb_y_man.setChecked(True)
            else:
                xaxis_title_edit.setText("Frequency"); yaxis_title_edit.setText(txt)
                rb_x_log.setChecked(True); rb_y_log.setChecked(True)
            for i in range(8): update_style(i)
            update_range_logic()
        self.graph_combo.currentIndexChanged.connect(update_axis_labels)

        # --- Units Tab ---
        units_tab_widget = QtWidgets.QWidget(); ul = QtWidgets.QVBoxLayout(units_tab_widget); ul.setContentsMargins(4,4,4,4)
        ug = QtWidgets.QGroupBox("Units"); ugl = QtWidgets.QGridLayout(ug)
        ugl.addWidget(QtWidgets.QLabel("X:"), 0, 0); uxa = QtWidgets.QComboBox(); uxa.addItems(["-", "s", "Hz"]); ugl.addWidget(uxa, 0, 1)
        ugl.addWidget(QtWidgets.QLabel("Y:"), 1, 0); uya = QtWidgets.QComboBox(); uya.addItems(["-", "m", "V", "pk/rtHz"]); ugl.addWidget(uya, 1, 1)
        ul.addWidget(ug)
        dg = QtWidgets.QGroupBox("Display"); dgl = QtWidgets.QGridLayout(dg)
        dgl.addWidget(QtWidgets.QLabel("Y:"), 1, 0); self.display_y_combo = QtWidgets.QComboBox(); self.display_y_combo.addItems(["None", "Magnitude", "Phase", "dB"]); dgl.addWidget(self.display_y_combo, 1, 1)
        ul.addWidget(dg)

        # --- Cursor Tab ---
        cursor_tab_widget = QtWidgets.QWidget(); cl = QtWidgets.QVBoxLayout(cursor_tab_widget); cl.setContentsMargins(4,4,4,4)
        cl.addWidget(QtWidgets.QLabel("Trace:"))
        ctabs = QtWidgets.QTabBar(); ctabs.setExpanding(False); [ctabs.addTab(str(i)) for i in range(8)]; cl.addWidget(ctabs)
        r_curs = QtWidgets.QHBoxLayout()
        ag = QtWidgets.QGroupBox("Active"); agl = QtWidgets.QVBoxLayout(ag); cur_act1 = QtWidgets.QCheckBox("1"); cur_act2 = QtWidgets.QCheckBox("2"); agl.addWidget(cur_act1); agl.addWidget(cur_act2); r_curs.addWidget(ag)
        stg = QtWidgets.QGroupBox("Style"); stgl = QtWidgets.QGridLayout(stg); rb_style_none = QtWidgets.QRadioButton("None"); rb_style_vert = QtWidgets.QRadioButton("Vert."); rb_style_cross = QtWidgets.QRadioButton("Cross"); rb_style_horiz = QtWidgets.QRadioButton("Horiz."); rb_style_none.setChecked(True); stgl.addWidget(rb_style_none, 0,0); stgl.addWidget(rb_style_vert, 0,1); stgl.addWidget(rb_style_cross, 1,0); stgl.addWidget(rb_style_horiz, 1,1); r_curs.addWidget(stg)
        tyg = QtWidgets.QGroupBox("Type"); tygl = QtWidgets.QVBoxLayout(tyg); rb_type_abs = QtWidgets.QRadioButton("Abs."); rb_type_delta = QtWidgets.QRadioButton("Delta"); rb_type_abs.setChecked(True); tygl.addWidget(rb_type_abs); tygl.addWidget(rb_type_delta); r_curs.addWidget(tyg); cl.addLayout(r_curs)
        vg = QtWidgets.QGroupBox("Values"); vgl = QtWidgets.QGridLayout(vg); txt_x1 = QtWidgets.QLineEdit(); txt_y1 = QtWidgets.QLineEdit(); txt_x2 = QtWidgets.QLineEdit(); txt_y2 = QtWidgets.QLineEdit(); vgl.addWidget(QtWidgets.QLabel("X1:"),0,0); vgl.addWidget(txt_x1,0,1); vgl.addWidget(QtWidgets.QLabel("Y1:"),0,2); vgl.addWidget(txt_y1,0,3); vgl.addWidget(QtWidgets.QLabel("X2:"),1,0); vgl.addWidget(txt_x2,1,1); vgl.addWidget(QtWidgets.QLabel("Y2:"),1,2); vgl.addWidget(txt_y2,1,3); cl.addWidget(vg)

        cursors = {'c1_v': pg.InfiniteLine(90, movable=True, pen='g'), 'c1_h': pg.InfiniteLine(0, movable=True, pen='g'), 'c2_v': pg.InfiniteLine(90, movable=True, pen='y'), 'c2_h': pg.InfiniteLine(0, movable=True, pen='y')}
        for line in cursors.values():
            self.target_plot.addItem(line); line.setVisible(False)
            line.sigPositionChanged.connect(lambda: update_cursor_values())

        def update_cursor_values():
            if cursors['c1_v'].isVisible(): txt_x1.setText(f"{cursors['c1_v'].value():.4g}")
            if cursors['c1_h'].isVisible(): txt_y1.setText(f"{cursors['c1_h'].value():.4g}")
            if cursors['c2_v'].isVisible(): txt_x2.setText(f"{cursors['c2_v'].value():.4g}")
            if cursors['c2_h'].isVisible(): txt_y2.setText(f"{cursors['c2_h'].value():.4g}")

        def update_cursor_visibility():
            style = "None"
            if rb_style_vert.isChecked(): style = "Vert"
            elif rb_style_cross.isChecked(): style = "Cross"
            elif rb_style_horiz.isChecked(): style = "Horiz"
            act1, act2 = cur_act1.isChecked(), cur_act2.isChecked()
            cursors['c1_v'].setVisible(act1 and style in ["Vert", "Cross"]); cursors['c1_h'].setVisible(act1 and style in ["Cross", "Horiz"])
            cursors['c2_v'].setVisible(act2 and style in ["Vert", "Cross"]); cursors['c2_h'].setVisible(act2 and style in ["Cross", "Horiz"])
            update_cursor_values()
        for w in [cur_act1, cur_act2, rb_style_none, rb_style_vert, rb_style_cross, rb_style_horiz]: w.toggled.connect(update_cursor_visibility)

        # Style tab items for labels
        style_tab_widget = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(style_tab_widget); sl.setContentsMargins(4,4,4,4)
        tg = QtWidgets.QGroupBox("Title"); tgl = QtWidgets.QGridLayout(tg); style_title_edit = QtWidgets.QLineEdit("Time series"); tgl.addWidget(style_title_edit, 0, 0, 1, 4); sl.addWidget(tg)
        style_title_edit.textChanged.connect(lambda: self.target_plot.setTitle(style_title_edit.text() if style_title_edit.text() else None))

        xaxis_tab_widget = QtWidgets.QWidget(); xal = QtWidgets.QVBoxLayout(xaxis_tab_widget); xal.setContentsMargins(4,4,4,4)
        xtg = QtWidgets.QGroupBox("Title"); xtgl = QtWidgets.QGridLayout(xtg); xaxis_title_edit = QtWidgets.QLineEdit("Time"); xtgl.addWidget(xaxis_title_edit, 0, 0, 1, 4); xal.addWidget(xtg)
        x_grid_chk = QtWidgets.QCheckBox("Grid"); xal.addWidget(x_grid_chk)
        def update_xaxis(): self.target_plot.setLabel('bottom', xaxis_title_edit.text()); self.target_plot.showGrid(x=x_grid_chk.isChecked())
        xaxis_title_edit.textChanged.connect(update_xaxis); x_grid_chk.toggled.connect(update_xaxis)

        yaxis_tab_widget = QtWidgets.QWidget(); yal = QtWidgets.QVBoxLayout(yaxis_tab_widget); yal.setContentsMargins(4,4,4,4)
        ytg = QtWidgets.QGroupBox("Title"); ytgl = QtWidgets.QGridLayout(ytg); yaxis_title_edit = QtWidgets.QLineEdit("Signal"); ytgl.addWidget(yaxis_title_edit, 0, 0, 1, 4); yal.addWidget(ytg)
        y_grid_chk = QtWidgets.QCheckBox("Grid"); yal.addWidget(y_grid_chk)
        def update_yaxis(): self.target_plot.setLabel('left', yaxis_title_edit.text()); self.target_plot.showGrid(y=y_grid_chk.isChecked())
        yaxis_title_edit.textChanged.connect(update_yaxis); y_grid_chk.toggled.connect(update_yaxis)

        legend_tab_widget = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(legend_tab_widget); ll.setContentsMargins(4,4,4,4)
        legend_show_chk = QtWidgets.QCheckBox("Show"); ll.addWidget(legend_show_chk)
        target_legend = self.target_plot.addLegend(); target_legend.hide()
        legend_show_chk.toggled.connect(lambda: target_legend.show() if legend_show_chk.isChecked() else target_legend.hide())

        param_tab_widget = QtWidgets.QWidget(); pl = QtWidgets.QVBoxLayout(param_tab_widget); pl.setContentsMargins(4,4,4,4); pl.addWidget(QtWidgets.QCheckBox("Show"))

        # Config Tab Items
        config_tab_widget = QtWidgets.QWidget(); cfl = QtWidgets.QVBoxLayout(config_tab_widget); cfl.setContentsMargins(4,4,4,4); cfl.addWidget(QtWidgets.QGroupBox("Auto configuration"))

        # Stack Setup
        self.main_stack.addWidget(traces_tab_widget) # 0
        self.main_stack.addWidget(range_tab_widget)  # 1
        self.main_stack.addWidget(units_tab_widget)  # 2
        self.main_stack.addWidget(cursor_tab_widget) # 3
        self.main_stack.addWidget(config_tab_widget) # 4
        self.main_stack.addWidget(style_tab_widget)  # 5
        self.main_stack.addWidget(xaxis_tab_widget)  # 6
        self.main_stack.addWidget(yaxis_tab_widget)  # 7
        self.main_stack.addWidget(legend_tab_widget) # 8
        self.main_stack.addWidget(param_tab_widget)  # 9

        self.tab_row2.setCurrentIndex(0)
        update_axis_labels()

    def to_graph_info(self):
        return {
            'graph_combo': self.graph_combo,
            'traces': self.trace_controls,
            'range_updater': self.update_range_logic,
            'units': {'display_y': self.display_y_combo}
        }
