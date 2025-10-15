# --- Reopened canvas (no functional change) ---
# monte_carlo_gui_pg.py
# Monte Carlo RoR simulator (PyQtGraph) — performance tuned for 4K
# Dark theme version (charcoal background, white text)
# + Presets with persistence (QSettings)

from __future__ import annotations
import sys, time, json
import numpy as np
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# ----- Optional OpenGL backend (safe fallback if not available) -----
try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    _HAS_GL = True
except Exception:
    QOpenGLWidget = None
    _HAS_GL = False

# ---- Theme colors (tweak here) ------------------------------------
DARK_BG      = "#1f2125"   # window bg
PANEL_BG     = "#24262b"   # card bg
ALT_BG       = "#2b2e34"   # controls
BORDER       = "#3a3e46"
TEXT         = "#eaeaea"
SUBTEXT      = "#b8bec9"
ACCENT       = "#7aa2f7"   # for focus/hover outlines
GRID_COLOR   = (255, 255, 255, 40)  # faint grid on dark

# ================================================================
# Monte Carlo core (unchanged logic; dollar costs per trade)
# ================================================================
def expectancy(win_rate: float, R: float) -> float:
    return win_rate * R - (1.0 - win_rate) * 1.0


def simulate_paths_numpy(
    starting_capital: float,
    risk_perc: float,
    win_rate: float,
    reward_risk: float,
    trades: int,
    sims: int,
    ruin_tr_pct: float,
    seed: int,
    commission_per_trade: float = 15.0,
    slippage_per_trade: float = 15.0,
    rr_var_abs: float = 0.0,
    prop_mode: bool = False,
    use_float32: bool = True,
):
    dtype = np.float32 if use_float32 else np.float64
    rng = np.random.default_rng(seed)

    # optionally randomize RR per trade around base using a uniform ±rr_var_abs range
    if rr_var_abs and rr_var_abs > 0:
        rr_low = max(0.0, reward_risk - rr_var_abs)
        rr_high = reward_risk + rr_var_abs
        rr_rand = rng.uniform(rr_low, rr_high, size=(sims, trades)).astype(dtype)
        win_mult_matrix = 1.0 + risk_perc * rr_rand
        win_mult = None
    else:
        win_mult = np.array(1.0 + risk_perc * reward_risk, dtype=dtype)
        win_mult_matrix = None
    loss_mult = np.array(1.0 - risk_perc, dtype=dtype)

    wins = rng.random((sims, trades)) < win_rate
    if win_mult_matrix is not None:
        mult = np.where(wins, win_mult_matrix, loss_mult)
        mult = mult.astype(dtype, copy=False)
    else:
        mult = np.where(wins, win_mult, loss_mult).astype(dtype, copy=False)

    cost = np.array(commission_per_trade + slippage_per_trade, dtype=dtype)

    equity_paths = np.empty((sims, trades + 1), dtype=dtype)
    equity_paths[:, 0] = starting_capital

    for t in range(trades):
        eq_after_mult = equity_paths[:, t] * mult[:, t]
        equity_paths[:, t + 1] = eq_after_mult - cost

    running_max = np.maximum.accumulate(equity_paths, axis=1)
    drawdown = 1.0 - equity_paths / np.where(running_max == 0, np.nan, running_max)

    if prop_mode:
        # Trailing floor: trails equity peak by max-loss amount until capped at start
        max_loss = starting_capital * (ruin_tr_pct / 100.0)
        floor = np.minimum(starting_capital, running_max - max_loss)
        ruined_mask = (equity_paths <= floor).any(axis=1)
    else:
        if ruin_tr_pct >= 100.0:
            ruined_mask = (equity_paths <= np.array(1e-12, dtype=dtype)).any(axis=1)
        else:
            thr = ruin_tr_pct / 100.0
            ruined_mask = (drawdown >= thr).any(axis=1)

    ending_equity = equity_paths[:, -1]
    dd_max = np.nanmax(drawdown, axis=1)

    stats = {
        "Expectancy_R_per_trade": float(expectancy(win_rate, reward_risk)),
        "Median_Ending_Equity": float(np.median(ending_equity)),
        "Prob_Ruin": float(np.mean(ruined_mask)),
        "Median_Max_Drawdown": float(np.median(dd_max)),
    }
    return equity_paths, stats


def ruin_curve_per_trade_numpy(equity_paths: np.ndarray, ruin_tr_pct: float, *, start_cap: float | None = None, prop_mode: bool = False) -> np.ndarray:
    running_max = np.maximum.accumulate(equity_paths, axis=1)
    if prop_mode:
        if start_cap is None:
            raise ValueError("start_cap is required when prop_mode=True")
        max_loss = start_cap * (ruin_tr_pct / 100.0)
        floor = np.minimum(start_cap, running_max - max_loss)
        ruined_by_trade = (equity_paths <= floor)
    else:
        drawdown = 1.0 - equity_paths / np.where(running_max == 0, np.nan, running_max)
        if ruin_tr_pct >= 100.0:
            ruined_by_trade = (equity_paths <= 1e-12)
        else:
            thr = ruin_tr_pct / 100.0
            ruined_by_trade = (drawdown >= thr)
    ruined_cum = np.minimum.accumulate(~ruined_by_trade, axis=1) == False
    return ruined_cum.mean(axis=0)


def percentile_med(paths: np.ndarray) -> np.ndarray:
    return np.percentile(paths, 50, axis=0)


def sample_max_drawdown(path: np.ndarray) -> float:
    running_max = np.maximum.accumulate(path)
    dd = 1.0 - path / np.where(running_max == 0, np.nan, running_max)
    return float(np.nanmax(dd))


class MoneyAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            try:
                out.append(f"${v:,.0f}")
            except Exception:
                out.append(str(v))
        return out


# ================================================================
# UI helpers
# ================================================================
class CheckableComboBox(QtWidgets.QComboBox):
    checked_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._docs_view_idx = 0  # 0 = Simple, 1 = Advanced
        self.setModel(QtGui.QStandardItemModel(self))
        self.view().pressed.connect(self.handle_item_pressed)
        self.setMinimumWidth(220)
        self.setEditable(True)
        self.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.lineEdit().setReadOnly(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def add_check_item(self, text: str, checked: bool = False):
        it = QtGui.QStandardItem(text)
        it.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
        it.setData(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked,
                   QtCore.Qt.ItemDataRole.CheckStateRole)
        self.model().appendRow(it)
        self.update_display()

    def handle_item_pressed(self, index: QtCore.QModelIndex):
        it = self.model().itemFromIndex(index)
        st = it.data(QtCore.Qt.ItemDataRole.CheckStateRole)
        it.setData(QtCore.Qt.CheckState.Unchecked if st == QtCore.Qt.CheckState.Checked
                   else QtCore.Qt.CheckState.Checked, QtCore.Qt.ItemDataRole.CheckStateRole)
        self.update_display()
        self.checked_changed.emit()

    def checked_items(self):
        out = []
        for i in range(self.model().rowCount()):
            it = self.model().item(i)
            if it.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(it.text())
        return out

    def update_display(self):
        mapping = {"Equity Curve": "EC", "Ruin Curve": "RoR", "Sample P&L (bars)": "PnL"}
        codes = [mapping.get(t, t) for t in self.checked_items()]
        self.lineEdit().setText(" + ".join(codes) if codes else "— none —")


# ================================================================
# Main Window
# ================================================================
class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monte Carlo RoR Simulator (PyQtGraph)")
        self.resize(1200, 720)

        # Prefer GPU + cheap paint (huge win at 4K)
        pg.setConfigOptions(
            useOpenGL=True,
            antialias=False,
            foreground=TEXT,             # white axes/labels on dark
        )

        self.defaults = {
            "start":   50_000.0,
            "risk%":   1.0,
            "win%":    60.0,
            "RR":      1.50,
            "RRvar":   0.00,  # ± absolute variance on RR (0 = fixed RR)
            "N":       500,
            "S":       500,
            "ruin%DD": 10.0,
            "seed":    5,
            "comm":    15.0,
            "slip":    15.0,
            "prop":    False,
        }

        # Preset storage (loaded in _load_settings)
        self._presets: dict[str, dict] = {}
        self._last_session: dict | None = None

        self.curve_equity_median = None
        self.curve_equity_sample = None
        self.curve_ror = None
        self.bar_pnl = None

        self._cache = {"xs": None, "q50s": None, "sample_s": None, "rxs": None, "rcs": None, "pnlx": None, "pnlh": None}
        self._last_fps_update = 0.0

        self._build_ui()
        self._load_settings()          # <- restore presets + last session
        self._apply_last_session_or_defaults()

        self._last_sample_trades = None  # (trade_idx, pnl, equity)
        self._last_outputs_rows = None   # [(field, value), ...]
        self._last_all_summaries = None  # per-path summary for CSV

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.simulate)

        self.simulate()
        # initialize panel visibility
        self._swap_panel(self.cmb_panel.currentIndex())

    # ---------------------- QSettings helpers (Presets) -----------------------
    def _settings(self) -> QtCore.QSettings:
        # Change org/app if you want an unshared store
        return QtCore.QSettings("MonteCarloSim", "RoRApp")

    def _load_settings(self):
        s = self._settings()
        try:
            presets_json = s.value("presets", "{}")
            self._presets = json.loads(presets_json) if isinstance(presets_json, str) else {}
        except Exception:
            self._presets = {}

        try:
            last_json = s.value("last_session", "")
            self._last_session = json.loads(last_json) if isinstance(last_json, str) and last_json else None
        except Exception:
            self._last_session = None

        self._refresh_preset_combo()

    def _write_settings(self):
        s = self._settings()
        try:
            s.setValue("presets", json.dumps(self._presets))
            if self._last_session is not None:
                s.setValue("last_session", json.dumps(self._last_session))
        except Exception:
            pass

    def _apply_last_session_or_defaults(self):
        data = self._last_session or self.defaults
        self._apply_inputs_from_dict(data)

    def _collect_inputs_as_dict(self) -> dict:
        """Collect inputs in UI units (risk%, win% as percents). Validate via _parse_inputs."""
        # Use _parse_inputs for validation & normalization; then convert back to UI units.
        start, risk, win, RR, RRvar, N, S, ruin, seed, comm, slip, prop = self._parse_inputs()
        return {
            "start":   float(start),
            "risk%":   float(risk * 100.0),
            "win%":    float(win * 100.0),
            "RR":      float(RR),
            "RRvar":   float(RRvar),
            "N":       int(N),
            "S":       int(S),
            "ruin%DD": float(ruin),
            "seed":    int(seed),
            "comm":    float(comm),
            "slip":    float(slip),
            "prop":    bool(prop),
        }

    def _apply_inputs_from_dict(self, d: dict):
        """Apply a saved dict (UI units: risk%, win% are percents)."""
        g = {**self.defaults, **(d or {})}
        self.e_start.setText(str(g["start"]))
        self.e_risk.setText(str(g["risk%"]))
        self.e_win.setText(str(g["win%"]))
        self.e_RR.setText(str(g["RR"]))
        self.e_RRvar.setText(str(g["RRvar"]))
        self.e_N.setText(str(g["N"]))
        self.e_S.setText(str(g["S"]))
        self.e_ruin.setText(str(g["ruin%DD"]))
        self.e_seed.setText(str(g["seed"]))
        self.e_comm.setText(str(g["comm"]))
        self.e_slip.setText(str(g["slip"]))
        self.chk_prop.setChecked(bool(g["prop"]))

    def _save_last_session(self):
        try:
            self._last_session = self._collect_inputs_as_dict()
            self._write_settings()
        except Exception:
            pass

    def _refresh_preset_combo(self):
        self.cmb_presets.blockSignals(True)
        self.cmb_presets.clear()
        names = sorted(self._presets.keys(), key=str.casefold)
        self.cmb_presets.addItem("— Select preset —")
        for n in names:
            self.cmb_presets.addItem(n)
        self.cmb_presets.blockSignals(False)

    def _on_select_preset(self, idx: int):
        if idx <= 0:
            return
        name = self.cmb_presets.currentText().strip()
        data = self._presets.get(name)
        if data:
            self._apply_inputs_from_dict(data)

    def _on_save_preset(self):
        try:
            # collect (validates)
            data = self._collect_inputs_as_dict()
        except Exception:
            return

        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            return

        self._presets[name] = data
        self._write_settings()
        self._refresh_preset_combo()

        # select it
        ix = self.cmb_presets.findText(name, QtCore.Qt.MatchFlag.MatchFixedString)
        if ix >= 0:
            self.cmb_presets.setCurrentIndex(ix)

    def _on_delete_preset(self):
        idx = self.cmb_presets.currentIndex()
        if idx <= 0:
            return
        name = self.cmb_presets.currentText().strip()
        if not name:
            return
        confirm = QtWidgets.QMessageBox.question(
            self, "Delete Preset", f"Delete preset '{name}'?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._presets.pop(name, None)
        self._write_settings()
        self._refresh_preset_combo()
        self.cmb_presets.setCurrentIndex(0)

    # --- Hotkeys ---------------------------------------------------------------
    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        """Global hotkeys:
        S = run simulation
        1 / 2 / 3 = switch panel (Inputs / Outputs / Sample Trades)
        D = open documentation
        """
        try:
            mods = e.modifiers()
            if mods == QtCore.Qt.KeyboardModifier.NoModifier:
                k = e.key()

                # S => simulate
                if k == QtCore.Qt.Key.Key_S:
                    self.simulate()
                    e.accept()
                    return

                # 1/2/3 => switch panel via the combo (safe if fewer than 3 items)
                if k in (QtCore.Qt.Key.Key_1, QtCore.Qt.Key.Key_2, QtCore.Qt.Key.Key_3):
                    if hasattr(self, "cmb_panel") and isinstance(self.cmb_panel, QtWidgets.QComboBox):
                        mapping = {
                            QtCore.Qt.Key.Key_1: 0,  # Inputs
                            QtCore.Qt.Key.Key_2: 1,  # Outputs
                            QtCore.Qt.Key.Key_3: 2,  # Sample Trades (if present)
                        }
                        idx = mapping[k]
                        if 0 <= idx < self.cmb_panel.count():
                            self.cmb_panel.setCurrentIndex(idx)
                            e.accept()
                            return

                # D => documentation
                if k == QtCore.Qt.Key.Key_D and hasattr(self, "_show_docs"):
                    self._show_docs()
                    e.accept()
                    return

        except Exception:
            # if anything odd happens, fall through to default handling
            pass

        # default behavior
        super().keyPressEvent(e)

    def _make_card(self, title: str):
        """Return a QGroupBox styled like a card with a large, bold header and
        an inner VBox layout to which children can be added."""
        card = QtWidgets.QGroupBox("")
        card.setObjectName(f"card_{title.lower().replace(' ', '_')}")
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        lbl = QtWidgets.QLabel(title)
        lbl.setStyleSheet("font-size: 22px; font-weight: 700;")
        v.addWidget(lbl)
        return card, v

    def _apply_app_styles(self):
        # Global stylesheet for a modern dark UI
        self.setStyleSheet(f"""
            QWidget {{
                background: {DARK_BG};
                color: {TEXT};
                font-family: Segoe UI, Inter, Arial;
                font-size: 12.5px;
            }}
            QGroupBox {{
                background: transparent;
                border: 0px;
                border-radius: 10px;
                margin-top: 0px;
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {ALT_BG};
                border: 1px solid {BORDER};
                padding: 4px 6px;
                border-radius: 6px;
                selection-background-color: {ACCENT};
                selection-color: #101215;
            }}
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {ACCENT};
            }}
            QTableWidget {{
                background: {ALT_BG};
                gridline-color: {BORDER};
                selection-background-color: {ACCENT};
                selection-color: #101215;
            }}
            QHeaderView::section {{
                background: {PANEL_BG};
                color: {SUBTEXT};
                border: 1px solid {BORDER};
                padding: 4px 8px;
            }}
            QPushButton {{
                background: {ALT_BG};
                border: 1px solid {BORDER};
                padding: 6px 10px;
                border-radius: 8px;
            }}
            QPushButton:hover {{ border-color: {ACCENT}; }}
            QToolTip {{
                background: {PANEL_BG};
                color: {TEXT};
                border: 1px solid {BORDER};
            }}
            QLabel {{ color: {TEXT}; }}
        """)

    def _build_ui(self):
        self._apply_app_styles()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QGridLayout(central)
        root.setColumnStretch(0, 0)
        root.setColumnStretch(1, 1)
        root.setRowStretch(0, 0)
        root.setRowStretch(1, 1)

        self.left_panel = QtWidgets.QWidget()
        self.left_panel.setObjectName("leftPanel")
        self.left_panel.setMinimumWidth(300)
        self.left_panel.setMaximumWidth(360)

        left = QtWidgets.QVBoxLayout(self.left_panel)
        root.addWidget(self.left_panel, 0, 0, 2, 1)

        root.setContentsMargins(6, 6, 6, 6)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)

        # Panel selector (swap left content)
        self.cmb_panel = QtWidgets.QComboBox()
        self.cmb_panel.addItems(["Inputs", "Outputs", "Trades (Sample)"])
        self.cmb_panel.currentIndexChanged.connect(self._swap_panel)
        left.addWidget(self.cmb_panel)

        # Inputs card (styled same as Outputs)
        self.inputs_group, _vin = self._make_card("Inputs")

        left.addWidget(self.inputs_group)
        self.inputs_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                        QtWidgets.QSizePolicy.Policy.Fixed)

        # --- Preset row -----------------------------------------------------
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(QtWidgets.QLabel("Preset:"))
        self.cmb_presets = QtWidgets.QComboBox()
        self.cmb_presets.currentIndexChanged.connect(self._on_select_preset)
        self.btn_preset_save = QtWidgets.QPushButton("Save As…")
        self.btn_preset_delete = QtWidgets.QPushButton("Delete")
        self.btn_preset_save.clicked.connect(self._on_save_preset)
        self.btn_preset_delete.clicked.connect(self._on_delete_preset)
        preset_row.addWidget(self.cmb_presets, 1)
        preset_row.addWidget(self.btn_preset_save)
        preset_row.addWidget(self.btn_preset_delete)
        _vin.addLayout(preset_row)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        _vin.addLayout(form)

        def make_line(v, w=140):
            le = QtWidgets.QLineEdit(str(v))
            le.setFixedWidth(w)
            return le

        self.e_start = make_line(self.defaults["start"])
        self.e_risk  = make_line(self.defaults["risk%"])
        self.e_win   = make_line(self.defaults["win%"])
        self.e_RR    = make_line(self.defaults["RR"])
        self.e_RRvar = make_line(self.defaults["RRvar"])
        self.e_N     = make_line(self.defaults["N"])
        self.e_S     = make_line(self.defaults["S"])
        self.e_ruin  = make_line(self.defaults["ruin%DD"])
        self.e_seed  = make_line(self.defaults["seed"])
        self.e_comm  = make_line(self.defaults["comm"])
        self.e_slip  = make_line(self.defaults["slip"])
        self.chk_prop = QtWidgets.QCheckBox("Prop Firm Mode")
        self.chk_prop.setChecked(self.defaults["prop"])

        form.addRow("Starting Capital ($)", self.e_start)
        form.addRow("Risk per Trade (%)",   self.e_risk)
        form.addRow("Win Rate (%)",         self.e_win)
        form.addRow("Reward : Risk (RR)",   self.e_RR)
        form.addRow("RR variance (±)",      self.e_RRvar)
        form.addRow("Trades (N)",            self.e_N)
        form.addRow("Simulations (S)",       self.e_S)
        form.addRow("Max Drawdown %",        self.e_ruin)
        form.addRow("Seed",                   self.e_seed)
        form.addRow("Commission ($/trade)",   self.e_comm)
        form.addRow("Slippage ($/trade)",     self.e_slip)
        form.addRow(self.chk_prop)

        # Outputs card
        self.outputs_group, _vout = self._make_card("Simulation Outputs")
        self.table = QtWidgets.QTableWidget(10, 2)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        _vout.addWidget(self.table)
        out_btns = QtWidgets.QHBoxLayout()
        self.btn_dl_outputs = QtWidgets.QPushButton("Download Outputs")
        self.btn_dl_outputs.clicked.connect(self._download_outputs_current)
        self.btn_dl_all = QtWidgets.QPushButton("Download ALL Sims")
        self.btn_dl_all.clicked.connect(self._download_all_summaries)
        out_btns.addWidget(self.btn_dl_outputs)
        out_btns.addWidget(self.btn_dl_all)
        _vout.addLayout(out_btns)
        left.addWidget(self.outputs_group)
        self.outputs_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                         QtWidgets.QSizePolicy.Policy.Fixed)

        left.addSpacing(6)

        # Controls group
        self.controls_group = QtWidgets.QGroupBox()
        _ctrl = QtWidgets.QVBoxLayout(self.controls_group)
        _ctrl.setContentsMargins(8, 8, 8, 8)
        _ctrl.setSpacing(6)

        auto_row = QtWidgets.QHBoxLayout()
        self.chk_auto = QtWidgets.QCheckBox("Auto-run")
        self.chk_auto.stateChanged.connect(self._auto_toggled)
        auto_row.addWidget(self.chk_auto); auto_row.addStretch(1)
        auto_row.addWidget(QtWidgets.QLabel("Interval (ms)"))
        self.e_interval = QtWidgets.QLineEdit("250"); self.e_interval.setFixedWidth(80)
        auto_row.addWidget(self.e_interval)
        _ctrl.addLayout(auto_row)

        self.btn_docs = QtWidgets.QPushButton("Documentation"); _ctrl.addWidget(self.btn_docs)
        self.btn_docs.clicked.connect(self._show_docs)
        self.btn_sim  = QtWidgets.QPushButton("Simulate"); self.btn_sim.clicked.connect(self.simulate)
        _ctrl.addWidget(self.btn_sim)

        # Trades card
        self.trades_group, _vtr = self._make_card("Sample Trades")
        _vtr.setContentsMargins(8, 8, 8, 8)
        _vtr.setSpacing(6)
        self.trades_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                        QtWidgets.QSizePolicy.Policy.Expanding)

        self.table_trades = QtWidgets.QTableWidget(0, 4)
        self.table_trades.setHorizontalHeaderLabels(["Trade #", "PnL ($)", "Equity ($)", "DD Limit ($)"])
        self.table_trades.verticalHeader().setVisible(False)
        self.table_trades.horizontalHeader().setStretchLastSection(True)
        self.table_trades.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_trades.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table_trades.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                        QtWidgets.QSizePolicy.Policy.Expanding)
        _vtr.addWidget(self.table_trades, 1)
        self.btn_dl_trades = QtWidgets.QPushButton("Download CSV")
        self.btn_dl_trades.clicked.connect(self._download_trades)
        _vtr.addWidget(self.btn_dl_trades)
        left.addWidget(self.trades_group, 1)
        self.trades_group.setVisible(False)

        left.addStretch(1)
        left.addWidget(self.controls_group)

        # Topbar (charts + sample selector + FPS)
        topbar = QtWidgets.QHBoxLayout()
        self.cmb_charts = CheckableComboBox()
        self.cmb_charts.add_check_item("Equity Curve", checked=True)
        self.cmb_charts.add_check_item("Ruin Curve",   checked=False)
        self.cmb_charts.add_check_item("Sample P&L (bars)", checked=False)
        self.cmb_charts.checked_changed.connect(self._rebuild_plots)
        topbar.addWidget(QtWidgets.QLabel("Charts:")); topbar.addWidget(self.cmb_charts)

        topbar.addWidget(QtWidgets.QLabel("Sample path #"))
        self.sb_sample_idx = QtWidgets.QSpinBox()
        self.sb_sample_idx.setRange(0, 0)
        self.sb_sample_idx.setFixedWidth(80)
        self.sb_sample_idx.valueChanged.connect(self.simulate)
        topbar.addWidget(self.sb_sample_idx)

        topbar.addStretch(1)
        self.lbl_fps = QtWidgets.QLabel("— ms (— FPS)")
        self.lbl_fps.setStyleSheet(f"color:{SUBTEXT};")
        topbar.addWidget(self.lbl_fps)
        root.addLayout(topbar, 0, 1)

        # Graph area
        self.graphs = pg.GraphicsLayoutWidget()
        if _HAS_GL:
            try:
                self.graphs.setViewport(QOpenGLWidget())
            except Exception:
                pass
        self.graphs.setBackground(DARK_BG)  # dark plot background
        root.addWidget(self.graphs, 1, 1)

        # Pens (unchanged)
        self.pen_blue   = pg.mkPen(QtGui.QColor(33,150,243), width=1)
        self.pen_orange = pg.mkPen(QtGui.QColor(255,152,0),  width=1)

        self._rebuild_plots()

    def _swap_panel(self, idx: int):
        if not hasattr(self, "inputs_group"):
            return
        self.inputs_group.setVisible(idx == 0)
        self.outputs_group.setVisible(idx == 1)
        self.trades_group.setVisible(idx == 2)

    def _rebuild_plots(self):
        self.graphs.clear()
        self.curve_equity_median = None
        self.curve_equity_sample = None
        self.curve_ror = None
        self.bar_pnl = None

        selected = self.cmb_charts.checked_items()
        if not selected:
            return

        show_titles = True
        grid_alpha = 0.22  # faint grid on dark

        for name in selected:
            if name == "Equity Curve":
                p = self.graphs.addPlot(axisItems={'left': MoneyAxis(orientation='left')})
                p.getViewBox().setBackgroundColor(DARK_BG)
                p.showGrid(x=True, y=True, alpha=grid_alpha)
                p.getAxis('bottom').setPen(pg.mkPen(TEXT))
                p.getAxis('left').setPen(pg.mkPen(TEXT))
                p.getAxis('bottom').setTextPen(pg.mkPen(TEXT))
                p.getAxis('left').setTextPen(pg.mkPen(TEXT))
                p.getAxis('left').setGrid(255)   # make sure visible on dark
                if show_titles: p.setTitle("<span style='color:{}'>Equity Curve (Median + Sample)</span>".format(TEXT))
                p.setLabel('bottom', f"<span style='color:{TEXT}'>Trade #</span>")
                p.setLabel('left',   f"<span style='color:{TEXT}'>Equity ($)</span>")
                self.curve_equity_sample = p.plot(pen=self.pen_blue)
                self._apply_curve_perf(self.curve_equity_sample)
                self.curve_equity_median = p.plot(pen=self.pen_orange)
                self._apply_curve_perf(self.curve_equity_median)
                self.graphs.nextRow()

            elif name == "Ruin Curve":
                p = self.graphs.addPlot()
                p.getViewBox().setBackgroundColor(DARK_BG)
                p.showGrid(x=True, y=True, alpha=grid_alpha)
                p.getAxis('bottom').setPen(pg.mkPen(TEXT))
                p.getAxis('left').setPen(pg.mkPen(TEXT))
                p.getAxis('bottom').setTextPen(pg.mkPen(TEXT))
                p.getAxis('left').setTextPen(pg.mkPen(TEXT))
                if show_titles: p.setTitle("<span style='color:{}'>Cumulative Probability of Ruin by Trade</span>".format(TEXT))
                p.setLabel('bottom', f"<span style='color:{TEXT}'>Trade #</span>")
                p.setLabel('left',   f"<span style='color:{TEXT}'>Prob. of Ruin</span>")
                self.curve_ror = p.plot(pen=self.pen_blue)
                self._apply_curve_perf(self.curve_ror)
                self.graphs.nextRow()

            elif name == "Sample P&L (bars)":
                p = self.graphs.addPlot()
                p.getViewBox().setBackgroundColor(DARK_BG)
                p.showGrid(x=True, y=True, alpha=grid_alpha)
                p.getAxis('bottom').setPen(pg.mkPen(TEXT))
                p.getAxis('left').setPen(pg.mkPen(TEXT))
                p.getAxis('bottom').setTextPen(pg.mkPen(TEXT))
                p.getAxis('left').setTextPen(pg.mkPen(TEXT))
                if show_titles: p.setTitle("<span style='color:{}'>Sample Trade P&amp;L</span>".format(TEXT))
                p.setLabel('bottom', f"<span style='color:{TEXT}'>Trade #</span>")
                p.setLabel('left',   f"<span style='color:{TEXT}'>PnL ($)</span>")
                self.bar_pnl = pg.BarGraphItem(x=[], height=[], width=0.8, brush=QtGui.QColor(33,150,243))
                p.addItem(self.bar_pnl)
                self.graphs.nextRow()

        self._update_plots_with_cache()

    def _apply_curve_perf(self, curve):
        if curve is None:
            return
        curve.setDownsampling(auto=True, method='peak')
        curve.setClipToView(True)

    def _autosize_outputs_table(self):
        if not hasattr(self, "table") or self.table is None:
            return
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        total_h = self.table.horizontalHeader().height() + 2 * self.table.frameWidth() + 2
        for r in range(self.table.rowCount()):
            total_h += self.table.rowHeight(r)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setFixedHeight(total_h)

    # ---------------- Documentation dialog (unchanged content) ----------------
    def _show_docs(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Monte Carlo Simulator — Documentation")
        dlg.resize(900, 700)
        dlg.setStyleSheet(self.styleSheet())  # inherit dark palette

        vbox = QtWidgets.QVBoxLayout(dlg)
        topbar = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel("View:")
        cmb = QtWidgets.QComboBox()
        cmb.addItems(["Simple", "Advanced"])
        topbar.addWidget(lbl)
        topbar.addWidget(cmb)
        topbar.addStretch(1)
        vbox.addLayout(topbar)

        browser = QtWidgets.QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet("QTextBrowser { background: %s; font-size: 13px; line-height: 1.38; }" % ALT_BG)
        vbox.addWidget(browser, 1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Close).clicked.connect(dlg.close)
        vbox.addWidget(btns)

        # (Keep your existing HTML content here — shortened for brevity)
        simple_html = """
        <div style='padding:8px'>
        <h2 style='margin:0 0 10px'>Monte Carlo RoR Simulator — Quick Overview</h2>

        <p>This tool simulates many alternate futures by <b>randomly sampling wins and losses</b>
        using your inputs (Win Rate, Risk %, Reward:Risk, costs). Each simulation compounds equity
        trade-by-trade, and we summarize the distribution (expectancy, volatility, risk of ruin, etc.).</p>

        <h3>Per-trade update</h3>
        <pre style='background:#383838;padding:10px;border-radius:6px;white-space:pre'>
    if win:   E_{t+1} = E_t × (1 + r × R_t) − cost
    else:     E_{t+1} = E_t × (1 − r)      − cost
        </pre>
        <ul>
            <li><b>r</b> = risk per trade (fraction of equity)</li>
            <li><b>R</b> = reward:risk multiple on wins (can vary ±RR variance)</li>
            <li><b>cost</b> = (commission + slippage) in dollars per trade</li>
        </ul>

        <h3>Risk of Ruin</h3>
        <ul>
            <li><b>Standard:</b> ruin once drawdown <code>DD = 1 − E/peak</code> reaches the threshold.</li>
            <li><b>Prop Firm:</b> a trailing floor sits <code>start × threshold%</code> below the running peak,
                but it never rises above start. Ruin occurs when <code>E ≤ floor</code>.</li>
        </ul>

        <h3>Charts</h3>
        <ul>
            <li><b>Equity Curve:</b> blue = sample path, orange = median path</li>
            <li><b>Ruin Curve:</b> cumulative probability that paths breached the rule by each trade</li>
            <li><b>Sample P&amp;L:</b> per-trade PnL bars for the selected sample path</li>
        </ul>
        </div>
        """

        advanced_html = """
        <div style='padding:10px'>
        <h2 style='margin:0 0 10px'>Advanced — Math, Logic &amp; Definitions</h2>

        <h3>1) Monte Carlo Engine</h3>
        <p>For <i>S</i> simulations of <i>N</i> trades we draw i.i.d. Bernoulli wins with probability <b>p</b> (Win Rate).
            Let risk fraction be <b>r</b>, fixed dollar costs per trade be <b>c</b>, and base reward:risk be <b>R</b>.
            If RR variance ΔR &gt; 0, wins use <code>R_t ∼ Uniform(max(0,R−ΔR), R+ΔR)</code>.</p>
        <pre style='background:#383838;padding:10px;border-radius:6px;white-space:pre'>
    E_{t+1} = E_t × (1 + r × R_t) · 𝟙{win} + E_t × (1 − r) · 𝟙{loss} − c
        </pre>
        <p>Equity compounds each step and costs are subtracted in dollars after the percent move.</p>

        <h3>2) Inputs → Math</h3>
        <ul>
            <li><b>Starting Capital</b> <code>E_0</code>.</li>
            <li><b>Risk per Trade (r)</b>: fraction of equity staked. Loss size at time t is <code>r×E_t</code>.</li>
            <li><b>Win Rate (p)</b>: probability of a win.</li>
            <li><b>Reward:Risk (R)</b>: win size is <code>r×R×E_t</code> before costs.</li>
            <li><b>RR variance (ΔR)</b>: on wins, <code>R_t ∈ [max(0,R−ΔR),R+ΔR]</code>.</li>
            <li><b>Max Drawdown %</b>: threshold T used by ruin logic (standard or prop).</li>
            <li><b>Commission / Slippage (c)</b>: fixed dollars per trade, subtracted after the percent change.</li>
            <li><b>Seed</b>: RNG seed for reproducible paths.</li>
            <li><b>Simulations S / Trades N</b>: number of paths and steps.</li>
            <li><b>Prop Firm Mode</b>: activates trailing-floor rule below.</li>
        </ul>

        <h3>3) Ruin Logic</h3>
        <p><b>Standard:</b> running peak <code>peak_t = max_{0..t} E_t</code>, drawdown <code>DD_t = 1 − E_t/peak_t</code>.
            Ruin whenever <code>max_t DD_t ≥ T</code>, where T is the threshold (e.g., 10%).</p>
        <p><b>Prop Firm:</b> define a trailing floor <code>floor_t = min(E_0, peak_t − E_0·T)</code>.
            Ruin when <code>E_t ≤ floor_t</code>. The floor increases with the peak but caps at <code>E_0</code>.</p>

        <h3>4) Outputs — how they’re computed</h3>
        <ul>
            <li><b>Expectancy (R) per trade</b>: in R-multiples <code>E[R] = p·R − (1−p)</code> (independent of r and c).</li>
            <li><b>Sample Ending Equity</b>: terminal equity of the displayed sample path.</li>
            <li><b>Median Ending Equity</b>: 50th percentile of terminal equity across all S paths.</li>
            <li><b>Sample / Median Max Drawdown (%)</b>:
                <code>DD_t = 1 − E_t/peak_t</code>; take the maximum over t for each path; median is the 50th percentile.</li>
            <li><b>Arithmetic MPTM</b> (mean per-trade multiplier):
            <div style='margin-top:6px'>
            If ΔR = 0: <code>μ_M = p·(1+rR) + (1−p)·(1−r)</code><br>
            If ΔR &gt; 0: use <code>\\bar{R} = (max(0,R−ΔR) + (R+ΔR))/2</code> in place of R.
            </div>
            </li>
            <li><b>Geo. Median multiplier</b>:
                <code>(MedianEnd / E_0)^{1/N}</code> — the per-trade factor that maps start to median end.</li>
            <li><b>Sigma (% / trade)</b>:
                standard deviation of sample per-trade returns <code>r_s(t) = (E_{t+1}−E_t)/E_t</code>.</li>
            <li><b>Beta (vs median)</b>:
                let benchmark returns be <code>r_b</code> from the median path, then <code>β = Cov(r_s,r_b)/Var(r_b)</code>.</li>
            <li><b>Alpha (% / trade)</b>:
                <code>α = mean(r_s − β r_b)</code>, i.e., average excess return of sample over beta-scaled median.</li>
            <li><b>Prob. of Ruin</b>:
                cumulative fraction of paths that have breached the rule by each trade index.</li>
        </ul>

        <h3>5) Charts — math behind visuals</h3>
        <ul>
            <li><b>Equity Curve:</b> shows the sample path and the component-wise median path
                (50th percentile at every trade index).</li>
            <li><b>Ruin Curve:</b> for each k, compute whether each path is ruined by k; plot the mean across paths.</li>
            <li><b>Sample P&amp;L bars:</b> per-trade P&amp;L <code>E_{t+1}−E_t</code>; aggregated into groups when zoomed out.</li>
        </ul>

        <h3>6) Practical notes</h3>
        <ul>
            <li>Fixed dollar costs (commission + slippage) drag returns more at lower equity.</li>
            <li>Higher ΔR widens the outcome distribution even if mean RR is unchanged.</li>
            <li>Prop-firm floors convert equity peaks into hard loss limits until capped at start.</li>
        </ul>

        <h3>7) Vocabulary</h3>
        <ul>
            <li><b>R-multiple</b>: outcome measured in units of risk (<code>r×E_t</code>).</li>
            <li><b>Drawdown</b>: <code>1 − E/peak</code>.</li>
            <li><b>Median path</b>: per-index 50th-percentile equity across paths.</li>
            <li><b>Volatility (Sigma)</b>: variability of per-trade returns.</li>
            <li><b>Beta / Alpha</b>: sensitivity and excess return vs. the benchmark (median path).</li>
        </ul>
        </div>
        """

        try:
            cmb.setCurrentIndex(self._docs_view_idx)
        except Exception:
            cmb.setCurrentIndex(0)
        browser.setHtml(simple_html if cmb.currentIndex() == 0 else advanced_html)

        def swap(idx: int):
            self._docs_view_idx = idx
            browser.setHtml(simple_html if idx == 0 else advanced_html)

        cmb.currentIndexChanged.connect(swap)
        dlg.exec()

    # ---------------- Simulation plumbing (unchanged) ----------------
    def _auto_toggled(self, state):
        if self.chk_auto.isChecked():
            try:
                ms = max(1, int(float(self.e_interval.text())))
            except Exception:
                ms = 250; self.e_interval.setText(str(ms))
            self.timer.start(ms)
        else:
            self.timer.stop()

    def _parse_inputs(self):
        try:
            start = float(self.e_start.text())
            risk  = float(self.e_risk.text()) / 100.0
            win   = float(self.e_win.text()) / 100.0
            RR    = float(self.e_RR.text())
            RRvar = float(self.e_RRvar.text())
            N     = int(float(self.e_N.text()))
            S     = int(float(self.e_S.text()))
            ruin  = float(self.e_ruin.text())
            seed  = int(float(self.e_seed.text()))
            comm  = float(self.e_comm.text())
            slip  = float(self.e_slip.text())
            prop  = bool(self.chk_prop.isChecked())
            if not (0 <= risk < 1 and 0 <= win <= 1 and N > 0 and S > 0 and comm >= 0 and slip >= 0 and RR >= 0 and RRvar >= 0):
                raise ValueError
            return start, risk, win, RR, RRvar, N, S, ruin, seed, comm, slip, prop
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Invalid input", "Please enter valid numeric values.")
            raise

    def _display_cap(self) -> int:
        w = self.graphs.width() * self.devicePixelRatioF()
        w = max(800.0, float(w))
        return int(w * 0.6)

    def simulate(self):
        t0 = time.perf_counter()
        try:
            start, risk, win, RR, RRvar, N, S, ruin, seed, comm, slip, prop = self._parse_inputs()
        except Exception:
            return

        paths, stats = simulate_paths_numpy(
            starting_capital=start, risk_perc=risk, win_rate=win, reward_risk=RR,
            trades=N, sims=S, ruin_tr_pct=ruin, seed=seed,
            commission_per_trade=comm, slippage_per_trade=slip, rr_var_abs=RRvar, prop_mode=prop, use_float32=True
        )

        x = np.arange(paths.shape[1])
        q50 = percentile_med(paths)
        S_paths = int(paths.shape[0])
        self.sb_sample_idx.blockSignals(True)
        self.sb_sample_idx.setRange(0, max(0, S_paths - 1))
        if self.sb_sample_idx.value() > S_paths - 1:
            self.sb_sample_idx.setValue(max(0, S_paths - 1))
        self.sb_sample_idx.blockSignals(False)

        sample_idx = int(self.sb_sample_idx.value()) if S_paths > 0 else 0
        sample_path = paths[sample_idx]
        self._current_sample_idx = sample_idx
        sample_end  = float(sample_path[-1])
        sample_mdd  = sample_max_drawdown(sample_path)
        rcurve = ruin_curve_per_trade_numpy(paths, ruin, start_cap=start, prop_mode=prop)

        pnl = np.diff(sample_path)  # length N

        cap = self._display_cap()
        step = 1 if len(x) <= cap else max(1, len(x) // cap)
        xs = x[::step]
        q50s = q50[::step]
        sample_s = sample_path[::step]
        rxs = np.arange(len(rcurve))[::step]
        rcs = rcurve[::step]

        if step == 1:
            pnlx = np.arange(1, len(pnl)+1)
            pnlh = pnl
        else:
            n_groups = len(pnl) // step
            trimmed = pnl[:n_groups*step]
            pnlh = trimmed.reshape(n_groups, step).sum(axis=1)
            pnlx = (np.arange(n_groups) + 1) * step

        t_compute = time.perf_counter()

        self._cache.update({"xs": xs, "q50s": q50s, "sample_s": sample_s, "rxs": rxs, "rcs": rcs, "pnlx": pnlx, "pnlh": pnlh})
        self._update_plots_with_cache()
        QtWidgets.QApplication.processEvents()

        t_draw = time.perf_counter()

        total_ms   = (t_draw - t0) * 1000.0
        compute_ms = (t_compute - t0) * 1000.0
        draw_ms    = (t_draw - t_compute) * 1000.0
        now = time.perf_counter()
        if now - self._last_fps_update > 0.25:
            fps = 1000.0 / total_ms if total_ms > 0 else float("inf")
            self.lbl_fps.setText(f"{total_ms:,.0f} ms ({fps:,.1f} FPS)  •  compute {compute_ms:,.0f} ms | draw {draw_ms:,.0f} ms")
            self._last_fps_update = now

        exp_R   = stats["Expectancy_R_per_trade"]
        med_end = stats["Median_Ending_Equity"]
        med_dd  = stats["Median_Max_Drawdown"]
        prob_ruin = stats["Prob_Ruin"]

        rr_low = max(0.0, RR - RRvar)
        rr_high = RR + RRvar
        mean_RR = (rr_low + rr_high) / 2.0 if RRvar > 0 else RR
        per_trade_mult_mean = win*(1 + risk*mean_RR) + (1-win)*(1 - risk)
        geo_med_mult = (med_end / start)**(1.0/N) if N > 0 else 1.0

        # --- per-trade risk metrics (sample vs median) ---
        r_s = np.diff(sample_path) / np.maximum(sample_path[:-1], 1e-12)
        sigma_pt = float(np.std(r_s, ddof=1)) if r_s.size > 1 else 0.0
        r_b = np.diff(q50) / np.maximum(q50[:-1], 1e-12)
        if r_b.size > 1 and np.var(r_b, ddof=1) > 0:
            cov = float(np.cov(r_s, r_b, ddof=1)[0, 1])
            var_b = float(np.var(r_b, ddof=1))
            beta_capm = cov / var_b
            alpha_capm = float(np.mean(r_s - beta_capm * r_b))
        else:
            beta_capm = 0.0
            alpha_capm = 0.0

        rows = [
            ("Expectancy (R)",            f"{exp_R:.3f}"),
            ("Sample Ending Equity ($)",  f"{sample_end:,.0f}"),
            ("Median Ending Equity ($)",  f"{med_end:,.0f}"),
            ("Prob. of Ruin (%)",         f"{prob_ruin*100:.2f}"),
            ("Sample Max DD (%)",         f"{sample_mdd*100:.1f}"),
            ("Median Max DD (%)",         f"{med_dd*100:.1f}"),
            ("Arithmetic MPTM",           f"{per_trade_mult_mean:.5f}"),
            ("Geo. Median",               f"{geo_med_mult:.5f}"),
            ("Sigma (% / trade)",         f"{sigma_pt*100:.2f}"),
            ("Beta (vs median)",          f"{beta_capm:.3f}"),
            ("Alpha (% / trade)",         f"{alpha_capm*100:.2f}"),
        ]
        self.table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(v))
        self._last_outputs_rows = rows
        self._autosize_outputs_table()

        running_max_all = np.maximum.accumulate(paths, axis=1)
        if prop:
            max_loss_amt = start * (ruin / 100.0)
            floor_all = np.minimum(start, running_max_all - max_loss_amt)
            ruined_each = (paths <= floor_all).any(axis=1)
        else:
            if ruin >= 100.0:
                ruined_each = (paths <= 1e-12).any(axis=1)
            else:
                thr = ruin / 100.0
                dd_all = 1.0 - paths / np.where(running_max_all == 0, np.nan, running_max_all)
                ruined_each = (dd_all >= thr).any(axis=1)
        dd_all = 1.0 - paths / np.where(running_max_all == 0, np.nan, running_max_all)
        max_dd_each = np.nanmax(dd_all, axis=1)
        var_b = float(np.var(np.diff(q50) / np.maximum(q50[:-1], 1e-12), ddof=1)) if len(q50) > 1 else 0.0
        r_b = np.diff(q50) / np.maximum(q50[:-1], 1e-12)

        all_rows = []
        for i in range(S_paths):
            sp_i = paths[i]
            end_i = float(sp_i[-1])
            mdd_i = float(max_dd_each[i]) * 100.0
            r_s_i = np.diff(sp_i) / np.maximum(sp_i[:-1], 1e-12)
            sigma_i = float(np.std(r_s_i, ddof=1)) * 100.0 if r_s_i.size > 1 else 0.0
            if r_b.size > 1 and var_b > 0:
                cov_i = float(np.cov(r_s_i, r_b, ddof=1)[0, 1])
                beta_i = cov_i / var_b
                alpha_i = float(np.mean(r_s_i - beta_i * r_b)) * 100.0
            else:
                beta_i = 0.0
                alpha_i = 0.0
            ruined_i = int(bool(ruined_each[i]))
            all_rows.append((i, end_i, mdd_i, sigma_i, beta_i, alpha_i, ruined_i))
        self._last_all_summaries = all_rows

        sp = sample_path
        pnl_full = np.diff(sp)
        running_max = np.maximum.accumulate(sp)
        thr = ruin / 100.0
        if prop:
            max_loss_amt = start * thr
            floor_series = np.minimum(start, running_max - max_loss_amt)
        else:
            floor_series = running_max * (1.0 - thr)
        dd_limit = floor_series[1:]

        self.table_trades.setRowCount(len(pnl_full))
        for i in range(len(pnl_full)):
            self.table_trades.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i+1)))
            self.table_trades.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{pnl_full[i]:,.0f}"))
            self.table_trades.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{sp[i+1]:,.0f}"))
            self.table_trades.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{dd_limit[i]:,.0f}"))
        self._last_sample_trades = (np.arange(1, len(pnl_full)+1), pnl_full.copy(), sp[1:].copy(), dd_limit.copy())

        # Persist last session after a successful run
        self._save_last_session()

    def _download_trades(self):
        if not self._last_sample_trades:
            QtWidgets.QMessageBox.information(self, "No data", "Run a simulation first.")
            return
        trades, pnl, equity, mdd = self._last_sample_trades
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Sample Trades", "sample_trades.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("Trade #,PnL ($),Equity ($),DD Limit ($)\n")
                for i in range(len(trades)):
                    f.write(f"{int(trades[i])},{int(round(pnl[i]))},{int(round(equity[i]))},{int(round(mdd[i]))}\n")
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))

    def _download_outputs_current(self):
        if not self._last_outputs_rows:
            QtWidgets.QMessageBox.information(self, "No data", "Run a simulation first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Outputs", "outputs.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("Metric,Value\n")
                for k, v in self._last_outputs_rows:
                    f.write(f"{k},{v}\n")
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))

    def _download_all_summaries(self):
        if not self._last_all_summaries:
            QtWidgets.QMessageBox.information(self, "No data", "Run a simulation first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save All Sim Summaries", "all_sim_summaries.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("Path #,Ending Equity ($),Max DD (%),Sigma (%/trade),Beta,Alpha (%/trade),Ruined (0/1)\n")
                for row in self._last_all_summaries:
                    i, end_i, mdd_i, sigma_i, beta_i, alpha_i, ruined_i = row
                    f.write(f"{i},{int(round(end_i))},{mdd_i:.2f},{sigma_i:.3f},{beta_i:.3f},{alpha_i:.3f},{ruined_i}\n")
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))

    def _update_plots_with_cache(self):
        xs = self._cache.get("xs"); q50s = self._cache.get("q50s"); sample_s = self._cache.get("sample_s")
        rxs = self._cache.get("rxs"); rcs = self._cache.get("rcs"); pnlx = self._cache.get("pnlx"); pnlh = self._cache.get("pnlh")
        if xs is not None and self.curve_equity_sample is not None:
            self.curve_equity_sample.setData(xs, sample_s)
        if xs is not None and self.curve_equity_median is not None:
            self.curve_equity_median.setData(xs, q50s)
        if rxs is not None and self.curve_ror is not None:
            self.curve_ror.setData(rxs, rcs)
        if pnlx is not None and self.bar_pnl is not None:
            p = self.bar_pnl.getViewBox()
            if p is not None:
                p.removeItem(self.bar_pnl)
            self.bar_pnl = pg.BarGraphItem(x=pnlx, height=pnlh, width=0.8, brush=QtGui.QColor(33,150,243))
            if p is not None:
                p.addItem(self.bar_pnl)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = App(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
