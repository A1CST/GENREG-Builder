"""GENREG PLAYGROUND -- a PyQt6 control room.

  python3 playground.py

Left: controls (auto-built from the constraint registry + population sliders).
Centre: the live world (board).  Right: the PO cone (top) and the communication
chart (bottom). Toggle laws on/off, tune every parameter, watch behaviour and the
cone respond in real time.
"""
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QGridLayout, QPushButton, QSlider, QCheckBox, QLabel, QGroupBox,
                             QScrollArea, QFrame)

from sim_engine import Sim

pg.setConfigOptions(antialias=True, background="#0d0d12", foreground="#cfd3dc")


def hbar():
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine); f.setStyleSheet("color:#333"); return f


class FloatSlider(QWidget):
    """labelled slider mapping an int track to a float range; calls cb(value)."""
    def __init__(self, name, lo, hi, val, step, cb):
        super().__init__()
        self.lo, self.hi, self.step, self.cb = lo, hi, step, cb
        self.n = max(1, int(round((hi - lo) / step)))
        lay = QGridLayout(self); lay.setContentsMargins(2, 1, 2, 1); lay.setSpacing(2)
        self.lab = QLabel(name); self.lab.setStyleSheet("color:#9aa")
        self.val = QLabel(f"{val:.3g}"); self.val.setStyleSheet("color:#fff; font-weight:bold")
        self.val.setMinimumWidth(48); self.val.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.s = QSlider(Qt.Orientation.Horizontal); self.s.setRange(0, self.n)
        self.s.setValue(int(round((val - lo) / step)))
        self.s.valueChanged.connect(self._changed)
        lay.addWidget(self.lab, 0, 0); lay.addWidget(self.val, 0, 1); lay.addWidget(self.s, 1, 0, 1, 2)

    def _changed(self, iv):
        v = self.lo + iv * self.step
        self.val.setText(f"{v:.3g}"); self.cb(v)


class Playground(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GENREG Playground")
        self.resize(1500, 880)
        self.sim = Sim()
        self.sim.cmap["energy"].enabled = True      # start lively: survival + signalling on
        self.sim.cmap["comm"].enabled = True
        self.playing = False
        self.steps_per_tick = 2

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(6, 6, 6, 6); root.setSpacing(6)
        root.addWidget(self._controls(), 0)
        root.addWidget(self._board(), 3)
        right = QVBoxLayout(); right.setSpacing(6)
        right.addWidget(self._cone(), 3); right.addWidget(self._comm(), 2)
        rw = QWidget(); rw.setLayout(right); root.addWidget(rw, 2)

        self.timer = QTimer(); self.timer.timeout.connect(self.tick); self.timer.setInterval(45)
        self._redraw_cone(); self._redraw_board(); self._update_status()

    # ----------------------------- controls --------------------------------
    def _controls(self):
        panel = QWidget(); panel.setFixedWidth(320)
        v = QVBoxLayout(panel); v.setSpacing(6)
        title = QLabel("GENREG PLAYGROUND"); title.setStyleSheet("color:#2ecc71; font-size:15px; font-weight:bold")
        v.addWidget(title)
        # transport
        row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play"); self.btn_play.clicked.connect(self.toggle_play)
        b_step = QPushButton("Step"); b_step.clicked.connect(lambda: (self.sim.step(), self._refresh()))
        b_reset = QPushButton("Reset"); b_reset.clicked.connect(self.reset)
        for b in (self.btn_play, b_step, b_reset):
            b.setStyleSheet("background:#222; color:#eee; padding:5px; border:1px solid #444")
            row.addWidget(b)
        v.addLayout(row)
        v.addWidget(FloatSlider("speed (steps/frame)", 1, 12, 2, 1, self._set_speed))
        v.addWidget(hbar())
        # constraints (auto from registry)
        cg = QLabel("LAWS OF EXISTENCE  (toggle)"); cg.setStyleSheet("color:#e74c3c; font-weight:bold")
        v.addWidget(cg)
        for c in self.sim.constraints:
            box = QGroupBox(); box.setStyleSheet(f"QGroupBox{{border:1px solid {c.color}; border-radius:4px; margin-top:4px}}")
            bl = QVBoxLayout(box); bl.setContentsMargins(6, 4, 6, 4); bl.setSpacing(2)
            cb = QCheckBox(c.label); cb.setStyleSheet(f"color:{c.color}; font-weight:bold")
            cb.setChecked(c.enabled)
            cb.toggled.connect(lambda on, cc=c: self._toggle(cc, on))
            cb.setToolTip(c.desc)
            bl.addWidget(cb)
            for pr in c.params:
                bl.addWidget(FloatSlider("   " + pr["name"], pr["lo"], pr["hi"], pr["val"], pr["step"],
                                         lambda val, p=pr: p.__setitem__("val", val)))
            v.addWidget(box)
        v.addWidget(hbar())
        # population controls
        pg_ = QLabel("POPULATION / WORLD"); pg_.setStyleSheet("color:#3498db; font-weight:bold")
        v.addWidget(pg_)
        ctl = self.sim.ctrl
        v.addWidget(FloatSlider("population N", 8, 80, ctl["N"], 2, lambda x: self._setctrl("N", int(x), rebuild=True)))
        v.addWidget(FloatSlider("food patches", 6, 90, ctl["NF"], 2, lambda x: self._setctrl("NF", int(x), rebuild=True)))
        v.addWidget(FloatSlider("mutation rate", 0.02, 0.6, ctl["mut"], 0.02, lambda x: self._setctrl("mut", x)))
        v.addWidget(FloatSlider("selection rate (cull frac)", 0.02, 0.4, ctl["cull_frac"], 0.02, lambda x: self._setctrl("cull_frac", x)))
        v.addWidget(FloatSlider("kin spread (offspring near parent)", 1, 30, ctl["kin_spread"], 1, lambda x: self._setctrl("kin_spread", x)))
        v.addWidget(FloatSlider("exploration noise", 0.0, 1.2, ctl["noise"], 0.05, lambda x: self._setctrl("noise", x)))
        v.addStretch(1)
        # status
        self.status = QLabel(""); self.status.setStyleSheet("color:#9aa; font-family:monospace; font-size:11px")
        v.addWidget(self.status)

        scroll = QScrollArea(); scroll.setWidget(panel); scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340); scroll.setFrameShape(QFrame.Shape.NoFrame)
        return scroll

    # ----------------------------- views -----------------------------------
    def _board(self):
        w = pg.PlotWidget(title="THE WORLD")
        w.setAspectLocked(True); w.setXRange(0, self.sim.ctrl["L"]); w.setYRange(0, self.sim.ctrl["L"])
        w.hideAxis("bottom"); w.hideAxis("left")
        # climate: highlight the active growing-season quadrant (drawn behind everything)
        self.season_rect = QtWidgets.QGraphicsRectItem(0, 0, self.sim.ctrl["L"]/2, self.sim.ctrl["L"]/2)
        self.season_rect.setBrush(pg.mkBrush(26, 182, 196, 35)); self.season_rect.setPen(pg.mkPen(26, 182, 196, 110))
        self.season_rect.setZValue(-10); self.season_rect.setVisible(False); w.addItem(self.season_rect)
        self.it_food = pg.ScatterPlotItem(pen=None, symbol="s", brush=pg.mkBrush(46, 204, 113, 120))
        self.it_edge = pg.PlotCurveItem(pen=pg.mkPen(241, 196, 15, 120, width=1), connect="finite")
        self.it_org = pg.ScatterPlotItem(pen=pg.mkPen("#fff", width=0.4))
        w.addItem(self.it_food); w.addItem(self.it_edge); w.addItem(self.it_org)
        self.board = w; return w

    def _cone(self):
        w = pg.PlotWidget(title="PO CONE  (constraint over infinity)")
        w.setXRange(-1.15, 1.15); w.setYRange(-0.08, 1.12)
        w.hideAxis("bottom"); w.hideAxis("left"); w.setMouseEnabled(False, False)
        w.addItem(pg.PlotCurveItem([-1, 1, 0, -1], [1, 1, 0, 1], pen=pg.mkPen("#888", width=1.5)))
        self.cone = w
        self.cone_dyn = []  # items we clear/redraw
        self.txt_po = pg.TextItem("", color="#fff", anchor=(0.5, 0.5)); self.txt_po.setPos(-0.62, 0.34)
        self.txt_po.setFont(QFont("monospace", 16, QFont.Weight.Bold))
        self.txt_sp = pg.TextItem("", color="#2ecc71", anchor=(0.5, 0.5)); self.txt_sp.setPos(-0.62, 0.20)
        w.addItem(self.txt_po); w.addItem(self.txt_sp)
        return w

    def _comm(self):
        w = pg.PlotWidget(title="COMMUNICATION")
        w.setLabel("left", "count"); w.hideAxis("bottom"); w.addLegend(offset=(-1, 1))
        self.it_sig = pg.PlotDataItem(pen=pg.mkPen("#f1c40f", width=2), name="signalling")
        self.it_comm = pg.PlotDataItem(pen=pg.mkPen("#2ecc71", width=2), name="following",
                                       fillLevel=0, brush=pg.mkBrush(46, 204, 113, 60))
        w.addItem(self.it_sig); w.addItem(self.it_comm); self.commw = w
        self.txt_comm = pg.TextItem("", color="#9aa", anchor=(0, 1)); w.addItem(self.txt_comm)
        return w

    # ----------------------------- logic -----------------------------------
    def _toggle(self, c, on):
        c.enabled = on
        if not self.playing:
            self._refresh()
        self._redraw_cone()

    def _set_speed(self, v): self.steps_per_tick = int(v)

    def _setctrl(self, key, val, rebuild=False):
        self.sim.ctrl[key] = val
        if rebuild:
            self.sim.reset()
            self.board.setXRange(0, self.sim.ctrl["L"]); self.board.setYRange(0, self.sim.ctrl["L"])
            self._refresh()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("❚❚ Pause" if self.playing else "▶ Play")
        (self.timer.start() if self.playing else self.timer.stop())

    def reset(self):
        self.sim.reset(); self._refresh()

    def tick(self):
        for _ in range(self.steps_per_tick):
            self.sim.step()
        self._refresh()

    def _refresh(self):
        self._redraw_board(); self._redraw_comm(); self._redraw_cone_dynamic(); self._update_status()

    # ----------------------------- drawing ---------------------------------
    def _redraw_board(self):
        s = self.sim
        # climate: season highlight + food coloured by maturity (seedling brown -> mature green)
        if s.on("climate"):
            half = s.ctrl["L"] / 2; corner = [(0, 0), (half, 0), (half, half), (0, half)][s.active_quadrant]
            self.season_rect.setRect(corner[0], corner[1], half, half); self.season_rect.setVisible(True)
            m = np.clip(s.food_mat, 0, 1)
            fbr = [pg.mkBrush(int(120 + 20*mi), int(70 + 150*mi), int(40), int(50 + 170*mi)) for mi in m]
            self.it_food.setData(s.food_xy[:, 0], s.food_xy[:, 1], size=4 + 11 * m, brush=fbr)
        else:
            self.season_rect.setVisible(False)
            self.it_food.setData(s.food_xy[:, 0], s.food_xy[:, 1], size=6 + 10 * s.food_amt,
                                 brush=pg.mkBrush(46, 204, 113, 120))
        # organisms coloured by energy; signallers larger + yellow ring
        e = np.clip(s.energy / 18.0, 0.05, 1.0) if s.on("energy") else np.full(len(s.geno), 0.6)
        brushes = [pg.mkBrush(int(255), int(80 + 150 * ei), int(20), 230) for ei in e]
        sizes = np.where(s.signalers, 15, 9)
        pens = [pg.mkPen("#f1c40f", width=2) if sg else pg.mkPen("#fff", width=0.4) for sg in s.signalers]
        self.it_org.setData(s.pos[:, 0], s.pos[:, 1], brush=brushes, size=sizes, pen=pens)
        # comm edges as line segments (finite-connected pairs)
        if s.edges:
            n = len(s.pos); xs, ys = [], []
            for a, b in s.edges:
                if a < n and b < n:
                    xs += [s.pos[a, 0], s.pos[b, 0], np.nan]; ys += [s.pos[a, 1], s.pos[b, 1], np.nan]
            self.it_edge.setData(xs, ys)
        else:
            self.it_edge.setData([], [])

    def _redraw_comm(self):
        h = self.sim.comm_hist; sg = self.sim.sig_hist
        x = np.arange(len(h))
        self.it_comm.setData(x, np.array(h, float))
        self.it_sig.setData(x, np.array(sg, float))
        self.txt_comm.setText(f"signalling now: {int(self.sim.signalers.sum())}   "
                              f"following: {len(self.sim.edges)}")
        top = max(5, (max(h) if h else 0), (max(sg) if sg else 0)) * 1.1
        if h:
            self.commw.setYRange(0, top); self.txt_comm.setPos(0, top)

    def _redraw_cone(self):
        for it in self.cone_dyn:
            self.cone.removeItem(it)
        self.cone_dyn = []
        cons = self.sim.constraints; ncon = len(cons)
        for di, c in enumerate(cons, start=1):
            if not c.enabled:
                continue
            y = 1 - di / (ncon + 1)
            ln = pg.PlotCurveItem([-y, y], [y, y], pen=pg.mkPen(c.color, width=3))
            tx = pg.TextItem("+" + c.key.upper(), color=c.color, anchor=(0, 0.5)); tx.setPos(y + 0.03, y)
            tx.setFont(QFont("monospace", 7))
            self.cone.addItem(ln); self.cone.addItem(tx); self.cone_dyn += [ln, tx]
        self._redraw_cone_dynamic()

    def _redraw_cone_dynamic(self):
        s = self.sim; po = s.po(); ncon = len(s.constraints)
        yb = (1 - po / (ncon + 1)) if po else 0.98
        hw = max(0.02, yb * s.spread_frac())
        if not hasattr(self, "xsec"):
            self.xsec = pg.PlotCurveItem(fillLevel=0, pen=pg.mkPen("#2ecc71", width=8))
            self.cone.addItem(self.xsec)
        self.xsec.setData([-hw, hw], [yb, yb])
        self.txt_po.setText(f"PO = {po}")
        self.txt_sp.setText(f"strategies\n{100 * s.spread_frac():.0f}%")

    def _update_status(self):
        s = self.sim
        alive = int((s.energy > 0).sum()) if s.on("energy") else len(s.geno)
        repro = "  (mating)" if s.on("repro") else ""
        season = f"\nseason: quadrant {s.active_quadrant}" if s.on("climate") else ""
        self.status.setText(
            f"step {s.t}\ngeneration {s.generation()}\nPO {s.po()}\n"
            f"population {len(s.geno)}{repro}\nalive {alive}/{len(s.geno)}\n"
            f"strategies {100*s.spread_frac():.0f}%\nsignallers {int(s.signalers.sum())}{season}")


def main():
    app = QApplication(sys.argv)
    win = Playground(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
