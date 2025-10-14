# Monte Carlo RoR Simulator — Quick Introduction

Hi - I'm Envy :). This is a lightweight Monte Carlo engine with a PyQt/PyQtGraph UI. It stress-tests a trading model’s return/risk profile under randomness, shows the median path versus a concrete sample path, and estimates Probability of Ruin under fixed or “prop-firm” trailing rules. The goal: fast feedback, honest stats, no fluff.

# Prerequisites

Python 3.9+ (3.11/3.12 recommended)

Packages: pyqt6 (or PyQt5), pyqtgraph, numpy

Optional (for smoother rendering): a GPU that plays nicely with OpenGL

pip install pyqt6 pyqtgraph numpy
# If PyQt6 causes backend issues on your system:
pip install pyqt5 pyqtgraph numpy

To fall back to PyQT5

# How to Run
python montecarlo.py


Use the panel selector (top-left) to switch between Inputs, Outputs, and Sample Trades. The Charts menu lets you display Equity only or Equity + Probability of Ruin. Sample path # selects which randomized path overlays the median.

Hotkeys

S — Run a simulation

1 / 2 / 3 — Switch panel: Inputs / Outputs / Sample Trades

D — Open documentation

Esc — Close dialogs

Presets (Save & Restore)

The app can store named JSON presets (for example, presets/aggressive.json, presets/prop_firm.json) so you can toggle configurations without retyping. Presets include all inputs, including Prop-Firm mode. Load/Save via the Presets menu. Files live next to the script, so they persist across runs and machines.

# Input Definitions

Starting Capital ($) — Initial equity.

Risk per Trade (%) — Fraction of current equity at risk on a loss (this defines 1R).

Win Rate (%) — Probability that a trade wins.

Reward:Risk (RR) — Win size in R-units; a win adds +RR·R, a loss subtracts −1·R.

RR Variance (±) — Optional jitter around RR for wins (e.g., 0.2 → uniform on [RR−0.2, RR+0.2]).

Trades (N) — Trades per simulation path.

Simulations (S) — Number of independent paths.

Max Drawdown % — Fixed ruin floor as a percent below Start, unless…

Prop Firm Mode — Trailing floor: max( Start − MaxDD$, Peak − MaxDD$ ), capped at Start (the floor never rises above the initial balance).

Seed — Reproducible randomness.

Commission / Slippage ($/trade) — Flat per-trade costs deducted each trade.

# Output Definitions

Expectancy (R) — Expected R per trade: E[R] = p·RR − (1−p). Positive is required for growth.

Sample Ending / Median Ending — One realized ending equity versus the 50th percentile across paths.

Probability of Ruin (%) — Share of paths that breach the floor at any time.

Sample / Median Max DD (%) — Worst peak-to-trough drawdown in the sample versus the median across sims.

Arithmetic MPTM — Arithmetic mean per-trade multiplier (>1 implies average per-step growth).

Geometric Median — Geometric median multiplier; robust central tendency of per-trade multipliers.

Sigma (%/trade) — Standard deviation of per-trade returns (normalized per trade).

Beta (vs median) — Sensitivity of the sample path to the median path (co-movement).

Alpha (%/trade) — Average excess return of the sample versus the median after adjusting for beta.

# Math in Brief

Each trade is a Bernoulli trial with win probability p and loss probability 1−p. Let r be the risk fraction (“Risk per Trade %”). Losses multiply equity by (1 − r), wins multiply by (1 + RR·r) (optionally jittered around RR). We also subtract flat costs per trade.

Win: Equity_{t+1} = Equity_t × (1 + RR·r) − Cost

Loss: Equity_{t+1} = Equity_t × (1 − r) − Cost

Ruin occurs if equity crosses a floor: either a fixed threshold Start × (1 − MaxDD%) or the trailing prop floor above. Across S paths we summarize per-trade distributions (medians, dispersion) and path-level stats.

# Reading the Charts

Equity Curve (Median + Sample): Orange = median equity; blue = selected sample path. Large gaps indicate high path dispersion (risk).

Cumulative Probability of Ruin by Trade: Fraction of paths that have breached the floor up to each trade t. A flat line near zero is ideal.

# Practical Tips

Keep Risk% realistic (0.25–1.0% is common). High risk amplifies both compounding and drawdowns.

If Sigma is large, consider lowering Risk% or widening stops with smaller size to stabilize the multiplier.

Alpha slightly positive with Beta near 1 suggests your sample behaves like the median (healthy).

Use RR Variance to model time-based exits or partial profits when wins aren’t always at a fixed multiple.

# ——————————

That’s it. Drop your model into the sim, stress it, and trust the numbers.
— Envy
