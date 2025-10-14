# monte carlo ror simulator — quick intro (by envy)

hi. this is a lightweight monte carlo engine with a pyqt/pyqtgraph ui. it’s built to stress-test a trading model’s return/risk profile under randomness, show median vs. a concrete sample path, and estimate probability of ruin under fixed or “prop firm” trailing rules. the goal: fast feedback, honest stats, no fluff.

prerequisites

python 3.9+ (3.11/3.12 recommended)

packages: pyqt6 (or pyqt5), pyqtgraph, numpy

optional (speed/looks): a gpu that plays nice with opengl

pip install pyqt6 pyqtgraph numpy
# if pyqt6 gives you trouble on your system:
# pip install pyqt5 pyqtgraph numpy

run it
python montecarlo.py


choose a panel from the top-left (inputs / outputs / trades). the charts menu lets you show equity curve only or equity + probability of ruin. “sample path #” selects which randomized path to overlay against the median.

hotkeys

s — run a simulation

1 / 2 / 3 — switch panel: inputs / outputs / sample trades

d — open documentation

esc — close dialogs

presets (save & restore)

the app can store named preset json files (e.g., presets/aggressive.json, presets/prop_firm.json) so you can flip between configurations without retyping. presets hold every input, including prop-firm mode. load/save via the presets menu; files live next to the script so they persist across runs and machines.

what the inputs mean

starting capital ($) — initial equity.

risk per trade (%) — percent of current equity you’re willing to lose on a loss (1r).

win rate (%) — probability a trade wins.

reward:risk (rr) — win size in r-units; a win adds +rr·r, a loss subtracts −1·r.

rr variance (±) — optional jitter around rr per win (e.g., 0.2 → wins uniformly in [rr−0.2, rr+0.2]).

trades (n) — trades per simulation path.

simulations (s) — number of independent paths.

max drawdown % — fixed ruin floor (equity must not drop more than this % below start), unless…

prop firm mode — trailing floor: floor = max( start−max_dd$, peak−max_dd$ ), capped at start (no “raising” above initial balance).

seed — reproducible randomness.

commission/slippage ($/trade) — fixed per-trade costs deducted each trade.

what the outputs mean

expectancy (r) — expected r per trade: E[R] = p·rr − (1−p). positive is required for growth.

sample ending / median ending — one realized ending equity vs. 50th percentile across all paths.

probability of ruin (%) — share of paths that breach the floor at any time.

sample/median max dd (%) — worst peak-to-trough drawdown in the sample vs. median across sims.

arithmetic mptm — average per-trade multiplier (arithmetic mean). a value > 1 implies growth per step on average.

geo. median — geometric median multiplier; robust central tendency of per-trade multipliers.

sigma (%/trade) — std-dev of per-trade returns (in multiplier or r-space, normalized per trade).

beta (vs median) — sensitivity of the sample path to the median path (≈ co-movement).

alpha (%/trade) — average excess return of the sample vs. the median after adjusting for beta.

how the math works (short version)

each trade is a bernoulli trial with win probability p and loss probability 1−p. let r be the risked fraction of equity for that trade (from “risk per trade %”). losses subtract 1r, wins add rr·r (with optional jitter on rr). equity compounds multiplicatively:

on a win: equity_{t+1} = equity_t × (1 + rr·r) − cost

on a loss: equity_{t+1} = equity_t × (1 − r) − cost

where cost = commission + slippage. ruin triggers if equity crosses a floor: either start × (1 − max_dd%) (fixed) or the trailing prop floor described above. across S paths we compute medians and summary stats at each trade index.

reading the charts

equity curve (median + sample) — orange = median equity; blue = selected sample path. big gaps imply high path dispersion (risk).

cumulative probability of ruin by trade — fraction of paths that have ever breached the floor up to trade t. flat near zero is good.

practical tips

keep risk% realistic (0.25–1.0% is common). large risk makes compounding explosive and drawdowns brutal.

if sigma is high, consider reducing risk% or widening stops with smaller size to steady the multiplier.

alpha small but positive with beta near 1 means your path behaves like the median (good for expectations).

use rr variance to model time-based exits or partial profits (i.e., wins not always at fixed rr).

—

that’s it. ship your model into the sim, stress it, and be honest with the results.
— envy
