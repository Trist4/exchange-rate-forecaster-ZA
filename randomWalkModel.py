from econdatapy import read

from econdatapy import read
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # avoids tkinter issues if you don't need an interactive window
import matplotlib.pyplot as plt

usdzar = read.dataset(
  id = "MARKET_RATES",
  version = "1.0.0",
  series_key = "EXCX135.B.A",
  release = "ZARIBOR & ZARONIA")

# Make sure it's a clean, sorted, 1D series of prices
if isinstance(usdzar, pd.DataFrame):
    price = usdzar.iloc[:, 0]
else:
    price = usdzar

price = price.sort_index().dropna()

# --- 2. Log returns (standard for FX random walk models) ---
log_returns = np.log(price / price.shift(1)).dropna()

mu = log_returns.mean()      # drift term
sigma = log_returns.std()    # volatility (per period, matches your data's frequency)

print(f"Estimated daily drift (mu): {mu:.6f}")
print(f"Estimated daily volatility (sigma): {sigma:.6f}")

# --- 3. Simple random walk forecasts ---
last_price = price.iloc[-1]
horizon = 30  # e.g. 30 periods ahead

# No-drift random walk: best guess for any future period = today's price
rw_naive_forecast = np.repeat(last_price, horizon)

# Random walk WITH drift: price grows by mu each step on average
steps = np.arange(1, horizon + 1)
rw_drift_forecast = last_price * np.exp(mu * steps)

# --- 4. Monte Carlo simulation of future paths ---
n_sims = 1000
np.random.seed(42)

simulated_log_returns = np.random.normal(loc=mu, scale=sigma, size=(n_sims, horizon))
simulated_log_paths = np.cumsum(simulated_log_returns, axis=1)
simulated_price_paths = last_price * np.exp(simulated_log_paths)

# Summary stats across simulations
sim_mean = simulated_price_paths.mean(axis=0)
sim_lower = np.percentile(simulated_price_paths, 5, axis=0)
sim_upper = np.percentile(simulated_price_paths, 95, axis=0)

# --- 5. Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(price.index[-100:], price.values[-100:], label="Historical USDZAR", color="black")

future_index = pd.bdate_range(start=price.index[-1], periods=horizon + 1)[1:]

ax.plot(future_index, rw_drift_forecast, label="RW with drift (point forecast)", linestyle="--")
ax.plot(future_index, sim_mean, label="Monte Carlo mean path", color="tab:blue")
ax.fill_between(future_index, sim_lower, sim_upper, alpha=0.2, color="tab:blue", label="90% simulated range")

ax.axhline(last_price, color="gray", linestyle=":", label="Naive RW forecast")
ax.legend()
ax.set_title("USD/ZAR Random Walk Forecast")
ax.set_ylabel("ZAR per USD")

plt.savefig("usdzar_random_walk.png", dpi=150)
print("Saved plot to usdzar_random_walk.png")