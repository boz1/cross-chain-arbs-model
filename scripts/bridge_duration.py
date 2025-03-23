# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd


# Common Functions
def dex_out_with_fee(x_in, L_in, L_out, fee):
    """
    Computes the output amount on a DEX swap, taking into account the fee.
    Works elementwise if L_in and L_out are numpy arrays.
    """
    effective_in = x_in * (1 - fee)
    return (L_out * effective_in) / (L_in + effective_in)


def round_trip_profit_dex_dex(
    delta_y, dex1_x, dex1_y, dex2_x, dex2_y, dex1_fee, dex2_fee
):
    """
    Computes the immediate DEX-DEX arbitrage profit (no bridging).
    """
    x_out = dex_out_with_fee(delta_y, dex1_y, dex1_x, dex1_fee)
    y_out = dex_out_with_fee(x_out, dex2_x, dex2_y, dex2_fee)
    return y_out - delta_y


def vectorized_round_trip_profit_bridge(
    delta_y, dex1_y, dex1_x, dex1_fee, dex2_x, dex2_y, dex2_fee, bridging_fee, p_sim
):
    """
    Computes the cross-chain bridging arbitrage profit for a vector of simulated DEX2 prices.
    All heavy computations are performed with numpy vectorized operations.
    """
    # Step 1: Swap Y->X on DEX1 (scalar since delta_y is constant)
    x_out = dex_out_with_fee(delta_y, dex1_y, dex1_x, dex1_fee)
    # Step 2: Apply the bridging fee
    x_after_bridge = x_out * (1 - bridging_fee)
    # Step 3: Rebalance DEX2 reserves based on simulated prices and swap X->Y on DEX2 vectorized
    K = dex2_x * dex2_y
    X2_arbed = np.sqrt(K / p_sim)
    Y2_arbed = np.sqrt(K * p_sim)
    y_out = dex_out_with_fee(x_after_bridge, X2_arbed, Y2_arbed, dex2_fee)
    return y_out - delta_y


def round_trip_profit_cex_dex(delta_y, dex1_x, dex1_y, dex1_fee, cex_price, cex_fee):
    """Immediate CEX-DEX arbitrage. Sell on CEX after DEX1 swap."""
    x_out = dex_out_with_fee(delta_y, dex1_y, dex1_x, dex1_fee)
    y_cex = x_out * cex_price * (1 - cex_fee)  # Sell on CEX
    return y_cex - delta_y


# Command-line parameters:
static_bridge_fee = float(sys.argv[1])
max_bridge_duration = int(sys.argv[2])
simulation_count = int(sys.argv[3])
annual_sigma = float(sys.argv[4])
initial_price_factor = float(sys.argv[5])
M = int(sys.argv[6])

# Create results folder if it does not exist
if not os.path.exists("../results"):
    os.makedirs("../results")

# DEX1 parameters
dex1_x, dex1_y = 1e6, 1e6
dex1_fee = 0.003
p_dex1 = dex1_y / dex1_x

# DEX2 parameters
dex2_x = 1e6
dex2_fee = 0.003

# Simulation setup
bridge_durations = np.linspace(0, max_bridge_duration, simulation_count)

# ETH-USD market parameters
mu_annual = 0.0
seconds_per_year = 365 * 24 * 3600
mu = mu_annual / seconds_per_year
sigma = annual_sigma / np.sqrt(seconds_per_year)

# Initial DEX2 price (DEX2 starts more expensive)
p2 = p_dex1 * initial_price_factor
dex2_y = dex2_x * p2

# 1) Calculate optimal DEX input based on the initial difference
denominator = dex2_x + dex1_x * (1 - dex1_fee)
y = (dex1_y * dex2_x) / denominator
y_ = (dex2_y * dex1_x * (1 - dex1_fee)) / denominator
opt_dex_input = ((np.sqrt(y * y_ * (1 - dex1_fee))) - y) / (1 - dex1_fee)
opt_dex_input = max(opt_dex_input, 0)
print(f"Optimal DEX input: {opt_dex_input}")

# 2) Compute the instant DEX-DEX arbitrage profit (without bridging)
max_profit_dex = round_trip_profit_dex_dex(
    opt_dex_input, dex1_x, dex1_y, dex2_x, dex2_y, dex1_fee, dex2_fee
)
print(f"Instant DEX-DEX profit: {max_profit_dex}")

profit_diffs_duration = []

# Loop over each bridge duration
for i, T_sec in enumerate(bridge_durations):
    print(f"Processing bridge duration {i}/{simulation_count}")
    dt = T_sec
    # Simulate final prices after T_sec for M simulations using a vectorized approach
    W_T = np.random.normal(0, np.sqrt(dt), M)
    p2_simulated = (dex2_y / dex2_x) * np.exp((mu - 0.5 * sigma**2) * dt + sigma * W_T)

    # Compute bridging profit for all simulated prices at once
    bridging_profits = vectorized_round_trip_profit_bridge(
        opt_dex_input,
        dex1_y,
        dex1_x,
        dex1_fee,
        dex2_x,
        dex2_y,
        dex2_fee,
        static_bridge_fee,
        p2_simulated,
    )

    bridge_dex_expected_profit = np.mean(bridging_profits)
    profit_diffs_duration.append(max_profit_dex - bridge_dex_expected_profit)


# Plot the results
window_size = 50  # Adjust as needed
profit_diffs_series = pd.Series(profit_diffs_duration)
moving_avg = profit_diffs_series.rolling(
    window=window_size, center=True, min_periods=1
).mean()

# Plot the profit differences and the moving average versus bridge duration
fig, ax = plt.subplots(figsize=(6.4, 4), dpi=300)
ax.plot(
    bridge_durations,
    profit_diffs_duration,
    linestyle="-",
    label="Profit Difference",
)
ax.plot(bridge_durations, moving_avg, linestyle="--", label="Moving Average")
ax.set_xlabel("Bridging Duration (s)", fontsize=12)
ax.set_ylabel("Profit Difference", fontsize=12)
ax.tick_params(axis="both", labelsize=12)

# ax.set_title(
#     f"Profit Difference vs. Bridge Duration"
# )
ax.legend()
ax.grid(True, linestyle="--", linewidth=0.5)

# Limit x axis to 0-max_bridge_duration
ax.set_xlim(0, max_bridge_duration)

plt.tight_layout()
plt.savefig(
    f"../results/profit_diffs_vs_duration_{static_bridge_fee}_{max_bridge_duration}_{simulation_count}_{annual_sigma}_{initial_price_factor}.pdf",
    bbox_inches="tight",
    facecolor="auto",
    edgecolor="auto",
)

# plt.show()

# Save the results to a CSV file
results_df = pd.DataFrame(
    {
        "Bridge Duration": bridge_durations,
        "Profit Difference": profit_diffs_duration,
        "Moving Average": moving_avg,
    }
)
results_df.to_csv(
    f"../results/profit_diffs_vs_duration_{static_bridge_fee}_{max_bridge_duration}_{simulation_count}_{annual_sigma}_{initial_price_factor}.csv",
    index=False,
)
