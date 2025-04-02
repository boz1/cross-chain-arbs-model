import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd


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
price_sim_count = int(sys.argv[1])
max_price_diff = float(sys.argv[2])
annual_sigma = float(sys.argv[3])
bridging_delay = int(sys.argv[4])
M = int(sys.argv[5])

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

# CEX parameters
cex_fee = 0.001  # 0.1% fee

# Bridging parameters
bridging_fee = 0.005  # 0.5% bridging fee

# GBM parameters
mu_annual = 0.0
seconds_per_year = 365 * 24 * 3600
mu = mu_annual / seconds_per_year
sigma = annual_sigma / np.sqrt(seconds_per_year)

# Define a range of final prices p2, for immediate DEX2 or CEX
p2_values = np.linspace(p_dex1, p_dex1 * max_price_diff, price_sim_count)
price_diff = p2_values - p_dex1

# Arrays for storing the max profits (each point in p2_values)
max_profits_dex_dex = []
max_profits_cex_dex = []
max_profits_bridge_dex_mc = []

###############################################################################
# Outer loop: for each immediate final price p2 of the second DEX
###############################################################################
for p2 in p2_values:
    # Re-compute dex2_y for that p2
    dex2_y = dex2_x * p2

    # 1) Instant DEX-DEX: find best input
    denominator = dex2_x + dex1_x * (1 - dex1_fee)
    y = (dex1_y * dex2_x) / denominator
    y_ = (dex2_y * dex1_x * (1 - dex1_fee)) / denominator
    opt_dex_input = ((np.sqrt(y * y_ * (1 - dex1_fee))) - y) / (1 - dex1_fee)
    opt_dex_input = max(opt_dex_input, 0)
    # 2) Compute the instant DEX-DEX arbitrage profit (without bridging)
    max_profit_dex = round_trip_profit_dex_dex(
        opt_dex_input, dex1_x, dex1_y, dex2_x, dex2_y, dex1_fee, dex2_fee
    )
    max_profits_dex_dex.append(max_profit_dex)

    # 3) Instant CEX-DEX: find best input
    opt_cex_input = (
        np.sqrt(dex1_x * dex1_y * p2 * (1 - dex1_fee) * (1 - cex_fee)) - dex1_y
    ) / (1 - dex1_fee)
    cex_input = max(opt_cex_input, 0)
    # 4) Compute the instant CEX-DEX arbitrage profit
    max_profit_cex = round_trip_profit_cex_dex(
        cex_input, dex1_x, dex1_y, dex1_fee, p2, cex_fee
    )
    max_profits_cex_dex.append(max_profit_cex)

    # 5) Cross-chain bridging arbitrage: Monte Carlo simulation
    dt = bridging_delay
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
        bridging_fee,
        p2_simulated,
    )
    bridge_dex_expected_profit = np.mean(bridging_profits)
    max_profits_bridge_dex_mc.append(bridge_dex_expected_profit)

# Convert results to numpy arrays
max_profits_dex_dex = np.array(max_profits_dex_dex)
max_profits_cex_dex = np.array(max_profits_cex_dex)
max_profits_bridge_dex_mc = np.array(max_profits_bridge_dex_mc)

prof_diff_dex_dex_to_bridge_dex = max_profits_dex_dex - max_profits_bridge_dex_mc

# Save the results to a CSV file
df = pd.DataFrame(
    {
        "p2": p2_values,
        "price_diff": price_diff,
        "max_profits_dex_dex": max_profits_dex_dex,
        "max_profits_cex_dex": max_profits_cex_dex,
        "max_profits_bridge_dex_mc": max_profits_bridge_dex_mc,
        "prof_diff_dex_dex_to_bridge_dex": prof_diff_dex_dex_to_bridge_dex,
        "prof_diff_cex_dex_to_bridge_dex": prof_diff_cex_dex_to_bridge_dex,
        "prof_diff_cex_dex_to_dex_dex": prof_diff_cex_dex_to_dex_dex,
    }
)
df.to_csv(
    f"../results/profit_comparison_{bridging_fee}_{bridging_delay}_{price_sim_count}_{annual_sigma}_{max_price_diff}_{M}.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(6.4, 4), dpi=300)

max_price_diff = price_diff.max()

# Main plots (Primary Y-axis) with distinct line styles
(line1,) = ax.plot(price_diff, max_profits_dex_dex, label="Instant DEX-DEX", lw=2)
(line2,) = ax.plot(
    price_diff, max_profits_bridge_dex_mc, label="Bridge DEX-DEX", lw=2, linestyle="--"
)
(line3,) = ax.plot(price_diff, max_profits_cex_dex, label="CEX-DEX", lw=2)

ax.set_xlabel("Instant Price Discrepancy", fontsize=12)
ax.set_ylabel("Maximum Arbitrage Profit", fontsize=12)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, linestyle="--", linewidth=0.5)

# Make y-axis scientific
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# Limit x axis to 0 - max_price_diff
ax.set_xlim(0, price_diff.max())

# Secondary plot (Secondary Y-axis)
ax2 = ax.twinx()
(line4,) = ax2.plot(
    price_diff,
    prof_diff_dex_dex_to_bridge_dex,
    linestyle="-.",
    label="Instant vs. Bridge",
    color="gray",
    alpha=0.6,
    linewidth=2,
)

ax2.set_ylabel("Profit Difference", color="gray", fontsize=12)
ax2.tick_params(axis="y", labelcolor="gray", labelsize=12)

lines = [line1, line2, line3, line4]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="upper left")

plt.tight_layout()

# Save as PDF
plt.savefig(
    f"../results/profit_vs_price_diff_{max_price_diff}_{bridging_fee}.pdf",
    bbox_inches="tight",
    facecolor="auto",
    edgecolor="auto",
)
# plt.show()
