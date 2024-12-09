import numpy as np
import pandas as pd


def process_ticker_core(
    ticker,
    dates,
    spread,
    long_mean_spread,
    short_mean_spread,
    spread_std,
    long_upper_bands,
    long_lower_bands,
    short_upper_bands,
    short_lower_bands,
    slice_ends,
    max_entry_dollars,
    capital,
    max_indiv_allocation,
    limit_multiple_entries,
    print_trades,
    stop_multiplier,
    ticker_prices=None,
    ref_prices=None,
):
    n = len(spread)
    k = len(long_upper_bands[0])
    positions = np.zeros((n, k))
    entry_prices = np.zeros(k)
    stop_prices = np.zeros(k)  # Array to store stop prices for each position
    pl_per_trade = [[] for _ in range(k)]
    num_trades = np.zeros(k, dtype=np.int32)

    for i in range(1, n):
        # Calculate position size, respecting max_indiv_allocation
        pos_size = min(max_entry_dollars[i] / capital, max_indiv_allocation)
        # Initialize new_entries and position_changes for this time step
        new_entries = np.zeros(k, dtype=bool)
        position_changes = np.zeros(k)

        for j in range(k):
            current_position = positions[i - 1, j]

            # Long Position Management
            if current_position > 0:

                # Existing exit conditions
                if spread[i] > short_upper_bands[i, 0]:
                    # Close long and enter short
                    pl = (spread[i] - entry_prices[j]) * current_position
                    pl_per_trade[j].append(pl)
                    num_trades[j] += 1
                    position_changes[j] = (
                        -current_position - pos_size
                    )  # Close long and enter short
                    entry_prices[j] = spread[i]
                    stop_prices[j] = entry_prices[j] + stop_multiplier * spread_std[i]
                    new_entries[j] = True
                    if print_trades:
                        print(
                            f"{dates[i]} Closed long size {current_position} and entered short size {-pos_size}, price {ticker_prices[i]}, ref price {ref_prices[i]}, spread {spread[i]}, P&L {pl}"
                        )
                elif spread[i] > long_mean_spread[i]:
                    # Close long position
                    pl = (spread[i] - entry_prices[j]) * current_position
                    pl_per_trade[j].append(pl)
                    num_trades[j] += 1
                    position_changes[j] = -current_position
                    entry_prices[j] = 0
                    stop_prices[j] = 0
                    if print_trades:
                        print(
                            f"{dates[i]} Closed long size {current_position}, price {ticker_prices[i]}, ref price {ref_prices[i]}, spread {spread[i]}, P&L {pl}"
                        )

            # Short Position Management
            elif current_position < 0:

                # Existing exit conditions
                if spread[i] < long_lower_bands[i, 0]:
                    # Close short and enter long
                    pl = (entry_prices[j] - spread[i]) * abs(current_position)
                    pl_per_trade[j].append(pl)
                    num_trades[j] += 1
                    position_changes[j] = (
                        -current_position + pos_size
                    )  # Close short and enter long
                    entry_prices[j] = spread[i]
                    stop_prices[j] = entry_prices[j] - stop_multiplier * spread_std[i]
                    new_entries[j] = True
                    if print_trades:
                        print(
                            f"{dates[i]} Closed short size {current_position} and entered long size {pos_size}, price {ticker_prices[i]}, ref price {ref_prices[i]}, spread {spread[i]}, P&L {pl}"
                        )
                elif spread[i] < short_mean_spread[i]:
                    # Close short position
                    pl = (entry_prices[j] - spread[i]) * abs(current_position)
                    pl_per_trade[j].append(pl)
                    num_trades[j] += 1
                    position_changes[j] = -current_position
                    entry_prices[j] = 0
                    stop_prices[j] = 0
                    if print_trades:
                        print(
                            f"{dates[i]} Closed short size {current_position}, price {ticker_prices[i]}, ref price {ref_prices[i]}, spread {spread[i]}, P&L {pl}"
                        )

            # No Position - Entry Conditions
            else:
                curr_spread = spread[i]
                long_band = long_lower_bands[i, j]
                short_band = short_upper_bands[i, j]
                if (
                    spread[i - 1] >= long_lower_bands[i, j]
                    and spread[i] < long_lower_bands[i, j]
                ):
                    # Enter long position
                    position_changes[j] = pos_size
                    entry_prices[j] = spread[i]
                    stop_prices[j] = entry_prices[j] - stop_multiplier * spread_std[i]
                    new_entries[j] = True
                    if print_trades:
                        print(
                            f"{dates[i]} New long, size {pos_size}, price {ticker_prices[i]}, ref price {ref_prices[i]}, spread {spread[i]}"
                        )
                elif (
                    spread[i - 1] <= short_upper_bands[i, j]
                    and spread[i] > short_upper_bands[i, j]
                ):
                    # Enter short position
                    position_changes[j] = -pos_size
                    entry_prices[j] = spread[i]
                    stop_prices[j] = entry_prices[j] + stop_multiplier * spread_std[i]
                    new_entries[j] = True
                    if print_trades:
                        print(
                            f"{dates[i]} New short, size {-pos_size}, price {ticker_prices[i]}, ref price {ref_prices[i]}, spread {spread[i]}"
                        )

        # Apply position changes
        positions[i] = positions[i - 1] + position_changes

    return positions


def vectorized_backtest_reversals_single_net_position(
    dates,
    spread,
    long_mean_spread,
    short_mean_spread,
    long_lower_bands,
    long_upper_bands,
    short_lower_bands,
    short_upper_bands,
    slice_ends,
    max_entry_dollars,
    capital,
    max_indiv_allocation,
):
    n = len(spread)
    k = long_lower_bands.shape[1]

    pos_size = np.minimum(
        max_entry_dollars / capital, max_indiv_allocation
    )  # shape (n,)

    prev_spread = np.roll(spread, 1)
    prev_spread[0] = spread[0]

    # Entry signals
    long_entry_signal = (prev_spread[:, None] >= long_lower_bands) & (
        spread[:, None] < long_lower_bands
    )
    short_entry_signal = (prev_spread[:, None] <= short_upper_bands) & (
        spread[:, None] > short_upper_bands
    )

    # Potential entries ignoring current state
    all_entries = (long_entry_signal * pos_size[:, None]) - (
        short_entry_signal * pos_size[:, None]
    )

    # Preliminary positions to check flatness
    prelim_positions = np.cumsum(all_entries, axis=0)
    prev_day_positions = np.vstack([np.zeros((1, k)), prelim_positions[:-1, :]])
    flat_at_start = prev_day_positions == 0.0

    # Apply flat mask
    changes = np.where(flat_at_start, all_entries, 0.0)

    # Preliminary after flat mask
    preliminary_positions_after_mask = np.cumsum(changes, axis=0)
    pos_yesterday = np.vstack(
        [np.zeros((1, k)), preliminary_positions_after_mask[:-1, :]]
    )

    currently_long_yesterday = pos_yesterday > 0
    currently_short_yesterday = pos_yesterday < 0

    # Reversal logic
    reversal_to_short = currently_long_yesterday & short_entry_signal
    reversal_to_long = currently_short_yesterday & long_entry_signal

    i_rev_short, j_rev_short = np.where(reversal_to_short)
    for idx in range(len(i_rev_short)):
        i = i_rev_short[idx]
        j = j_rev_short[idx]
        changes[i, j] = -pos_yesterday[i, j] - pos_size[i]

    i_rev_long, j_rev_long = np.where(reversal_to_long)
    for idx in range(len(i_rev_long)):
        i = i_rev_long[idx]
        j = j_rev_long[idx]
        changes[i, j] = -pos_yesterday[i, j] + pos_size[i]

    # After reversal
    preliminary_positions_after_reversal = np.cumsum(changes, axis=0)

    slice_ends_set = set(slice_ends)
    slice_ends_mask = np.array([i in slice_ends_set for i in range(n)])[:, None]

    currently_long = preliminary_positions_after_reversal > 0
    currently_short = preliminary_positions_after_reversal < 0

    exit_condition_long = currently_long & (spread[:, None] > long_mean_spread[:, None])
    exit_condition_short = currently_short & (
        spread[:, None] < short_mean_spread[:, None]
    )

    exit_signal = slice_ends_mask | exit_condition_long | exit_condition_short

    preliminary_positions_after_exit_check = np.cumsum(changes, axis=0)
    exit_days, exit_levels = np.where(exit_signal)
    for idx in range(len(exit_days)):
        i = exit_days[idx]
        j = exit_levels[idx]
        changes[i, j] -= preliminary_positions_after_exit_check[i, j]

    # Final positions per band level
    positions = np.cumsum(changes, axis=0)

    # Aggregate across k levels to get a single net position column
    net_position = positions.sum(axis=1)

    return (
        net_position,
        positions,
        changes,
        long_entry_signal,
        short_entry_signal,
        exit_signal,
    )


if __name__ == "__main__":
    # Example test
    np.random.seed(42)
    n = 50
    dates = pd.date_range("2020-01-01", periods=n)

    t = np.linspace(0, 4 * np.pi, n)
    spread = 1.5 * np.sin(t)

    long_mean_spread = np.zeros(n)
    short_mean_spread = np.full(n, 0.5)

    k_levels = [2, 4, 6]
    k = len(k_levels)

    long_lower_bands = np.zeros((n, k))
    long_upper_bands = np.full((n, k), 1.0)
    short_lower_bands = long_lower_bands.copy()
    short_upper_bands = long_upper_bands.copy()

    slice_ends = [n - 1]
    max_entry_dollars = np.full(n, 1000.0)
    capital = 10000.0
    max_indiv_allocation = 0.1

    (
        net_position,
        positions,
        changes,
        long_entry_signal,
        short_entry_signal,
        exit_signal,
    ) = vectorized_backtest_reversals_single_net_position(
        dates=dates,
        spread=spread,
        long_mean_spread=long_mean_spread,
        short_mean_spread=short_mean_spread,
        long_lower_bands=long_lower_bands,
        long_upper_bands=long_upper_bands,
        short_lower_bands=short_lower_bands,
        short_upper_bands=short_upper_bands,
        slice_ends=slice_ends,
        max_entry_dollars=max_entry_dollars,
        capital=capital,
        max_indiv_allocation=max_indiv_allocation,
    )

    print("Dates:", dates)
    print("Spread:", spread.round(2))
    print("Net Position:\n", net_position)
    print("Positions per k-level:\n", positions)
    print("Changes:\n", changes)
