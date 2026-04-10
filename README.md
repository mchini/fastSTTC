# faststtc — Fast Spike Time Tiling Coefficient for Python

A fast, easy-to-use Python package for computing the Spike Time Tiling Coefficient (STTC) for neural spike train data.

---

## What is STTC?

STTC is a measure of how often two neurons fire close together in time. It produces a number between −1 and +1: a value near 0 means the two neurons fire independently (no more than expected by chance given their individual firing rates), 
a positive value means they tend to fire together, and a negative value means they tend to avoid firing at the same time. Unlike older measures such as the correlation index, STTC does not depend on firing rate — 
two neurons that both fire very rarely will not appear correlated simply because they share many silent periods. STTC was introduced by Cutts & Eglen (J. Neurosci., 2014) and has 
become a very widely used measure of pairwise spike train interactions.

---

## Why use this package?

- **2–9× faster** than existing NumPy implementations for short recordings (≤ 30,000 ms)
- **1.7–2.7× faster** for long recordings (≥ 300,000 ms, 5 minutes at 1 ms bins)
- Minimal dependencies — just **NumPy and SciPy**
- Statistical testing built in (null distribution + z-scores)
- Simple API: **one function call per task**

---

## How it works — and why it is fast

Most STTC implementations work by looping over every pair of units (i, j) and, for each pair, looping over every spike to count how many fall within the time window of the other unit. 
This is slow because the work scales with the number of pairs (N²) and the number of spikes, and Python loops are expensive.

`faststtc` replaces these loops with two vectorised operations that run over all units at once:

1. **Cumulative-sum tiling.** To mark every time bin within ±Δt of any spike, the standard approach applies a sliding window — effectively a convolution — which requires iterating over each bin. 
Instead, `faststtc` uses a cumulative-sum trick: compute the running total of spike counts, then subtract two offset copies of it. The result is a binary "tiled" matrix (one row per unit) that marks 
every bin covered by at least one spike window, computed in a single vectorised pass with no Python loops.

2. **Matrix multiply for all coincidences.** Once the tiled matrix is built, the number of spikes from unit A that fall inside the tiles of unit B is simply the dot product of A's spike row with B's tile row. 
Doing this for all N² pairs simultaneously is a single matrix multiply (`spike_matrix @ tiled_matrix.T`), which NumPy hands off to a highly optimised BLAS routine. This replaces an explicit double loop over pairs.

The null distribution (surrogate spike trains for statistical testing) uses a further optimisation: because the tiled version of the *original* signal does not change across surrogate iterations, 
it is computed once before the loop rather than once per iteration. This idea was inspired by [STTCPy](https://github.com/jeremi-chabros/STTCPy), which introduced the same hoisting strategy; `faststtc` extends it to the full vectorised setting.

I wrote (many years ago) the core algorithm in MATLAB. It has since been translated to Python, restructured as an installable package, and optimised with the help of Claude.

---

## Installation

```bash
pip install faststtc
```

For development (includes pytest, matplotlib, and Jupyter):

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/faststtc
cd faststtc
pip install -e ".[dev]"
```

---

## Quick start

### Example 1: Pairwise STTC

```python
import faststtc as ft

# spike_times is a list: one entry per unit, each entry is an array of
# spike times in milliseconds
spike_matrix = ft.bin_spike_times(spike_times, T_ms=300_000)

S = ft.sttc(spike_matrix, dt_values=50)   # Δt = 50 ms
print(S)   # (N × N) matrix
```

### Example 2: Is the correlation statistically significant?

```python
z = ft.zscore_sttc(spike_matrix, dt=50, n_shifts=200)
# z[i,j] > 3  →  units i and j co-fire significantly more than chance
```

### Example 3: Compute STTC at multiple time windows

```python
S = ft.sttc(spike_matrix, dt_values=[10, 25, 50, 100])
# S has shape (N, N, 4) — one matrix per Δt
```

---

## Data format

Understanding the data format is the most common source of confusion for new users:

- **`bin_spike_times`** expects a **list** where each element is a 1-D NumPy array of spike times **in milliseconds**. For example, if you have 50 units, you pass a list of 50 arrays.
- **`T_ms`** is the **total recording duration in milliseconds**. For a 5-minute recording, `T_ms = 300_000`.
- The **binned spike matrix** returned by `bin_spike_times` is a 2-D NumPy array of shape `(N units, T bins)` with dtype `uint8`, where `1` means a spike occurred and `0` means no spike. This matrix is what you pass to `sttc()`, `sttc_null()`, and `zscore_sttc()`.

---

## Choosing Δt

The half-window `dt` determines how close two spikes must be in time to count as co-firing. For multi-electrode array recordings of retinal waves or cortical activity, `dt = 50 ms` is a common default. For studies of spike-timing-dependent plasticity, 10–20 ms is typical. You can pass a list to `sttc()` to compute at multiple values in a single call, which is more efficient than calling `sttc()` multiple times.

---

## Choosing `n_shifts` for the null distribution

`n_shifts=200` (the default) is sufficient for exploratory analysis. Use `n_shifts >= 500` for results intended for publication to ensure the null distribution is well sampled.

---

## How it works — and why it is fast

Most STTC implementations work by looping over every pair of units (i, j) and, for each pair, looping over every spike to count how many fall within the time window of the other unit. This is slow because the work scales with the number of pairs (N²) and the number of spikes, and Python loops are expensive.

`faststtc` replaces these loops with two vectorised operations that run over all units at once:

1. **Cumulative-sum tiling.** To mark every time bin within ±Δt of any spike, the standard approach applies a sliding window — effectively a convolution — which requires iterating over each bin. Instead, `faststtc` uses a cumulative-sum trick: compute the running total of spike counts, then subtract two offset copies of it. The result is a binary "tiled" matrix (one row per unit) that marks every bin covered by at least one spike window, computed in a single vectorised pass with no Python loops.

2. **Matrix multiply for all coincidences.** Once the tiled matrix is built, the number of spikes from unit A that fall inside the tiles of unit B is simply the dot product of A's spike row with B's tile row. Doing this for all N² pairs simultaneously is a single matrix multiply (`spike_matrix @ tiled_matrix.T`), which NumPy hands off to a highly optimised BLAS routine. This replaces an explicit double loop over pairs.

The null distribution (surrogate spike trains for statistical testing) uses a further optimisation: because the tiled version of the *original* signal does not change across surrogate iterations, it is computed once before the loop rather than once per iteration. This idea was inspired by [STTCPy](https://github.com/jeremi-chabros/STTCPy), which introduced the same hoisting strategy; `faststtc` extends it to the full vectorised setting.

The core algorithm was originally written in MATLAB. It has since been translated to Python, restructured as an installable package, and optimised with the help of Claude (Anthropic).

---

## Citation

If you use this package, please cite the original STTC paper:

> Cutts CS, Eglen SJ (2014). Detecting pairwise correlations in spike trains: an objective comparison of methods and application to the study of retinal waves. *J Neurosci* 34(43):14288–14303.

---

## License

MIT
