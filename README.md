# faststtc — Fast Spike Time Tiling Coefficient for Python

A fast, easy-to-use Python package for computing the Spike Time Tiling Coefficient (STTC) for neural spike train data.

---

## What is STTC?

STTC is a measure of how often two neurons fire close together in time. It produces a number between −1 and +1: a value near 0 means the two neurons fire independently (no more than expected by chance given their individual firing rates), a positive value means they tend to fire together, and a negative value means they tend to avoid firing at the same time. Unlike older measures such as the correlation index, STTC does not depend on firing rate — two neurons that both fire very rarely will not appear correlated simply because they share many silent periods. STTC was introduced by Cutts & Eglen (J. Neurosci., 2014) and has become the standard measure of pairwise spike train correlation in multi-electrode array studies.

---

## Why use this package?

- **2–9× faster** than existing NumPy implementations for short recordings (≤ 30,000 ms)
- **1.7–2.7× faster** for long recordings (≥ 300,000 ms, 5 minutes at 1 ms bins)
- No Neo or elephant dependency — just **NumPy and SciPy**
- Statistical testing built in (null distribution + z-scores)
- Simple API: **one function call per task**

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

## Citation

If you use this package, please cite the original STTC paper:

> Cutts CS, Eglen SJ (2014). Detecting pairwise correlations in spike trains: an objective comparison of methods and application to the study of retinal waves. *J Neurosci* 34(43):14288–14303.

---

## License

MIT
