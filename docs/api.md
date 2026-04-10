# faststtc API Reference

## Which function should I use?

| I want to...                                   | Use this function          |
|------------------------------------------------|----------------------------|
| Measure co-firing between all pairs            | `sttc()`                   |
| Test if co-firing is statistically significant | `zscore_sttc()`            |
| Get the null distribution itself               | `sttc_null()`              |
| Convert spike times to a matrix                | `bin_spike_times()`        |
| Generate test data                             | `generate_spike_trains()`  |

---

## `sttc(spike_matrix, dt_values)`

Computes the pairwise Spike Time Tiling Coefficient for all pairs of units in a recording. For each pair (i, j), STTC measures how often unit i fires within a time window of ±Δt around any spike of unit j, and vice versa, then combines both directions into a single symmetric score between −1 and +1. A score near 0 means the two units fire independently; a positive score means they tend to co-fire; a negative score means they tend to avoid each other. You can pass a single Δt or a list to compute at multiple windows in one call.

| Parameter     | Type                    | Description                                                                 |
|---------------|-------------------------|-----------------------------------------------------------------------------|
| `spike_matrix`| `(N, T)` uint8 or bool  | Binary spike matrix: rows = units, columns = time bins, 1 = spike           |
| `dt_values`   | int or list of int      | Half-window(s) in bins. Pass a single int or a list for multiple Δt values  |

**Returns:** `np.ndarray`
- Shape `(N, N)` if `dt_values` is a scalar — one STTC matrix.
- Shape `(N, N, n_dt)` if `dt_values` is a list — one matrix per Δt, stacked along the last axis.
- Diagonal entries are `NaN` (a unit is not compared with itself).
- Units with zero spikes produce `NaN` for their entire row and column.

**Example:**
```python
import faststtc as ft

spike_matrix = ft.bin_spike_times(spike_times, T_ms=300_000)

# Single Δt
S = ft.sttc(spike_matrix, dt_values=50)      # shape (N, N)

# Multiple Δt values in one call
S = ft.sttc(spike_matrix, dt_values=[10, 25, 50])  # shape (N, N, 3)
print(S[:, :, 1])  # STTC at dt=25
```

**Notes:**
- The function uses a cumulative-sum tiling trick (O(N·T)) and a single BLAS matrix multiply to compute all N² coincidence counts at once. The only Python loop is over `dt_values`.
- `dt_values` is in **bins**, not milliseconds. If your bins are 1 ms wide and you want Δt = 50 ms, pass `dt_values=50`.

---

## `sttc_null(spike_matrix, dt, n_shifts, shuffle, seed)`

Generates a null distribution for the pairwise STTC by creating surrogate spike trains and recomputing STTC on each surrogate. Each surrogate is made by either circularly shifting or randomly shuffling the spike trains, which destroys genuine cross-unit correlations while approximately preserving each unit's firing rate and (for circular shifts) its autocorrelation structure. The resulting distribution tells you what STTC values look like when there is no genuine correlation between units, allowing you to assess whether observed STTC values are statistically significant.

| Parameter     | Type          | Description                                                                                   |
|---------------|---------------|-----------------------------------------------------------------------------------------------|
| `spike_matrix`| `(N, T)` uint8| Binary spike matrix                                                                           |
| `dt`          | int           | Half-window in bins (must match the Δt used for the real STTC)                                |
| `n_shifts`    | int           | Number of surrogate matrices to generate. Default: 200. Use ≥ 500 for publication.           |
| `shuffle`     | bool          | If True, randomly permute time bins instead of circular shift. Default: False.                |
| `seed`        | int or None   | Random seed for reproducibility                                                               |

**Returns:** `np.ndarray` of shape `(n_shifts, N, N)`, float32.
- `null[s, i, j]` is the STTC between unit i and the surrogate of unit j at iteration s.
- Diagonal entries are `NaN`.

**Example:**
```python
null = ft.sttc_null(spike_matrix, dt=50, n_shifts=200, seed=42)
# null has shape (200, N, N)

# Mean null value for the pair (0, 1):
import numpy as np
print(np.nanmean(null[:, 0, 1]))
```

**Notes:**
- Circular shifts are drawn uniformly from `[max(dt+1, T//10), T − max(dt+1, T//10)]` to ensure each surrogate is genuinely decorrelated.
- The tiled version of the original signal (and therefore TA) is constant across all shifts and is computed once before the loop, making this significantly faster than a naive implementation.

---

## `zscore_sttc(spike_matrix, dt, n_shifts, shuffle, seed)`

Computes the observed pairwise STTC matrix and a null distribution via `sttc_null`, then returns a z-score for each pair: how many standard deviations above the null mean is the observed value? A z-score above ~2–3 indicates that units i and j co-fire significantly more than expected by chance given their individual firing rates.

| Parameter     | Type          | Description                                                                 |
|---------------|---------------|-----------------------------------------------------------------------------|
| `spike_matrix`| `(N, T)` uint8| Binary spike matrix                                                         |
| `dt`          | int           | Half-window in bins                                                         |
| `n_shifts`    | int           | Surrogate count. Default: 200. Use ≥ 500 for publication.                  |
| `shuffle`     | bool          | See `sttc_null`. Default: False.                                            |
| `seed`        | int or None   | Random seed                                                                 |

**Returns:** `(N, N)` float32 array. Diagonal entries are `NaN`.

**Example:**
```python
z = ft.zscore_sttc(spike_matrix, dt=50, n_shifts=200, seed=42)

import numpy as np
# Count significantly correlated pairs (|z| > 3):
sig = np.sum(np.abs(z[np.triu_indices_from(z, k=1)]) > 3)
print(f"{sig} significantly correlated pairs")
```

---

## `bin_spike_times(spike_times_list, T_ms, bin_size_ms)`

Converts a list of spike time arrays (one per unit) into a binary spike matrix suitable for STTC computation. Each spike time is assigned to the bin `floor(t / bin_size_ms)`. If two spikes fall in the same bin, only one is counted (the matrix value is capped at 1).

| Parameter          | Type              | Description                                                             |
|--------------------|-------------------|-------------------------------------------------------------------------|
| `spike_times_list` | list of array-like| Each element is a 1-D array of spike times **in milliseconds** for one unit |
| `T_ms`             | float             | Total recording duration in milliseconds                                |
| `bin_size_ms`      | float             | Width of each time bin in ms. Default: 1 ms.                           |

**Returns:** `(N, T_bins)` uint8 array, where `T_bins = ceil(T_ms / bin_size_ms)`.

**Example:**
```python
# 3 units with different numbers of spikes
spike_times = [
    np.array([100.0, 250.0, 800.0]),   # unit 0
    np.array([101.0, 251.0]),           # unit 1
    np.array([]),                       # unit 2 — silent
]
mat = ft.bin_spike_times(spike_times, T_ms=1000, bin_size_ms=1)
# mat.shape == (3, 1000)
```

**Notes:**
- Spikes outside `[0, T_ms)` are silently ignored.
- Units with no spikes produce an all-zero row (not an error).

---

## `generate_spike_trains(N, T_ms, rate_hz, correlation, seed)`

Generates synthetic Poisson spike trains, optionally with shared firing events to introduce correlations. Useful for testing, benchmarking, and tutorials.

| Parameter     | Type        | Description                                                                                       |
|---------------|-------------|---------------------------------------------------------------------------------------------------|
| `N`           | int         | Number of units                                                                                   |
| `T_ms`        | float       | Recording duration in milliseconds                                                                |
| `rate_hz`     | float       | Mean firing rate in Hz. Default: 5 Hz.                                                            |
| `correlation` | float [0,1] | Fraction of spikes shared across all units via a common Poisson process. 0 = independent.        |
| `seed`        | int or None | Random seed                                                                                       |

**Returns:** List of N sorted float32 arrays (spike times in ms).

**Example:**
```python
# 10 independent units, 5 min recording, 5 Hz
trains = ft.generate_spike_trains(10, T_ms=300_000, rate_hz=5, seed=0)

# 10 correlated units (50% shared spikes)
trains_corr = ft.generate_spike_trains(10, T_ms=300_000, rate_hz=5,
                                        correlation=0.5, seed=0)

mat = ft.bin_spike_times(trains, T_ms=300_000)
S   = ft.sttc(mat, dt_values=50)
```
