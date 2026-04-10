"""
Helper functions for data preparation and synthetic data generation.

Public functions
----------------
bin_spike_times(spike_times_list, T_ms, bin_size_ms)
generate_spike_trains(N, T_ms, rate_hz, seed)
"""
import numpy as np


def bin_spike_times(
    spike_times_list,
    T_ms: float,
    bin_size_ms: float = 1.0,
) -> np.ndarray:
    """
    Convert spike times to a binary spike matrix suitable for STTC.

    Parameters
    ----------
    spike_times_list : list of array-like
        Each element contains the spike times (in ms) for one unit.
        The list can have any length (= number of units).
    T_ms             : float — total recording duration in ms
    bin_size_ms      : float — width of each time bin in ms (default: 1 ms)

    Returns
    -------
    spike_matrix : (N, T_bins) uint8 array
        1 where a spike occurred, 0 elsewhere.
        If two spikes fall in the same bin only one is counted.

    Examples
    --------
    >>> spike_matrix = bin_spike_times(
    ...     spike_times_list=my_spike_times,   # list of N arrays, times in ms
    ...     T_ms=300_000,                       # 5-minute recording
    ...     bin_size_ms=1,                      # 1 ms bins (default)
    ... )
    >>> spike_matrix.shape   # (N, 300000)
    """
    T_bins = int(np.ceil(T_ms / bin_size_ms))
    N      = len(spike_times_list)
    mat    = np.zeros((N, T_bins), dtype=np.uint8)

    for i, spikes in enumerate(spike_times_list):
        spikes = np.asarray(spikes, dtype=np.float64)
        if len(spikes) == 0:
            continue
        idx = np.floor(spikes / bin_size_ms).astype(int)
        idx = idx[(idx >= 0) & (idx < T_bins)]
        mat[i, idx] = 1

    return mat


def generate_spike_trains(
    N: int,
    T_ms: float,
    rate_hz: float = 5.0,
    correlation: float = 0.0,
    seed=None,
) -> list:
    """
    Generate synthetic Poisson spike trains, optionally correlated.

    Useful for testing and for the examples notebook.

    Parameters
    ----------
    N           : int — number of units
    T_ms        : float — recording duration in ms
    rate_hz     : float — mean firing rate in Hz (default: 5)
    correlation : float in [0, 1] — fraction of spikes that are shared
                  across all units via a common Poisson process.
                  0 = independent, 1 = perfectly correlated.
    seed        : int or None

    Returns
    -------
    spike_times_list : list of N sorted float32 arrays (spike times in ms)
    """
    rng     = np.random.default_rng(seed)
    T_s     = T_ms / 1000.0

    spike_times_list = []

    # shared events (drive correlations)
    if correlation > 0:
        n_shared = rng.poisson(rate_hz * T_s * correlation)
        shared   = np.sort(rng.uniform(0, T_ms, n_shared).astype(np.float32))
    else:
        shared = np.array([], dtype=np.float32)

    indep_rate = rate_hz * (1.0 - correlation)

    for _ in range(N):
        n_indep = rng.poisson(indep_rate * T_s)
        indep   = rng.uniform(0, T_ms, n_indep).astype(np.float32)
        spikes  = np.sort(np.concatenate([shared, indep]))
        spike_times_list.append(spikes)

    return spike_times_list
