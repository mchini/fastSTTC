"""
Pairwise Spike Time Tiling Coefficient.

Public functions
----------------
sttc(spike_matrix, dt_values)
sttc_null(spike_matrix, dt, n_shifts, shuffle, seed)
zscore_sttc(spike_matrix, dt, n_shifts, shuffle, seed)
"""
import numpy as np
from ._tiling import tile_spikes


def sttc(
    spike_matrix: np.ndarray,
    dt_values,
) -> np.ndarray:
    """
    Pairwise STTC for all units at one or more Δt values.

    Strategy
    --------
    For each Δt:
      1. Build the tiled matrix with the cumsum trick  — O(N·T)
      2. Compute all N² coincidence counts in **one BLAS matmul**:
             C = spike_matrix  @  tiled.T        shape (N, N)
      3. Derive PA, TA and apply the STTC formula element-wise.

    The only Python loop is over ``dt_values`` (typically ≤ 20 iterations).

    Parameters
    ----------
    spike_matrix : (N, T) binary array (uint8 or bool)
        N units × T time bins. Each 1 indicates a spike in that bin.
    dt_values : int or sequence of int
        Half-window(s) in bins.  Pass a single int for one Δt; pass a list
        for several Δt values computed in one call.

    Returns
    -------
    result : np.ndarray
        - If ``dt_values`` is a scalar int: shape (N, N), float32.
          Diagonal entries are NaN (a unit is not compared with itself).
        - If ``dt_values`` is a sequence: shape (N, N, n_dt), float32.
          ``result[:, :, k]`` is the STTC matrix at ``dt_values[k]``.

    Examples
    --------
    >>> mat = bin_spike_times(spike_times, T_ms=300_000)
    >>> S = sttc(mat, dt_values=50)          # single Δt → (N, N)
    >>> S = sttc(mat, dt_values=[10,25,50])  # three Δt  → (N, N, 3)
    """
    scalar_input = np.isscalar(dt_values)
    dt_list = [dt_values] if scalar_input else list(dt_values)
    n_dt    = len(dt_list)

    sf      = spike_matrix.astype(np.float32)
    N, T    = sf.shape
    n_sp    = sf.sum(axis=1)                    # (N,) spike counts

    out = np.full((N, N, n_dt), np.nan, dtype=np.float32)

    for k, dt in enumerate(dt_list):
        tiled = tile_spikes(spike_matrix, dt)   # (N, T) float32
        TA    = tiled.mean(axis=1)              # (N,)
        C     = sf @ tiled.T                   # (N, N) — all coincidences

        with np.errstate(invalid="ignore", divide="ignore"):
            PA = np.where(n_sp[:, None] > 0, C / n_sp[:, None], np.nan)   # (N,N)

            # PA[a,b] = fraction of a's spikes in tiles of b
            # PB[a,b] = PA[b,a] = fraction of b's spikes in tiles of a
            d1  = 1.0 - PA   * TA[None, :]     # denominator for (PA-TB)/(1-PA·TB)
            d2  = 1.0 - PA.T * TA[:, None]     # denominator for (PB-TA)/(1-PB·TA)
            t1  = np.where(np.abs(d1) > 1e-10, (PA   - TA[None, :]) / d1, np.nan)
            t2  = np.where(np.abs(d2) > 1e-10, (PA.T - TA[:, None]) / d2, np.nan)

        mat_k = 0.5 * (t1 + t2)
        np.fill_diagonal(mat_k, np.nan)
        out[:, :, k] = mat_k

    return out[:, :, 0] if scalar_input else out


def sttc_null(
    spike_matrix: np.ndarray,
    dt: int,
    n_shifts: int = 200,
    shuffle: bool = False,
    seed=None,
) -> np.ndarray:
    """
    Null distribution for pairwise STTC via circular shifts or random shuffles.

    For each of ``n_shifts`` iterations the spike trains are either circularly
    shifted by a random amount or randomly shuffled (destroying temporal
    structure while preserving spike counts), and the pairwise STTC is
    recomputed.  The resulting distribution approximates what STTC values look
    like when there is **no genuine correlation** between units.

    Optimisation (borrowed from STTCPy, vectorised here):
        The tiled version of the *original* signal and TA are constant across
        all shifts → computed **once** before the loop.  Each iteration only
        tiles the shifted signal (one O(N·T) call instead of two).

    Parameters
    ----------
    spike_matrix : (N, T) binary array (uint8 or bool)
    dt           : int — half-window in bins (same Δt as used for the real STTC)
    n_shifts     : int — number of surrogate matrices to generate (default 200)
    shuffle      : bool — if True, randomly permute time bins instead of
                   circularly shifting.  Circular shift (default) is preferred
                   because it preserves the autocorrelation structure of each
                   unit; shuffle destroys it.
    seed         : int or None — random seed for reproducibility

    Returns
    -------
    null : (n_shifts, N, N) float32 array
        ``null[s, i, j]`` is the STTC between unit i and the surrogate of
        unit j at shift s.  Diagonal entries are NaN.

    Notes
    -----
    Circular shifts are drawn uniformly from
    [max(dt+1, T//10),  T − max(dt+1, T//10)]
    to ensure each surrogate is genuinely decorrelated from the original.
    """
    rng   = np.random.default_rng(seed)
    sf    = spike_matrix.astype(np.float32)
    N, T  = sf.shape
    n_sp  = sf.sum(axis=1)                                  # (N,) — constant

    # ── hoist: tile original ONCE ─────────────────────────────────────────────
    tiled_orig = tile_spikes(spike_matrix, dt).astype(np.float32)   # (N, T)
    TA         = tiled_orig.mean(axis=1)                             # (N,)

    # shift values: random, well-separated from both boundaries
    min_shift = max(dt + 1, T // 10)
    shifts    = rng.integers(min_shift, T - min_shift, size=n_shifts)

    null = np.full((n_shifts, N, N), np.nan, dtype=np.float32)

    for s, shift_val in enumerate(shifts):
        if shuffle:
            shifted = spike_matrix[:, rng.permutation(T)]
        else:
            shifted = np.roll(spike_matrix, int(shift_val), axis=1)

        sf_s    = shifted.astype(np.float32)
        tiled_s = tile_spikes(shifted, dt).astype(np.float32)   # 1 tile call per iter
        TB      = tiled_s.mean(axis=1)                          # (N,)

        # C_AB[i,j] = #{spikes of orig[i]   within tiles of shifted[j]}
        # C_BA[i,j] = #{spikes of shifted[j] within tiles of orig[i]}
        C_AB = sf         @ tiled_s.T    # (N, N)  — orig spikes  × shifted tiles
        C_BA = tiled_orig @ sf_s.T       # (N, N)  — orig tiles   × shifted spikes
        #       ↑ reuses the hoisted tiled_orig

        with np.errstate(invalid="ignore", divide="ignore"):
            PA = np.where(n_sp[:, None] > 0, C_AB / n_sp[:, None], np.nan)
            PB = np.where(n_sp[None, :] > 0, C_BA / n_sp[None, :], np.nan)
            d1 = 1 - PA * TB[None, :]
            d2 = 1 - PB * TA[:, None]
            t1 = np.where(np.abs(d1) > 1e-10, (PA - TB[None, :]) / d1, np.nan)
            t2 = np.where(np.abs(d2) > 1e-10, (PB - TA[:, None]) / d2, np.nan)

        mat_s = 0.5 * (t1 + t2)
        np.fill_diagonal(mat_s, np.nan)
        null[s] = mat_s

    return null


def zscore_sttc(
    spike_matrix: np.ndarray,
    dt: int,
    n_shifts: int = 200,
    shuffle: bool = False,
    seed=None,
) -> np.ndarray:
    """
    Z-scored pairwise STTC.

    Computes the observed STTC matrix and a null distribution (via
    :func:`sttc_null`), then returns::

        z[i, j] = (STTC[i,j] − mean(null[:, i, j])) / std(null[:, i, j])

    A z-score above ~2–3 indicates that units i and j co-fire significantly
    more than expected by chance given their individual firing rates.

    Parameters
    ----------
    spike_matrix : (N, T) binary array (uint8 or bool)
    dt           : int — half-window in bins
    n_shifts     : int — surrogate count (default 200; use ≥ 500 for publication)
    shuffle      : bool — see :func:`sttc_null`
    seed         : int or None — for reproducibility

    Returns
    -------
    z : (N, N) float32 array.  Diagonal entries are NaN.
    """
    observed = sttc(spike_matrix, dt)
    null     = sttc_null(spike_matrix, dt, n_shifts=n_shifts,
                         shuffle=shuffle, seed=seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (observed - np.nanmean(null, axis=0)) / np.nanstd(null, axis=0)
    return z.astype(np.float32)
