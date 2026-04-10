"""
faststtc — Fast vectorised Spike Time Tiling Coefficient
=========================================================

Quick reference
---------------
Pairwise STTC (all pairs):
    sttc(spike_matrix, dt_values)

Statistical testing:
    sttc_null(spike_matrix, dt, n_shifts)
    zscore_sttc(spike_matrix, dt, n_shifts)

Data preparation:
    bin_spike_times(spike_times_list, T_ms)
    generate_spike_trains(N, T_ms, rate_hz, correlation)

See https://github.com/YOUR_GITHUB_USERNAME/faststtc for full documentation.
"""

from .sttc   import sttc, sttc_null, zscore_sttc
from .utils  import bin_spike_times, generate_spike_trains

__version__ = "0.1.0"

__all__ = [
    "sttc",
    "sttc_null",
    "zscore_sttc",
    "bin_spike_times",
    "generate_spike_trains",
]
