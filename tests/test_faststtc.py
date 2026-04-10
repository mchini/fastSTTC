"""
Test suite for the faststtc package.

Run with:  pytest tests/
"""
import numpy as np
import numpy.testing as npt
import pytest

from faststtc._tiling import tile_spikes
from faststtc import (
    sttc,
    sttc_null,
    zscore_sttc,
    bin_spike_times,
    generate_spike_trains,
)


# ─────────────────────────────────────────────────────────────────────────────
# TestTiling
# ─────────────────────────────────────────────────────────────────────────────

class TestTiling:

    def test_tile_output_values(self):
        """Spikes at t=1 and t=5 with dt=1 should tile bins 0-2 and 4-6."""
        mat = np.zeros((1, 8), dtype=np.uint8)
        mat[0, 1] = 1
        mat[0, 5] = 1
        tiled = tile_spikes(mat, dt=1)
        expected = np.array([[1, 1, 1, 0, 1, 1, 1, 0]], dtype=np.float32)
        npt.assert_array_equal(tiled, expected)

    def test_tile_zero_dt(self):
        """dt=0 should return the input unchanged (as float32)."""
        mat = np.array([[1, 0, 0, 1, 0]], dtype=np.uint8)
        tiled = tile_spikes(mat, dt=0)
        npt.assert_array_equal(tiled, mat.astype(np.float32))

    def test_tile_shape(self):
        """Output shape must equal input shape."""
        mat = np.random.randint(0, 2, size=(5, 100), dtype=np.uint8)
        tiled = tile_spikes(mat, dt=10)
        assert tiled.shape == mat.shape


# ─────────────────────────────────────────────────────────────────────────────
# TestSTTC
# ─────────────────────────────────────────────────────────────────────────────

class TestSTTC:

    @pytest.fixture
    def small_matrix(self):
        """4 units, 1000 bins, seed=0."""
        trains = generate_spike_trains(4, T_ms=1000, rate_hz=10, seed=0)
        return bin_spike_times(trains, T_ms=1000)

    def test_sttc_symmetry(self, small_matrix):
        """STTC matrix must be symmetric: result[i,j] == result[j,i]."""
        S = sttc(small_matrix, dt_values=20)
        npt.assert_array_almost_equal(S, S.T, decimal=6)

    def test_sttc_range(self, small_matrix):
        """All off-diagonal values must be in [-1, 1]."""
        S = sttc(small_matrix, dt_values=20)
        off_diag = S[~np.isnan(S)]
        assert np.all(off_diag >= -1.0 - 1e-6)
        assert np.all(off_diag <= 1.0 + 1e-6)

    def test_sttc_identical_trains(self):
        """STTC of identical trains should be 1."""
        trains = generate_spike_trains(1, T_ms=5000, rate_hz=5, seed=1)
        mat = bin_spike_times(trains * 2, T_ms=5000)   # duplicate the single unit
        S = sttc(mat, dt_values=50)
        assert abs(S[0, 1] - 1.0) < 1e-5

    def test_sttc_diagonal_nan(self, small_matrix):
        """Diagonal entries must all be NaN."""
        S = sttc(small_matrix, dt_values=20)
        assert np.all(np.isnan(np.diag(S)))

    def test_sttc_scalar_dt(self, small_matrix):
        """Scalar dt_values must return (N, N), not (N, N, 1)."""
        S = sttc(small_matrix, dt_values=20)
        assert S.ndim == 2
        assert S.shape == (small_matrix.shape[0], small_matrix.shape[0])

    def test_sttc_multi_dt(self, small_matrix):
        """List dt_values must return (N, N, n_dt)."""
        N = small_matrix.shape[0]
        S = sttc(small_matrix, dt_values=[10, 20, 50])
        assert S.shape == (N, N, 3)

    def test_sttc_independent(self):
        """Two independent Poisson units should have STTC close to 0."""
        rng = np.random.default_rng(42)
        T = 300_000
        mat = rng.integers(0, 2, size=(2, T), dtype=np.uint8)
        # Force low firing rate to make independence clearer
        mat = (mat == 0).astype(np.uint8)   # ~50% density — fine for check
        trains = generate_spike_trains(2, T_ms=300_000, rate_hz=5,
                                       correlation=0.0, seed=42)
        mat2 = bin_spike_times(trains, T_ms=300_000)
        S = sttc(mat2, dt_values=50)
        assert abs(S[0, 1]) < 0.15   # loose bound for stochastic data

    def test_sttc_zero_spike_unit(self):
        """A unit with no spikes should yield NaN for its entire row/column."""
        mat = np.zeros((3, 500), dtype=np.uint8)
        mat[1, 100] = 1
        mat[2, 200] = 1
        S = sttc(mat, dt_values=10)
        assert np.all(np.isnan(S[0, :]))
        assert np.all(np.isnan(S[:, 0]))


# ─────────────────────────────────────────────────────────────────────────────
# TestSTTCNull
# ─────────────────────────────────────────────────────────────────────────────

class TestSTTCNull:

    @pytest.fixture
    def mat(self):
        trains = generate_spike_trains(4, T_ms=10_000, rate_hz=5, seed=7)
        return bin_spike_times(trains, T_ms=10_000)

    def test_null_shape(self, mat):
        """Output shape must be (n_shifts, N, N)."""
        null = sttc_null(mat, dt=50, n_shifts=20, seed=0)
        N = mat.shape[0]
        assert null.shape == (20, N, N)

    def test_null_diagonal_nan(self, mat):
        """Diagonal of each shift matrix must be NaN."""
        null = sttc_null(mat, dt=50, n_shifts=10, seed=0)
        for s in range(null.shape[0]):
            assert np.all(np.isnan(np.diag(null[s])))

    def test_null_range(self, mat):
        """All non-NaN null values must be in [-1, 1]."""
        null = sttc_null(mat, dt=50, n_shifts=20, seed=0)
        vals = null[~np.isnan(null)]
        assert np.all(vals >= -1.0 - 1e-6)
        assert np.all(vals <= 1.0 + 1e-6)

    def test_null_independent_mean(self):
        """Mean of null distribution for independent units should be near 0."""
        trains = generate_spike_trains(4, T_ms=100_000, rate_hz=5,
                                       correlation=0.0, seed=13)
        mat = bin_spike_times(trains, T_ms=100_000)
        null = sttc_null(mat, dt=50, n_shifts=100, seed=0)
        mean_null = np.nanmean(null)
        assert abs(mean_null) < 0.1

    def test_null_seed_reproducible(self, mat):
        """Same seed must give identical results."""
        n1 = sttc_null(mat, dt=50, n_shifts=10, seed=99)
        n2 = sttc_null(mat, dt=50, n_shifts=10, seed=99)
        npt.assert_array_equal(n1, n2)


# ─────────────────────────────────────────────────────────────────────────────
# TestZScoreSTTC
# ─────────────────────────────────────────────────────────────────────────────

class TestZScoreSTTC:

    def test_zscore_shape(self):
        """Output must be (N, N)."""
        trains = generate_spike_trains(4, T_ms=10_000, rate_hz=5, seed=0)
        mat = bin_spike_times(trains, T_ms=10_000)
        z = zscore_sttc(mat, dt=50, n_shifts=20, seed=0)
        N = mat.shape[0]
        assert z.shape == (N, N)

    def test_zscore_correlated(self):
        """Highly correlated units should have z-score > 5."""
        trains = generate_spike_trains(2, T_ms=300_000, rate_hz=5,
                                       correlation=0.9, seed=42)
        mat = bin_spike_times(trains, T_ms=300_000)
        z = zscore_sttc(mat, dt=50, n_shifts=200, seed=0)
        assert z[0, 1] > 5.0

    def test_zscore_independent(self):
        """Independent units should have z-score near 0."""
        trains = generate_spike_trains(2, T_ms=300_000, rate_hz=5,
                                       correlation=0.0, seed=42)
        mat = bin_spike_times(trains, T_ms=300_000)
        z = zscore_sttc(mat, dt=50, n_shifts=200, seed=0)
        assert abs(z[0, 1]) < 4.0   # loose bound; stochastic


# ─────────────────────────────────────────────────────────────────────────────
# TestBinSpikeTimes
# ─────────────────────────────────────────────────────────────────────────────

class TestBinSpikeTimes:

    def test_bin_basic(self):
        """Spikes at 0, 500, 999 ms (bin_size=1) should set the correct bins."""
        mat = bin_spike_times([[0.0, 500.0, 999.0]], T_ms=1000)
        assert mat[0, 0]   == 1
        assert mat[0, 500] == 1
        assert mat[0, 999] == 1
        assert mat[0].sum() == 3

    def test_bin_shape(self):
        """Output shape must be (N, T_bins)."""
        trains = [[0.0, 100.0], [50.0, 200.0], [300.0]]
        mat = bin_spike_times(trains, T_ms=500, bin_size_ms=1)
        assert mat.shape == (3, 500)

    def test_bin_collision(self):
        """Two spikes in the same bin should produce a value of 1, not 2."""
        mat = bin_spike_times([[0.0, 0.3]], T_ms=10, bin_size_ms=1)
        assert mat[0, 0] == 1
        assert mat[0].sum() == 1

    def test_bin_empty_unit(self):
        """A unit with no spikes should produce an all-zero row."""
        mat = bin_spike_times([[], [100.0]], T_ms=500)
        npt.assert_array_equal(mat[0], np.zeros(500, dtype=np.uint8))

    def test_bin_custom_bin_size(self):
        """bin_size_ms=2 should halve the number of bins."""
        mat1 = bin_spike_times([[100.0]], T_ms=1000, bin_size_ms=1)
        mat2 = bin_spike_times([[100.0]], T_ms=1000, bin_size_ms=2)
        assert mat2.shape[1] == mat1.shape[1] // 2
