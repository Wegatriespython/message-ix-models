"""Tests for tools.impacts.risk — ensemble reductions.

predict_rime is mocked at the risk module import site so tests exercise
the reduction logic without requiring RIME NetCDF data.
"""

from unittest.mock import patch

import numpy as np
import pytest

from message_ix_models.tools.impacts.risk import (
    cvar_coherent,
    cvar_pointwise,
    joint_at_quantile,
    predict_with_reduction,
)

# Canonical ensemble shape: (n_runs=10, n_spatial=5, n_years=3)
_RNG = np.random.default_rng(42)
_ENSEMBLE = _RNG.normal(100, 20, size=(10, 5, 3))

_GMT = np.linspace(1.0, 2.5, 3)
_GMT_2D = np.tile(_GMT, (10, 1))
_PATH = "dummy.nc"
_VAR = "qtot_mean"


def _mock_predict_rime(ensemble):
    """Return a mock for predict_rime that ignores inputs and yields *ensemble*."""
    return patch(
        "message_ix_models.tools.impacts.risk.predict_rime",
        return_value=ensemble,
    )


class TestCvarPointwise:
    def test_shape(self):
        with _mock_predict_rime(_ENSEMBLE):
            result = cvar_pointwise(_GMT, _PATH, _VAR, alpha=10)
        assert result.shape == (5, 3)

    def test_alpha_bounds(self):
        with _mock_predict_rime(_ENSEMBLE):
            with pytest.raises(ValueError, match="alpha must be between"):
                cvar_pointwise(_GMT, _PATH, _VAR, alpha=0)
            with pytest.raises(ValueError, match="alpha must be between"):
                cvar_pointwise(_GMT, _PATH, _VAR, alpha=100)

    def test_leq_expectation(self):
        """Lower-tail CVaR <= pointwise expectation at every cell."""
        with _mock_predict_rime(_ENSEMBLE):
            cvar_10 = cvar_pointwise(_GMT, _PATH, _VAR, alpha=10)
        expectation = np.mean(_ENSEMBLE, axis=0)
        assert np.all(cvar_10 <= expectation + 1e-10)

    def test_monotonicity(self):
        """CVaR_10 <= CVaR_50 at every cell (tighter tail is worse)."""
        with _mock_predict_rime(_ENSEMBLE):
            cvar_10 = cvar_pointwise(_GMT, _PATH, _VAR, alpha=10)
            cvar_50 = cvar_pointwise(_GMT, _PATH, _VAR, alpha=50)
        assert np.all(cvar_10 <= cvar_50 + 1e-10)

    def test_all_same(self):
        """Uniform ensemble: CVaR equals the constant value."""
        uniform = np.full((10, 5, 3), 7.0)
        with _mock_predict_rime(uniform):
            result = cvar_pointwise(_GMT, _PATH, _VAR, alpha=25)
        assert pytest.approx(result) == np.full((5, 3), 7.0)

    def test_known_values(self):
        """1-spatial, 1-year ensemble: worst 10% of 10 runs = bottom 1."""
        # runs sorted ascending: [10, 20, ..., 100]; worst 10% = [10], mean = 10
        values = np.arange(10, 101, 10, dtype=float).reshape(10, 1, 1)
        with _mock_predict_rime(values):
            result = cvar_pointwise(_GMT, _PATH, _VAR, alpha=10)
        assert pytest.approx(result[0, 0]) == 10.0


class TestCvarCoherent:
    def test_shape(self):
        with _mock_predict_rime(_ENSEMBLE):
            result = cvar_coherent(_GMT, _PATH, _VAR, alpha=10)
        assert result.shape == (5, 3)

    def test_alpha_bounds(self):
        with _mock_predict_rime(_ENSEMBLE):
            with pytest.raises(ValueError, match="alpha must be between"):
                cvar_coherent(_GMT, _PATH, _VAR, alpha=0)
            with pytest.raises(ValueError, match="alpha must be between"):
                cvar_coherent(_GMT, _PATH, _VAR, alpha=100)

    def test_global_mean_leq_expectation(self):
        """Coherent CVaR selects worst trajectories globally, so the
        global mean of the result is <= the global expectation."""
        with _mock_predict_rime(_ENSEMBLE):
            cvar_10 = cvar_coherent(_GMT, _PATH, _VAR, alpha=10)
        expectation = np.mean(_ENSEMBLE, axis=0)
        assert np.mean(cvar_10) <= np.mean(expectation) + 1e-10

    def test_all_same(self):
        """Uniform ensemble: CVaR equals the constant value."""
        uniform = np.full((10, 5, 3), 7.0)
        with _mock_predict_rime(uniform):
            result = cvar_coherent(_GMT, _PATH, _VAR, alpha=25)
        assert pytest.approx(result) == np.full((5, 3), 7.0)

    def test_selects_worst_trajectories(self):
        """Worst run (all values 0) must appear in coherent CVaR at alpha=10."""
        # Run 0: all zeros (worst globally). Runs 1-9: all ones.
        ensemble = np.ones((10, 5, 3))
        ensemble[0] = 0.0
        with _mock_predict_rime(ensemble):
            result = cvar_coherent(_GMT, _PATH, _VAR, alpha=10)
        # alpha=10 of 10 runs -> cutoff=1 -> selects run 0 -> result = 0
        assert pytest.approx(result) == np.zeros((5, 3))


class TestJointAtQuantile:
    def test_q_impact_must_be_shipped(self):
        with _mock_predict_rime(_ENSEMBLE):
            with pytest.raises(ValueError, match="q_impact must be one of"):
                joint_at_quantile(_GMT_2D, _PATH, _VAR, q_impact=0.25)
            with pytest.raises(ValueError, match="q_impact must be one of"):
                joint_at_quantile(_GMT_2D, _PATH, _VAR, q_impact=0.0)

    def test_q_warming_bounds(self):
        with _mock_predict_rime(_ENSEMBLE):
            with pytest.raises(ValueError, match="q_warming must be in"):
                joint_at_quantile(_GMT_2D, _PATH, _VAR, q_warming=0.0)
            with pytest.raises(ValueError, match="q_warming must be in"):
                joint_at_quantile(_GMT_2D, _PATH, _VAR, q_warming=1.0)

    def test_reads_percentile_variable(self):
        """Function rewrites var_name with the _pXX suffix matching q_impact."""
        with _mock_predict_rime(_ENSEMBLE) as mock:
            joint_at_quantile(_GMT_2D, _PATH, "capacity_factor", q_impact=0.5)
        # Last call: positional (gmt, path, var_name=...) — var_name is the third arg
        last_call = mock.call_args
        assert last_call.args[2] == "capacity_factor_p50"

    def test_p10_p50_p90_select_correct_var(self):
        for q, suffix in [(0.1, "_p10"), (0.5, "_p50"), (0.9, "_p90")]:
            with _mock_predict_rime(_ENSEMBLE) as mock:
                joint_at_quantile(_GMT_2D, _PATH, "EI_cool", q_impact=q)
            assert mock.call_args.args[2] == "EI_cool" + suffix

    def test_2d_matches_per_cell_quantile(self):
        """2D input: warming-axis reduction is np.quantile(..., q_warming, axis=0)."""
        with _mock_predict_rime(_ENSEMBLE):
            result = joint_at_quantile(
                _GMT_2D, _PATH, _VAR, q_impact=0.5, q_warming=0.5
            )
        assert result.shape == (5, 3)
        np.testing.assert_allclose(result, np.median(_ENSEMBLE, axis=0))

    def test_warming_quantile_continuous(self):
        """q_warming is forwarded to np.quantile verbatim."""
        with _mock_predict_rime(_ENSEMBLE):
            r25 = joint_at_quantile(_GMT_2D, _PATH, _VAR, q_warming=0.25)
            r75 = joint_at_quantile(_GMT_2D, _PATH, _VAR, q_warming=0.75)
        np.testing.assert_allclose(r25, np.quantile(_ENSEMBLE, 0.25, axis=0))
        np.testing.assert_allclose(r75, np.quantile(_ENSEMBLE, 0.75, axis=0))
        # Higher warming quantile is at least as high pointwise on this ensemble
        assert np.all(r75 >= r25 - 1e-10)

    def test_1d_input_skips_warming_reduction(self):
        """1D gmt: function returns the impact-percentile lookup verbatim."""
        flat = np.full((5, 3), 42.0)  # mock pretends predict_rime returned 2D
        with _mock_predict_rime(flat):
            result = joint_at_quantile(_GMT, _PATH, _VAR, q_impact=0.5)
        assert result.shape == (5, 3)
        np.testing.assert_allclose(result, flat)

    def test_uniform_ensemble_idempotent(self):
        uniform = np.full((10, 5, 3), 7.0)
        with _mock_predict_rime(uniform):
            result = joint_at_quantile(_GMT_2D, _PATH, _VAR)
        np.testing.assert_allclose(result, np.full((5, 3), 7.0))


class TestPredictWithReduction:
    def test_mean_dispatches_to_predict_rime(self):
        """reduction='mean' calls predict_rime with aggregate='mean' verbatim."""
        flat = np.full((5, 3), 1.5)
        with _mock_predict_rime(flat) as mock:
            result = predict_with_reduction(_GMT_2D, _PATH, _VAR, reduction="mean")
        np.testing.assert_allclose(result, flat)
        # Verify the underlying call asked for the mean field, not _p50
        assert mock.call_args.args[2] == _VAR
        assert mock.call_args.kwargs.get("aggregate") == "mean"

    def test_joint_p50_dispatches_to_quantile(self):
        """`joint_p50` reads `_p50` and requests the full run ensemble."""
        with _mock_predict_rime(_ENSEMBLE) as mock:
            result = predict_with_reduction(
                _GMT_2D, _PATH, "capacity_factor", reduction="joint_p50"
            )
        np.testing.assert_allclose(result, np.median(_ENSEMBLE, axis=0))
        assert mock.call_args.args[2] == "capacity_factor_p50"
        assert mock.call_args.kwargs.get("aggregate") == "none"

    def test_default_is_mean(self):
        flat = np.full((5, 3), 9.9)
        with _mock_predict_rime(flat):
            result = predict_with_reduction(_GMT_2D, _PATH, _VAR)
        np.testing.assert_allclose(result, flat)

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="Unsupported reduction mode"):
            predict_with_reduction(_GMT_2D, _PATH, _VAR, reduction="typo")  # type: ignore[arg-type]
