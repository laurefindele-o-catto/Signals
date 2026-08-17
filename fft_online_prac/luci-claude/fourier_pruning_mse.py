import numpy as np
import sys


# =====================================================================
# Given class — paste your offline implementation where indicated
# =====================================================================

class FourierEpicycles:
    """
    Represents a closed curve via its Fourier series coefficients.
    (Given — paste your offline implementation of compute_coeffs/reconstruct/
    evaluate_reconstruction_error/prune_harmonics_by_energy here.)
    """

    def __init__(self, t, f_vals, N, T):
        self.t = t                # sample times, shape (M,)
        self.f_vals = f_vals      # ground-truth complex signal x(t)+i*y(t), shape (M,)
        self.N = N                # harmonics range -N..N
        self.T = T                # period
        self.coeffs = None        # shape (2N+1,), index n+N -> c_n

    def compute_coeffs(self):
        raise NotImplementedError("Paste your offline compute_coeffs here.")

    def reconstruct(self, coeffs=None):
        """(Given from offline) rebuild f_hat(t) at self.t from (possibly pruned) coeffs."""
        raise NotImplementedError("Paste your offline reconstruct here.")

    def evaluate_reconstruction_error(self, coeffs=None):
        """(Given from offline B) MSE between f_vals and reconstruction from coeffs."""
        raise NotImplementedError("Paste your offline evaluate_reconstruction_error here.")

    def prune_harmonics_by_energy(self, r):
        """(Given from offline B) energy-ratio pruning — paste it here for comparison."""
        raise NotImplementedError("Paste your offline prune_harmonics_by_energy here.")


# =====================================================================
# Task A — Top-K magnitude pruning
# =====================================================================

class TopKPruner:
    """Practice: prune to the K harmonics with largest |c_n|, instead of by energy ratio."""

    def prune_top_k_magnitude(self, epicycles: FourierEpicycles, k):
        """
        TODO:
          - Rank harmonics by |c_n| descending.
          - Keep the top k, zero the rest.
        Return (pruned_coeffs, retained_indices) where retained_indices are
        the harmonic numbers n (not array indices) that were kept.
        """
        raise NotImplementedError


# =====================================================================
# Task B — MSE vs N sweep
# =====================================================================

class MSESweep:
    """Practice: sweep reconstruction MSE as the number of retained harmonics grows."""

    def mse_vs_n_sweep(self, epicycles: FourierEpicycles, n_values):
        """
        TODO: for each cutoff n in n_values, truncate epicycles.coeffs to
        harmonics with |harmonic index| <= n (zero the rest), compute MSE via
        epicycles.evaluate_reconstruction_error, and collect results.
        Return a dict {n: mse}.
        """
        raise NotImplementedError

    def mse_vs_k_sweep(self, epicycles: FourierEpicycles, pruner: TopKPruner, k_values):
        """
        TODO: same idea, but using TopKPruner.prune_top_k_magnitude instead of
        a frequency cutoff. Return a dict {k: mse}.
        """
        raise NotImplementedError


# =====================================================================
# Validator — checks your filled-in methods against independent ground truth
# =====================================================================

class PruningValidator:
    """
    Runs independent checks against your implementations above.
    Does NOT reimplement your reconstruct()/MSE math — it checks the
    invariants those methods are supposed to satisfy.
    """

    def check_top_k_count_and_zeros(self, pruner: TopKPruner, epicycles: FourierEpicycles, k):
        pruned, retained = pruner.prune_top_k_magnitude(epicycles, k)
        nonzero_count = np.count_nonzero(pruned)
        count_ok = nonzero_count <= k
        # every retained coefficient should be among the k largest magnitudes
        order = np.argsort(np.abs(epicycles.coeffs))[::-1]
        true_top_k_indices = set(order[:k].tolist())
        kept_indices = set(np.nonzero(pruned)[0].tolist())
        subset_ok = kept_indices.issubset(true_top_k_indices)
        return (count_ok and subset_ok), nonzero_count

    def check_full_reconstruction_zero_error(self, epicycles: FourierEpicycles, tol=1e-6):
        """Using all coefficients (no pruning) should give ~0 MSE against ground truth."""
        mse = epicycles.evaluate_reconstruction_error(epicycles.coeffs)
        return mse < tol, mse

    def check_mse_monotonic_nonincreasing(self, mse_dict):
        """
        As the retained-harmonic budget (n or k) grows, MSE should not increase.
        mse_dict: {budget: mse}, ordered by ascending budget.
        """
        budgets = sorted(mse_dict.keys())
        mses = [mse_dict[b] for b in budgets]
        diffs = np.diff(mses)
        is_monotonic = np.all(diffs <= 1e-9)  # small tolerance for float noise
        return is_monotonic, diffs


# =====================================================================
# Entry point — wire everything together once your TODOs are filled in
# =====================================================================
if __name__ == "__main__":
    # Load samples the same way your offline's fs_redrawer.py does
    # (e.g. from svgs/heart.svg -> parametrized t, f_vals). Swap this stub
    # out for however your offline extracts samples.
    M = 500
    T = 1.0
    t = np.linspace(0, T, M, endpoint=False)
    # placeholder ground truth — replace with your actual SVG-derived signal
    f_vals = np.exp(1j * 2 * np.pi * t) + 0.3 * np.exp(1j * 2 * np.pi * 5 * t)

    N = 150
    epi = FourierEpicycles(t, f_vals, N, T)
    epi.coeffs = epi.compute_coeffs()

    pruner = TopKPruner()
    sweep = MSESweep()
    validator = PruningValidator()

    ok, mse0 = validator.check_full_reconstruction_zero_error(epi)
    print(f"[full reconstruction ~0 error] valid={ok} mse={mse0:.2e}")

    k = 20
    ok, count = validator.check_top_k_count_and_zeros(pruner, epi, k)
    print(f"[top-{k} pruning] valid={ok} nonzero_count={count}")

    n_values = [5, 10, 20, 50, 100, 150]
    mse_by_n = sweep.mse_vs_n_sweep(epi, n_values)
    for n in n_values:
        print(f"  N cutoff={n:>3}  MSE={mse_by_n[n]:.6e}")
    ok, diffs = validator.check_mse_monotonic_nonincreasing(mse_by_n)
    print(f"[MSE vs N monotonic non-increasing] valid={ok}")

    k_values = [5, 10, 20, 50, 100, 150]
    mse_by_k = sweep.mse_vs_k_sweep(epi, pruner, k_values)
    for k in k_values:
        print(f"  K={k:>3}  MSE={mse_by_k[k]:.6e}")
    ok, diffs = validator.check_mse_monotonic_nonincreasing(mse_by_k)
    print(f"[MSE vs K monotonic non-increasing] valid={ok}")
