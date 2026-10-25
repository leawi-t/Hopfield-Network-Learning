import numpy as np

def sgn(x: np.ndarray | float) -> np.ndarray | float:
    """Bipolar activation function ensuring values remain in {-1, +1}."""
    res = np.sign(x)
    if isinstance(res, np.ndarray):
        res[res == 0] = 1
    elif res == 0:
        res = 1
    return res

def generate_patterns(size=50, amount=3) -> list[np.ndarray]:
    return [np.random.choice([-1, 1], size=size) for _ in range(amount)]

def compute_weight(patterns: list[np.ndarray]) -> np.ndarray:
    n = len(patterns[0])
    weight = np.zeros((n, n))
    for pattern in patterns:
        weight += np.outer(pattern, pattern)

    weight /= n
    np.fill_diagonal(weight, 0)
    return weight

def compute_storkey_weights(patterns: list[np.ndarray]) -> np.ndarray:
    n = len(patterns[0])
    w = np.zeros((n, n))

    for xi in patterns:
        # Full local field vector under current weights: shape (N,)
        h = w @ xi

        # Outer products
        xi_xi = np.outer(xi, xi)  # Term 1: ξ_i * ξ_j
        xi_h  = np.outer(xi, h)   # Term 2: ξ_i * h_j
        h_xi  = np.outer(h, xi)   # Term 3: h_i * ξ_j

        # Storkey increment
        delta_w = (1.0 / n) * (xi_xi - xi_h - h_xi + 2.0 * w)
        w = w + delta_w

        # Ensure diagonal stays zero at every pattern addition
        np.fill_diagonal(w, 0)

    return w

def compute_energy(weight: np.ndarray, state: np.ndarray) -> float:
    """Calculates Lyapunov energy E = -0.5 * s^T * W * s."""
    return -0.5 * float(state @ weight @ state)

def generate_probe(pattern: np.ndarray, flip_ratio=0.1) -> np.ndarray:
    n = len(pattern)
    num_flips = max(1, int(n * flip_ratio))
    selected_indices = np.random.choice(n, size=num_flips, replace=False)
    probe = pattern.copy()
    probe[selected_indices] *= -1
    return probe

def async_update_loop(weight: np.ndarray, probe: np.ndarray, max_sweeps=10) -> tuple[np.ndarray, list[float]]:
    """
    Asynchronously updates neurons one by one in random order.
    Tracks energy after every single unit update.
    """
    s = probe.copy()
    n = len(s)
    energy_history = [compute_energy(weight, s)]

    for sweep in range(max_sweeps):
        # Visit units in random order per sweep to avoid update ordering bias
        indices = np.random.permutation(n)
        state_changed = False

        for j in indices:
            # Local field (net input) arriving at unit j
            h_j = np.dot(weight[j], s)
            new_val = sgn(h_j)

            if new_val != s[j]:
                s[j] = new_val
                state_changed = True
                energy_history.append(compute_energy(weight, s))

        # Reached a stable attractor early
        if not state_changed:
            break

    return s, energy_history

def test_one_step_fixed(patterns: list[np.ndarray], weight: np.ndarray) -> bool:
    for pattern in patterns:
        local_fields = np.dot(weight, pattern)
        reconstructed = sgn(local_fields)
        if not np.array_equal(pattern, reconstructed):
            return False
    return True

def test_pathology(weights: np.ndarray, pattern: np.ndarray) -> bool:
    inverse = -pattern
    is_attractor = True

    # 1. One-step fixed point test
    if not np.array_equal(inverse, sgn(weights @ inverse)):
        print("Inverse is not a stable attractor: fails immediate fixed point test.")
        is_attractor = False

    # 2. Energy symmetry test (using np.isclose for float safety)
    pattern_energy = compute_energy(weights, pattern)
    inverse_energy = compute_energy(weights, inverse)
    if not np.isclose(pattern_energy, inverse_energy):
        print("Inverse is not a stable attractor: fails energy parity test.")
        is_attractor = False

    return is_attractor

def compute_sweep(size=100, amounts=(10, 14, 20, 25, 30), num_trials=15):
    hebb_fixed_rate = []
    storkey_fixed_rate = []

    for p in amounts:
        hebb_trial_rates = []
        storkey_trial_rates = []

        for _ in range(num_trials):
            patterns = generate_patterns(size=size, amount=p)
            P = np.stack(patterns)  # Shape: (p, size)

            w_hebb = compute_weight(patterns)
            w_storkey = compute_storkey_weights(patterns)

            # Parallel reconstruction check for all p patterns
            # Since W is symmetric, (P @ W)[k] == W @ patterns[k]
            hebb_recon = sgn(P @ w_hebb)
            storkey_recon = sgn(P @ w_storkey)

            hebb_trial_rates.append(np.mean(np.all(P == hebb_recon, axis=1)))
            storkey_trial_rates.append(np.mean(np.all(P == storkey_recon, axis=1)))

        hebb_fixed_rate.append(float(np.mean(hebb_trial_rates)))
        storkey_fixed_rate.append(float(np.mean(storkey_trial_rates)))

    return hebb_fixed_rate, storkey_fixed_rate

def main():
    np.random.seed(42)  # For reproducible verification
    patterns = generate_patterns(size=50, amount=3)
    hebb_weights = compute_weight(patterns)
    storkey_weights = compute_storkey_weights(patterns)

    print(f"Diagonals are zero: {np.all(np.diag(hebb_weights) == 0)}")
    print(f"Weight is Symmetric: {np.allclose(hebb_weights, hebb_weights.T)}")
    print(f"Has immediate fixed point: {test_one_step_fixed(patterns, hebb_weights)}")

    print(f"Storkey Diagonals are zero: {np.all(np.diag(storkey_weights) == 0)}")
    print(f"Storkey Weight is Symmetric: {np.allclose(storkey_weights, storkey_weights.T)}")
    print(f"Storkey Has immediate fixed point: {test_one_step_fixed(patterns, storkey_weights)}")

    # Test retrieval from corrupted pattern
    target_pattern = patterns[0]
    corrupted_probe = generate_probe(target_pattern, flip_ratio=0.2)
    hamming_dist_start = np.sum(target_pattern != corrupted_probe)
    print(f"Initial corrupted bits: {hamming_dist_start}/50")

    recovered_state, energy_history = async_update_loop(hebb_weights, corrupted_probe)
    hamming_dist_end = np.sum(target_pattern != recovered_state)
    print(f"Bits mismatched after relaxation: {hamming_dist_end}/50")

    print("\nEnergy descent history:")
    for step, e in enumerate(energy_history):
        print(f"  Step {step:02d}: E = {e:.4f}")

    # Monotonicity test
    energy_diffs = np.diff(energy_history)
    print(f"\nEnergy strictly non-increasing: {np.all(energy_diffs <= 1e-9)}")

    print("\n*************** Hebbian vs Storkey learning *************** \n")
    print("Testing memory capacity")

    large_patterns = generate_patterns(50, 10)
    large_hebb_weights = compute_weight(large_patterns)
    large_storkey_weights = compute_storkey_weights(large_patterns)

    # Testing one step fixed:
    print(f"Hebbian has immediate fixed point: {test_one_step_fixed(large_patterns, large_hebb_weights)}")
    print(f"Storkey has immediate fixed point: {test_one_step_fixed(large_patterns, large_storkey_weights)}")

    print("\n============ Pathology Check ============\n")
    print(f"Is Hebbian inverse a stable attractor: {test_pathology(hebb_weights, patterns[0])}")
    print(f"Is storkey inverse a stable attractor: {test_pathology(storkey_weights, patterns[0])}")

    print("\n============ Sweep Results ============\n")
    amounts = [10,14,20,25,30]
    hebb_results, storkey_results = compute_sweep()

    print("Amount \t Hebbian Fixed Rate \t Storkey Fixed Rate")
    for i, amount in enumerate(amounts):
        print(f"{amount} \t {hebb_results[i]:.4f} \t {storkey_results[i]:.4f}")

if __name__ == "__main__":
    main()