"""Coverage-guided neighborhood search helpers."""

import numpy as np


def _projection_coverage(projection):
    grid_size = projection.shape[0]
    points = np.argwhere(projection)
    if not len(points):
        return 0
    differences = points[:, None, :] - points[None, :, :]
    support = np.zeros((2 * grid_size - 1, 2 * grid_size - 1), dtype=bool)
    support[
        differences[..., 0] + grid_size - 1,
        differences[..., 1] + grid_size - 1,
    ] = True
    return int(np.count_nonzero(support))


def _occupancy_score(occupancy):
    projections = (
        occupancy.any(axis=2),
        occupancy.any(axis=1),
        occupancy.any(axis=0),
    )
    denominator = (2 * occupancy.shape[0] - 1) ** 2
    coverage = sum(_projection_coverage(projection) for projection in projections)
    return coverage / denominator


def satellite_coverage_guidance(candidate, position_maps, grid_size, n_sat):
    """Measure each satellite's score contribution at the weakest observation."""
    candidate = np.asarray(candidate, dtype=np.float64)
    if candidate.shape != (6 * n_sat,) or not np.isfinite(candidate).all():
        raise ValueError(f"Candidate must contain {6 * n_sat} finite values.")
    if grid_size < 2:
        raise ValueError("Grid size must be at least 2.")

    states = candidate.reshape(6, n_sat).T
    observation_scores = []
    observation_occupancies = []
    observation_indices = []
    center = int(grid_size / 2)

    for position_map in np.asarray(position_maps, dtype=np.float64):
        if position_map.shape != (3, 6):
            raise ValueError("Each observation position map must have shape (3, 6).")
        points = np.einsum("nj,ij->ni", states, position_map)
        valid = np.all((points > -1.0) & (points < 1.0), axis=1)
        indices = (points * grid_size / 2 + center).astype(np.int64)
        occupancy_counts = np.zeros((grid_size,) * 3, dtype=np.int16)
        np.add.at(
            occupancy_counts,
            tuple(indices[valid].T),
            1,
        )
        occupancy = occupancy_counts > 0
        observation_occupancies.append(occupancy_counts)
        observation_indices.append((indices, valid))
        observation_scores.append(_occupancy_score(occupancy))

    weakest_observation = int(np.argmin(observation_scores))
    occupancy_counts = observation_occupancies[weakest_observation]
    indices, valid = observation_indices[weakest_observation]
    occupancy = occupancy_counts > 0
    full_score = observation_scores[weakest_observation]
    contributions = np.zeros(n_sat, dtype=np.float64)

    for satellite in np.flatnonzero(valid):
        cell = tuple(indices[satellite])
        occupancy_counts[cell] -= 1
        contributions[satellite] = max(
            0.0, full_score - _occupancy_score(occupancy_counts > 0)
        )
        occupancy_counts[cell] += 1

    positive = contributions[contributions > 0]
    if not len(positive):
        weights = np.full(n_sat, 1.0 / n_sat)
    else:
        floor = max(float(np.median(positive)) * 0.1, 1e-12)
        weights = 1.0 / (contributions + floor)
        weights /= weights.sum()
    return weakest_observation, np.asarray(observation_scores), contributions, weights


def make_coverage_guided_offspring(
    base_candidates,
    count,
    lower,
    upper,
    n_sat,
    mutation_map,
    satellite_weights,
    rng,
    step_multiplier=1.0,
):
    """Perturb low-contribution satellites around incumbent and diverse parents."""
    base_candidates = np.asarray(base_candidates, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    mutation_map = np.asarray(mutation_map, dtype=np.float64)
    satellite_weights = np.asarray(satellite_weights, dtype=np.float64)
    if count < 1 or not np.isfinite(step_multiplier) or step_multiplier <= 0:
        raise ValueError("Count and step multiplier must be positive and finite.")
    if base_candidates.ndim == 1:
        if base_candidates.shape != (6 * n_sat,):
            raise ValueError(
                f"Base candidate must contain {6 * n_sat} values."
            )
        base_candidates = np.broadcast_to(
            base_candidates, (count, 6 * n_sat)
        )
    if base_candidates.shape != (count, 6 * n_sat) or not np.isfinite(
        base_candidates
    ).all():
        raise ValueError(
            f"Base candidates must have shape ({count}, {6 * n_sat}) and be finite."
        )
    if lower.shape != (6 * n_sat,) or upper.shape != (6 * n_sat,):
        raise ValueError("Bounds must match the candidate shape.")
    if mutation_map.shape != (6, 6):
        raise ValueError("Mutation map must have shape (6, 6).")
    if (
        satellite_weights.shape != (n_sat,)
        or not np.isfinite(satellite_weights).all()
        or np.any(satellite_weights < 0)
    ):
        raise ValueError(
            f"Satellite weights must be finite and non-negative with shape ({n_sat},)."
        )
    if not np.isclose(satellite_weights.sum(), 1.0):
        raise ValueError("Satellite weights must sum to 1.")
    greedy_count = max(1, count // 8)
    move_sizes = np.minimum(
        rng.choice(
            np.asarray([1, 2, 4, 8, 16, n_sat]),
            size=count,
            p=[0.28, 0.22, 0.18, 0.14, 0.10, 0.08],
        ),
        n_sat,
    )
    selected = np.zeros((count, n_sat), dtype=bool)
    for row, move_size in enumerate(move_sizes):
        satellites = rng.choice(
            n_sat, size=int(move_size), replace=False, p=satellite_weights
        )
        selected[row, satellites] = True

    genes = base_candidates.reshape(count, 6, n_sat).transpose(0, 2, 1).copy()
    steps = np.exp(rng.uniform(np.log(0.35), np.log(12.0), size=count))
    steps *= step_multiplier
    noise = np.einsum(
        "bnj,ij->bni",
        rng.normal(size=(count, n_sat, 6)),
        mutation_map,
    )
    genes += noise * steps[:, None, None] * selected[..., None]

    reset = (rng.random((count, n_sat)) < 0.08) & selected
    reset_rows, reset_satellites = np.nonzero(reset)
    if len(reset_rows):
        random_values = rng.uniform(
            lower,
            upper,
            size=(len(reset_rows), lower.size),
        ).reshape(-1, 6, n_sat).transpose(0, 2, 1)
        genes[reset_rows, reset_satellites] = random_values[
            np.arange(len(reset_rows)), reset_satellites
        ]

    candidates = genes.transpose(0, 2, 1).reshape(count, 6 * n_sat)
    candidates = np.clip(candidates, lower, upper)

    greedy_rows = np.arange(count - greedy_count, count)
    greedy_satellites = rng.choice(
        n_sat,
        size=greedy_count,
        replace=True,
        p=satellite_weights,
    )
    greedy_axes = rng.integers(0, 6, size=greedy_count)
    greedy_signs = rng.choice(np.asarray([-1.0, 1.0]), size=greedy_count)
    greedy_steps = rng.choice(
        np.asarray([0.5, 1.0, 2.0, 4.0, 8.0, 12.0]),
        size=greedy_count,
    )
    greedy_candidates = base_candidates[greedy_rows].reshape(
        greedy_count, 6, n_sat
    ).copy()
    greedy_candidates[
        np.arange(greedy_count), :, greedy_satellites
    ] += (
        greedy_signs[:, None]
        * greedy_steps[:, None]
        * mutation_map[:, greedy_axes].T
    )
    candidates[greedy_rows] = np.clip(
        greedy_candidates.reshape(greedy_count, 6 * n_sat),
        lower,
        upper,
    )
    return candidates
