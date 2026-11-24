"""Helpers for retaining and restarting from diverse high-quality candidates."""

import numpy as np


def candidate_distance(left, right):
    """Return normalized Euclidean distance for two canonical candidates."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("Candidates must have matching shapes.")
    if left.ndim != 1:
        raise ValueError("Candidates must be one-dimensional vectors.")
    return float(np.linalg.norm(left - right) / np.sqrt(left.size))


def update_diverse_archive(
    archive_x,
    archive_fitness,
    archive_secondary,
    candidate,
    fitness,
    secondary,
    *,
    max_size=12,
    minimum_distance=0.05,
):
    """Retain strong candidates that occupy meaningfully different basins."""
    candidate = np.asarray(candidate, dtype=np.float64)
    fitness = float(fitness)
    secondary = float(secondary)

    if archive_x is None:
        return (
            candidate[None, :].copy(),
            np.asarray([fitness], dtype=np.float64),
            np.asarray([secondary], dtype=np.float64),
        )

    archive_x = np.asarray(archive_x, dtype=np.float64)
    archive_fitness = np.asarray(archive_fitness, dtype=np.float64)
    archive_secondary = np.asarray(archive_secondary, dtype=np.float64)

    if candidate.shape != archive_x.shape[1:]:
        raise ValueError("Archive candidates and candidate have incompatible shapes.")

    distances = np.asarray([
        candidate_distance(candidate, existing)
        for existing in archive_x
    ])
    closest = int(np.argmin(distances)) if len(distances) else -1

    if len(distances) and distances[closest] < minimum_distance:
        if (
            fitness < archive_fitness[closest]
            or (
                fitness == archive_fitness[closest]
                and secondary > archive_secondary[closest]
            )
        ):
            archive_x[closest] = candidate
            archive_fitness[closest] = fitness
            archive_secondary[closest] = secondary
        return archive_x.copy(), archive_fitness.copy(), archive_secondary.copy()

    archive_x = np.concatenate((archive_x, candidate[None, :]), axis=0)
    archive_fitness = np.concatenate((archive_fitness, [fitness]))
    archive_secondary = np.concatenate((archive_secondary, [secondary]))

    if len(archive_fitness) > max_size:
        fitness_order = np.argsort(archive_fitness)
        selected = []
        for index in fitness_order:
            if not selected:
                selected.append(int(index))
                continue
            if all(
                candidate_distance(archive_x[index], archive_x[other]) >= minimum_distance
                for other in selected
            ):
                selected.append(int(index))
            if len(selected) >= max_size:
                break
        if len(selected) < max_size:
            for index in fitness_order:
                index = int(index)
                if index not in selected:
                    selected.append(index)
                if len(selected) >= max_size:
                    break

        selected = np.asarray(selected, dtype=np.int64)
        archive_x = archive_x[selected]
        archive_fitness = archive_fitness[selected]
        archive_secondary = archive_secondary[selected]

    return archive_x.copy(), archive_fitness.copy(), archive_secondary.copy()


def select_archive_seed(archive_x, archive_fitness, archive_secondary, rng):
    """Choose a diverse archive member for a basin-hopping restart."""
    archive_x = np.asarray(archive_x, dtype=np.float64)
    archive_fitness = np.asarray(archive_fitness, dtype=np.float64)
    archive_secondary = np.asarray(archive_secondary, dtype=np.float64)

    if len(archive_x) == 0:
        raise ValueError("Archive cannot be empty.")

    rank_order = np.lexsort((-archive_secondary, archive_fitness))
    pool = rank_order[: max(1, len(rank_order) // 2)]
    weights = np.linspace(1.0, 0.25, len(pool))
    weights /= weights.sum()
    index = int(rng.choice(pool, p=weights))
    return (
        archive_x[index].copy(),
        float(archive_fitness[index]),
        float(archive_secondary[index]),
    )
