import numba
import numpy as np
import math

@numba.njit
def count_bounded_compositions( s, bounds, K ):
    """
    Counts the number of integer vectors x with sum s and x[k] <= bounds[k].
    Uses a 2-loop DP approach with complexity O(K*s).
    """
    dp = np.zeros((K + 1, s + 1), dtype=np.int64)
    dp[0, 0] = 1

    for k in range(1, K + 1):
        bound_k = bounds[k - 1]
        cumsum_prev_row = np.cumsum(dp[k - 1, :])
        for j in range(s + 1):
            upper_sum = cumsum_prev_row[j]
            lower_sum_idx = j - bound_k - 1
            lower_sum = cumsum_prev_row[lower_sum_idx] if lower_sum_idx >= 0 else 0
            dp[k, j] = upper_sum - lower_sum
    return dp[K, s]


@numba.njit
def _precompute_suffix_counts( column_sum, row_sums, K ):
    """
    Precompute suffix_counts[k, s] = number of compositions of s into
    (x[k], ..., x[K-1]) with x[i] <= row_sums[i].

    Built bottom-up from k = K-1 down to k = 0.  O(K * column_sum).
    """
    S = column_sum
    suffix = np.zeros((K + 1, S + 1), dtype=np.int64)
    suffix[K, 0] = 1

    for k in range(K - 1, -1, -1):
        bound_k = row_sums[k]
        cumsum_next = np.cumsum(suffix[k + 1, :])
        for s in range(S + 1):
            upper = cumsum_next[s]
            lower_idx = s - bound_k - 1
            lower = cumsum_next[lower_idx] if lower_idx >= 0 else 0
            suffix[k, s] = upper - lower

    return suffix


@numba.njit
def sample_uniform_column( column_sum, row_sums, K ):
    """
    Uniformly sample integer vector x with sum column_sum and x[i] <= row_sums[i],
    using suffix DP precomputation for O(K * column_sum) total work.

    Returns (x, total_count) where total_count = number of valid compositions.
    """
    suffix = _precompute_suffix_counts(column_sum, row_sums, K)
    total_count = suffix[0, column_sum]

    output = np.zeros( K, dtype = np.int64 )
    remaining = column_sum

    for k in range(K - 1):
        max_val = min( row_sums[k], remaining )

        # Sample from unnormalized weights via linear scan.
        # weight(val) = suffix[k+1, remaining - val]; sum = suffix[k, remaining].
        target = np.random.random() * float(suffix[k, remaining])
        cumulative = 0.0
        chosen = max_val
        for val in range(max_val + 1):
            cumulative += float(suffix[k + 1, remaining - val])
            if cumulative > target:
                chosen = val
                break

        output[k] = chosen
        remaining -= chosen

    output[K - 1] = remaining
    return output, total_count


@numba.njit
def sample_table_sis( row_sums, col_sums, K, C ):
    """
    Uniformly sample a single contingency table with fixed margins as in Chen Diaconis Holmes Liu (2005, JASA)
    """
    remaining = row_sums.copy()
    output = np.zeros( (K, C), dtype = np.int64 )
    log_q = 0.0

    for c in range(C - 1):
        col, total_count = sample_uniform_column( col_sums[c], remaining, K )
        output[:, c] = col
        log_q += math.log( total_count )
        remaining -= col

    output[:, C - 1] = remaining
    return output, -log_q


@numba.njit
def _sample_tables_core(row_sums, col_sums, num_samples, rng_seed):
    """
    Sequential inner loop over (num_samples, n_batch).

    row_sums: (n_batch, R) — each row is one batch element's row margins
    col_sums: (n_batch, C) — each row is one batch element's column margins
    """
    n_batch = row_sums.shape[0]
    R = row_sums.shape[1]
    C = col_sums.shape[1]

    tables = np.zeros((num_samples, n_batch, R, C), dtype=np.int64)
    logq = np.zeros((num_samples, n_batch))

    for s in range(num_samples):
        for b in range(n_batch):
            if rng_seed is not None:
                np.random.seed(rng_seed + s * 104729 + b * 131)
            table, lnq = sample_table_sis(row_sums[b].copy(), col_sums[b].copy(), R, C)
            tables[s, b] = table
            logq[s, b] = lnq

    return tables, logq


@numba.njit(parallel=True)
def _sample_tables_core_parallel(row_sums, col_sums, num_samples, rng_seed):
    """
    Parallel inner loop: prange over num_samples, sequential over n_batch.

    row_sums: (n_batch, R) — each row is one batch element's row margins
    col_sums: (n_batch, C) — each row is one batch element's column margins
    """
    n_batch = row_sums.shape[0]
    R = row_sums.shape[1]
    C = col_sums.shape[1]

    tables = np.zeros((num_samples, n_batch, R, C), dtype=np.int64)
    logq = np.zeros((num_samples, n_batch))

    for s in numba.prange(num_samples):
        for b in range(n_batch):
            if rng_seed is not None:
                np.random.seed(rng_seed + s * 104729 + b * 131)
            table, lnq = sample_table_sis(row_sums[b].copy(), col_sums[b].copy(), R, C)
            tables[s, b] = table
            logq[s, b] = lnq

    return tables, logq


def sample_tables(row_sums, col_sums, num_samples, rng_seed=None, parallel=True):
    """
    Sample contingency tables with fixed margins via sequential importance
    sampling, following Chen, Diaconis, Holmes, and Liu (2005, JASA).

    Inputs:
        row_sums : array of shape (R, *batch) — row margins
        col_sums : array of shape (C, *batch) — column margins
            Batch dimensions (if any) must match between row_sums and col_sums.
        num_samples : int — number of tables to draw
        rng_seed    : optional int — seed for reproducibility
        parallel    : bool — use numba parallel sampling (default True)

    Outputs:
        tables : int64 array of shape (num_samples, *batch, R, C)
        logq   : float64 array of shape (num_samples, *batch)
            Log importance weights. To estimate the number of tables with
            the given margins, compute np.exp(-logq).mean().
    """
    row_sums = np.asarray(row_sums, dtype=np.int64)
    col_sums = np.asarray(col_sums, dtype=np.int64)

    R = row_sums.shape[0]
    C = col_sums.shape[0]
    batch_shape = row_sums.shape[1:]

    assert col_sums.shape[1:] == batch_shape, (
        f"Batch shapes must match: row_sums {row_sums.shape[1:]} vs col_sums {col_sums.shape[1:]}"
    )

    n_batch = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1

    # Flatten batch dims and transpose so each batch element's margins are a
    # contiguous row: (R, n_batch) -> (n_batch, R), same for col_sums.
    row_flat = np.ascontiguousarray(row_sums.reshape(R, n_batch).T)
    col_flat = np.ascontiguousarray(col_sums.reshape(C, n_batch).T)

    if parallel:
        tables_flat, logq_flat = _sample_tables_core_parallel(
            row_flat, col_flat, num_samples, rng_seed
        )
    else:
        tables_flat, logq_flat = _sample_tables_core(
            row_flat, col_flat, num_samples, rng_seed
        )

    tables = tables_flat.reshape((num_samples,) + batch_shape + (R, C))
    logq = logq_flat.reshape((num_samples,) + batch_shape)

    return tables, logq
