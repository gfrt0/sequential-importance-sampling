import numba
import numpy as np
import math

@numba.njit
def rand_choice_nb( arr, prob ):
    """
    Sample from a vector with weights p. numba does not accept np.random.choice(p = ...)
    """
    return arr[np.searchsorted( np.cumsum(prob), np.random.random(), side = "right" )]

@numba.njit
def count_bounded_compositions( s, bounds, K ):
    """
    Counts the number of integer vectors x with sum s and x[k] <= bounds[k].
    This optimized version uses a 2-loop DP approach with complexity O(K*s).
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
def sample_uniform_column( column_sum, row_sums, K ):
    """
    Uniformly sample integer vector x with sum s and x[i] <= bounds[i],
    using recursive conditional sampling.

    Inputs:
    - s: total sum
    - bounds: upper bounds for each x[i]
    - rng: numba-compatible random number generator

    Outputs:
    - x: integer vector of length len(bounds)
    """
    output = np.zeros( K, dtype = np.int64 )
    remaining_in_column = column_sum

    for k in range(K - 1):
        max_val = min( row_sums[k], remaining_in_column )
        choices = np.arange( max_val + 1 )
        weights = np.zeros( max_val + 1, dtype = np.int64 )

        for val in range( max_val + 1 ):
            weights[val] = count_bounded_compositions( remaining_in_column - val, row_sums[k + 1:], K - ( k + 1 ) )

        probs = weights / np.sum( weights )
        output[k] = rand_choice_nb( choices, probs )
        remaining_in_column -= output[k]

    output[K - 1] = remaining_in_column
    return output


@numba.njit
def sample_table_sis( row_sums, col_sums, K, C ):
    """
    Uniformly sample a single contingency table with fixed margins as in Chen Diaconis Holmes Liu (2005, JASA)
    """
    remaining = row_sums.copy()
    output = np.zeros( (K, C), dtype = np.int64 )
    log_q = 0.0

    for c in range(C - 1):
        col = sample_uniform_column( col_sums[c], remaining, K )
        output[:, c] = col
        log_q += math.log( count_bounded_compositions( col_sums[c], remaining, K ) )
        remaining -= col

    output[:, C - 1] = remaining
    log_q += 0.0
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
