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
def sample_tables_sis( r_kjt, M_jlt, J, K, L, T, num_samples, rng_seed = None ):
    """
    Uniformly sample contingency tables with fixed margins as in Chen Diaconis Holmes Liu (2005, JASA)

    r_kjt should be (K + 1 x T x J), M_jlt should be (T x L x J)
    """

    tables = np.zeros( (num_samples, K+1, L, T, J), dtype = np.int64 )
    logq = np.zeros( (num_samples, T, J) )

    for s in range( num_samples ):
        for j in range( J ):
            for t in range( T ):
                if rng_seed is not None:
                    np.random.seed(rng_seed + s*104729 + j*131 + t)
                table, lnq = sample_table_sis( r_kjt[:, t, j].copy(), M_jlt[t, :, j].copy(), K+1, L )
                tables[s, :, :, t, j] = table
                logq[s, t, j] = lnq

    return tables, logq

@numba.njit(parallel = True)
def sample_tables_sis_parallel_singleperiod( r_kj, M_jl, J, K, L, num_samples, rng_seed = None ):
    """
    Uniformly sample contingency tables with fixed margins as in Chen Diaconis Holmes Liu (2005, JASA), in parallel.

    r_kjt should be (K + 1 x J), M_jlt should be (L x J)

    Outputs:
    - tables ( num_samples x K + 1 x L x J )
    - logq   ( num_samples x J )
    """

    tables = np.zeros( (num_samples, K+1, L, J), dtype = np.int64 )
    logq   = np.zeros( (num_samples, J) )

    for s in numba.prange(num_samples):
        for j in range(J):
            if rng_seed is not None:
                np.random.seed(rng_seed + s*104729 + j*131)
            table, lnq = sample_table_sis( r_kj[:, j].copy(), M_jl[:, j].copy(), K+1, L )
            tables[s, :, :, j] = table
            logq[s, j] = lnq

    return tables, logq

@numba.njit(parallel = True)
def sample_tables_sis_parallel( r_kjt, M_jlt, J, K, L, T, num_samples, rng_seed = None ):
    """
    Uniformly sample contingency tables with fixed margins as in Chen Diaconis Holmes Liu (2005, JASA), in parallel.
    Sorting rows and columns (as suggested by the authors) did not appear to provide a substantial improvement in testing, so I skip that here.

    r_kjt should be (K + 1 x T x J), M_jlt should be (T x L x J)

    Outputs:
    - tables ( num_samples x K + 1 x L x T x J )
    - logq   ( num_samples x T x J )
    """

    tables = np.zeros( (num_samples, K+1, L, T, J), dtype = np.int64 )
    logq   = np.zeros( (num_samples, T, J) )

    for s in numba.prange(num_samples):
        for t in range(T):
            for j in range(J):
                if rng_seed is not None:
                    np.random.seed(rng_seed + s*104729 + j*131 + t)
                table, lnq = sample_table_sis( r_kjt[:, t, j].copy(), M_jlt[t, :, j].copy(), K+1, L )
                tables[s, :, :, t, j] = table
                logq[s, t, j] = lnq

    return tables, logq
