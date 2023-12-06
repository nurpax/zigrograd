const std = @import("std");

const ndvec = @import("./vec_ops.zig");

pub fn gemm_nn(M: usize, N: usize, K: usize, alpha: f32, A: []f32, lda: usize, B: []f32, ldb: usize, C: []f32, ldc: usize) void {
    for (0..M) |i| {
        for (0..K) |k| {
            const ap = alpha * A[i * lda + k];
            for (0..N) |j| {
                C[i * ldc + j] += ap * B[k * ldb + j];
            }
        }
    }
}

// Slower reference implementation of the below gemm_nt path.
fn gemm_nt_ref(M: usize, N: usize, K: usize, alpha: f32, A: []f32, lda: usize, B: []f32, ldb: usize, C: []f32, ldc: usize) void {
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += alpha * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

pub fn gemm_nt(M: usize, N: usize, K: usize, alpha: f32, A: []f32, lda: usize, B: []f32, ldb: usize, C: []f32, ldc: usize) void {
    const K_rem = K & 3;
    const K4 = K >> 2;

    for (0..M) |i| {
        for (0..N) |j| {
            var totalv4 = @Vector(4, f32){ 0, 0, 0, 0 };
            var i_lda = i * lda;
            var j_ldb = j * ldb;
            for (0..K4) |_| {
                const av = @Vector(4, f32){ A[i_lda + 0], A[i_lda + 1], A[i_lda + 2], A[i_lda + 3] };
                const bv = @Vector(4, f32){ B[j_ldb + 0], B[j_ldb + 1], B[j_ldb + 2], B[j_ldb + 3] };
                totalv4 += @as(@Vector(4, f32), @splat(alpha)) * av * bv;
                i_lda += 4;
                j_ldb += 4;
            }
            var sum: f32 = @reduce(.Add, totalv4);
            for (0..K_rem) |_| {
                sum += alpha * A[i_lda] * B[j_ldb];
                i_lda += 1;
                j_ldb += 1;
            }
            C[i * ldc + j] += sum;
        }
    }
}

pub fn gemm_tn(M: usize, N: usize, K: usize, alpha: f32, A: []f32, lda: usize, B: []f32, ldb: usize, C: []f32, ldc: usize) void {
    for (0..M) |i| {
        for (0..K) |k| {
            const ap = alpha * A[k * lda + i];
            for (0..N) |j| {
                C[i * ldc + j] += ap * B[k * ldb + j];
            }
        }
    }
}

pub fn gemm_tt(M: usize, N: usize, K: usize, alpha: f32, A: []f32, lda: usize, B: []f32, ldb: usize, C: []f32, ldc: usize) void {
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += alpha * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

pub fn gemm(TA: bool, TB: bool, M: usize, N: usize, K: usize, alpha: f32, A: []f32, lda: usize, B: []f32, ldb: usize, beta: f32, C: []f32, ldc: usize) void {
    if (beta != 1.0) {
        for (0..M) |i| {
            for (0..N) |j| {
                C[i * ldc + j] *= beta;
            }
        }
    } else if (beta == 0) {
        for (0..M) |i| {
            for (0..N) |j| {
                C[i * ldc + j] = 0;
            }
        }
    }

    if (!TA and !TB) {
        gemm_nn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    } else if (TA and !TB) {
        gemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    } else if (!TA and TB) {
        gemm_nt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    } else {
        gemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    }
}
