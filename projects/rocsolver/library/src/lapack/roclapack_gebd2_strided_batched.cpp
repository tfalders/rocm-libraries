/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "roclapack_gebd2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename U>
rocblas_status rocsolver_gebd2_strided_batched_impl(rocblas_handle handle,
                                                    const rocblas_int m,
                                                    const rocblas_int n,
                                                    U A,
                                                    const rocblas_int lda,
                                                    const rocblas_stride strideA,
                                                    S* D,
                                                    const rocblas_stride strideD,
                                                    S* E,
                                                    const rocblas_stride strideE,
                                                    T* tauq,
                                                    const rocblas_stride strideQ,
                                                    T* taup,
                                                    const rocblas_stride strideP,
                                                    const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gebd2_strided_batched", "-m", m, "-n", n, "--lda", lda, "--strideA",
                        strideA, "--strideD", strideD, "--strideE", strideE, "--strideQ", strideQ,
                        "--strideP", strideP, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_gebd2_gebrd_argCheck(handle, m, n, lda, A, D, E, tauq, taup, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // memory workspace sizes:
    rocsolver_workspace_helper work_helper;
    rocsolver_gebd2_getMemorySize<false, T>(m, n, batch_count, &work_helper);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, work_helper.get_total_size<T>());

    // memory workspace allocation
    rocblas_device_malloc mem(handle, work_helper.get_total_size<T>());

    if(!mem)
        return rocblas_status_memory_error;

    ROCBLAS_CHECK(work_helper.assign_buffer<T>(handle, mem[0]));

    // execution
    return rocsolver_gebd2_template<T>(handle, m, n, A, shiftA, lda, strideA, D, strideD, E, strideE,
                                       tauq, strideQ, taup, strideP, batch_count, &work_helper);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgebd2_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* D,
                                                const rocblas_stride strideD,
                                                float* E,
                                                const rocblas_stride strideE,
                                                float* tauq,
                                                const rocblas_stride strideQ,
                                                float* taup,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gebd2_strided_batched_impl<float>(handle, m, n, A, lda, strideA, D,
                                                                  strideD, E, strideE, tauq, strideQ,
                                                                  taup, strideP, batch_count);
}

rocblas_status rocsolver_dgebd2_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* D,
                                                const rocblas_stride strideD,
                                                double* E,
                                                const rocblas_stride strideE,
                                                double* tauq,
                                                const rocblas_stride strideQ,
                                                double* taup,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gebd2_strided_batched_impl<double>(handle, m, n, A, lda, strideA, D,
                                                                   strideD, E, strideE, tauq, strideQ,
                                                                   taup, strideP, batch_count);
}

rocblas_status rocsolver_cgebd2_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* D,
                                                const rocblas_stride strideD,
                                                float* E,
                                                const rocblas_stride strideE,
                                                rocblas_float_complex* tauq,
                                                const rocblas_stride strideQ,
                                                rocblas_float_complex* taup,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gebd2_strided_batched_impl<rocblas_float_complex>(
        handle, m, n, A, lda, strideA, D, strideD, E, strideE, tauq, strideQ, taup, strideP,
        batch_count);
}

rocblas_status rocsolver_zgebd2_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* D,
                                                const rocblas_stride strideD,
                                                double* E,
                                                const rocblas_stride strideE,
                                                rocblas_double_complex* tauq,
                                                const rocblas_stride strideQ,
                                                rocblas_double_complex* taup,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gebd2_strided_batched_impl<rocblas_double_complex>(
        handle, m, n, A, lda, strideA, D, strideD, E, strideE, tauq, strideQ, taup, strideP,
        batch_count);
}

} // extern C
