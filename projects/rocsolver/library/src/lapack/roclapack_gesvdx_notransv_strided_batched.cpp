/* **************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_gesvdx_notransv.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    gesvdx_notransv is not intended for inclusion in the public API. It
 *    exists to provide a gesvda method with a signature identical to
 *    the cuSOLVER implementation, for use exclusively in hipSOLVER.
 * ===========================================================================
 */

template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvdx_notransv_strided_batched_impl(rocblas_handle handle,
                                                              const rocblas_svect left_svect,
                                                              const rocblas_svect right_svect,
                                                              const rocblas_srange srange,
                                                              const rocblas_int m,
                                                              const rocblas_int n,
                                                              W A,
                                                              const rocblas_int lda,
                                                              const rocblas_stride strideA,
                                                              const TT vl,
                                                              const TT vu,
                                                              const rocblas_int il,
                                                              const rocblas_int iu,
                                                              rocblas_int* nsv,
                                                              TT* S,
                                                              const rocblas_stride strideS,
                                                              T* U,
                                                              const rocblas_int ldu,
                                                              const rocblas_stride strideU,
                                                              T* V,
                                                              const rocblas_int ldv,
                                                              const rocblas_stride strideV,
                                                              rocblas_int* ifail,
                                                              const rocblas_stride strideF,
                                                              rocblas_int* info,
                                                              const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gesvdx_notransv_strided_batched", "--left_svect", left_svect,
                        "--right_svect", right_svect, "--srange", srange, "-m", m, "-n", n, "--lda",
                        lda, "--strideA", strideA, "--vl", vl, "--vu", vu, "--il", il, "--iu", iu,
                        "--strideS", strideS, "--ldu", ldu, "--strideU", strideU, "--ldv", ldv,
                        "--strideV", strideV, "--strideF", strideF, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gesvdx_notransv_argCheck(handle, left_svect, right_svect, srange,
                                                           m, n, A, lda, vl, vu, il, iu, nsv, S, U,
                                                           ldu, V, ldv, ifail, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // memory workspace sizes:
    rocsolver_workspace_helper work_helper;
    rocsolver_gesvdx_notransv_getMemorySize<false, T, TT>(left_svect, right_svect, srange, m, n, il,
                                                          iu, batch_count, &work_helper);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, work_helper.get_total_size<T>());

    // memory workspace allocation
    rocblas_device_malloc mem(handle, work_helper.get_total_size<T>());

    if(!mem)
        return rocblas_status_memory_error;

    ROCBLAS_CHECK(work_helper.assign_buffer<T>(handle, mem[0]));

    // execution
    return rocsolver_gesvdx_notransv_template<false, true, T>(
        handle, left_svect, right_svect, srange, m, n, A, shiftA, lda, strideA, vl, vu, il, iu, nsv,
        S, strideS, U, ldu, strideU, V, ldv, strideV, ifail, strideF, info, batch_count,
        &work_helper);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status
    rocsolver_sgesvdx_notransv_strided_batched(rocblas_handle handle,
                                               const rocblas_svect left_svect,
                                               const rocblas_svect right_svect,
                                               const rocblas_srange srange,
                                               const rocblas_int m,
                                               const rocblas_int n,
                                               float* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               const float vl,
                                               const float vu,
                                               const rocblas_int il,
                                               const rocblas_int iu,
                                               rocblas_int* nsv,
                                               float* S,
                                               const rocblas_stride strideS,
                                               float* U,
                                               const rocblas_int ldu,
                                               const rocblas_stride strideU,
                                               float* V,
                                               const rocblas_int ldv,
                                               const rocblas_stride strideV,
                                               rocblas_int* ifail,
                                               const rocblas_stride strideF,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesvdx_notransv_strided_batched_impl<float>(
        handle, left_svect, right_svect, srange, m, n, A, lda, strideA, vl, vu, il, iu, nsv, S,
        strideS, U, ldu, strideU, V, ldv, strideV, ifail, strideF, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status
    rocsolver_dgesvdx_notransv_strided_batched(rocblas_handle handle,
                                               const rocblas_svect left_svect,
                                               const rocblas_svect right_svect,
                                               const rocblas_srange srange,
                                               const rocblas_int m,
                                               const rocblas_int n,
                                               double* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               const double vl,
                                               const double vu,
                                               const rocblas_int il,
                                               const rocblas_int iu,
                                               rocblas_int* nsv,
                                               double* S,
                                               const rocblas_stride strideS,
                                               double* U,
                                               const rocblas_int ldu,
                                               const rocblas_stride strideU,
                                               double* V,
                                               const rocblas_int ldv,
                                               const rocblas_stride strideV,
                                               rocblas_int* ifail,
                                               const rocblas_stride strideF,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesvdx_notransv_strided_batched_impl<double>(
        handle, left_svect, right_svect, srange, m, n, A, lda, strideA, vl, vu, il, iu, nsv, S,
        strideS, U, ldu, strideU, V, ldv, strideV, ifail, strideF, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status
    rocsolver_cgesvdx_notransv_strided_batched(rocblas_handle handle,
                                               const rocblas_svect left_svect,
                                               const rocblas_svect right_svect,
                                               const rocblas_srange srange,
                                               const rocblas_int m,
                                               const rocblas_int n,
                                               rocblas_float_complex* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               const float vl,
                                               const float vu,
                                               const rocblas_int il,
                                               const rocblas_int iu,
                                               rocblas_int* nsv,
                                               float* S,
                                               const rocblas_stride strideS,
                                               rocblas_float_complex* U,
                                               const rocblas_int ldu,
                                               const rocblas_stride strideU,
                                               rocblas_float_complex* V,
                                               const rocblas_int ldv,
                                               const rocblas_stride strideV,
                                               rocblas_int* ifail,
                                               const rocblas_stride strideF,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesvdx_notransv_strided_batched_impl<rocblas_float_complex>(
        handle, left_svect, right_svect, srange, m, n, A, lda, strideA, vl, vu, il, iu, nsv, S,
        strideS, U, ldu, strideU, V, ldv, strideV, ifail, strideF, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status
    rocsolver_zgesvdx_notransv_strided_batched(rocblas_handle handle,
                                               const rocblas_svect left_svect,
                                               const rocblas_svect right_svect,
                                               const rocblas_srange srange,
                                               const rocblas_int m,
                                               const rocblas_int n,
                                               rocblas_double_complex* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               const double vl,
                                               const double vu,
                                               const rocblas_int il,
                                               const rocblas_int iu,
                                               rocblas_int* nsv,
                                               double* S,
                                               const rocblas_stride strideS,
                                               rocblas_double_complex* U,
                                               const rocblas_int ldu,
                                               const rocblas_stride strideU,
                                               rocblas_double_complex* V,
                                               const rocblas_int ldv,
                                               const rocblas_stride strideV,
                                               rocblas_int* ifail,
                                               const rocblas_stride strideF,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesvdx_notransv_strided_batched_impl<rocblas_double_complex>(
        handle, left_svect, right_svect, srange, m, n, A, lda, strideA, vl, vu, il, iu, nsv, S,
        strideS, U, ldu, strideU, V, ldv, strideV, ifail, strideF, info, batch_count);
}

} // extern C
