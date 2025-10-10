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

#include "rocauxiliary_bdsvdx.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_bdsvdx_impl(rocblas_handle handle,
                                     const rocblas_fill uplo,
                                     const rocblas_svect svect,
                                     const rocblas_srange srange,
                                     const rocblas_int n,
                                     T* D,
                                     T* E,
                                     const T vl,
                                     const T vu,
                                     const rocblas_int il,
                                     const rocblas_int iu,
                                     rocblas_int* nsv,
                                     T* S,
                                     T* Z,
                                     const rocblas_int ldz,
                                     rocblas_int* ifail,
                                     rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("bdsvdx", "--uplo", uplo, "--svect", svect, "--srange", srange, "-n", n,
                        "--vl", vl, "--vu", vu, "--il", il, "--iu", iu, "--ldz", ldz);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_bdsvdx_argCheck(handle, uplo, svect, srange, n, D, E, vl, vu, il,
                                                  iu, nsv, S, Z, ldz, ifail, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftZ = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideS = 0;
    rocblas_stride strideZ = 0;
    rocblas_stride strideIfail = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    rocsolver_workspace_helper work_helper;
    rocsolver_bdsvdx_getMemorySize<T>(n, batch_count, &work_helper);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, work_helper.get_total_size<T>());

    // memory workspace allocation
    rocblas_device_malloc mem(handle, work_helper.get_total_size<T>());
    if(!mem)
        return rocblas_status_memory_error;

    ROCBLAS_CHECK(work_helper.assign_buffer<T>(handle, mem[0]));

    // execution
    return rocsolver_bdsvdx_template<T>(handle, uplo, svect, srange, n, D, strideD, E, strideE, vl,
                                        vu, il, iu, nsv, S, strideS, Z, shiftZ, ldz, strideZ, ifail,
                                        strideIfail, info, batch_count, &work_helper);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sbdsvdx(rocblas_handle handle,
                                 const rocblas_fill uplo,
                                 const rocblas_svect svect,
                                 const rocblas_srange srange,
                                 const rocblas_int n,
                                 float* D,
                                 float* E,
                                 const float vl,
                                 const float vu,
                                 const rocblas_int il,
                                 const rocblas_int iu,
                                 rocblas_int* nsv,
                                 float* S,
                                 float* Z,
                                 const rocblas_int ldz,
                                 rocblas_int* ifail,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_bdsvdx_impl<float>(handle, uplo, svect, srange, n, D, E, vl, vu, il,
                                                   iu, nsv, S, Z, ldz, ifail, info);
}

rocblas_status rocsolver_dbdsvdx(rocblas_handle handle,
                                 const rocblas_fill uplo,
                                 const rocblas_svect svect,
                                 const rocblas_srange srange,
                                 const rocblas_int n,
                                 double* D,
                                 double* E,
                                 const double vl,
                                 const double vu,
                                 const rocblas_int il,
                                 const rocblas_int iu,
                                 rocblas_int* nsv,
                                 double* S,
                                 double* Z,
                                 const rocblas_int ldz,
                                 rocblas_int* ifail,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_bdsvdx_impl<double>(handle, uplo, svect, srange, n, D, E, vl, vu,
                                                    il, iu, nsv, S, Z, ldz, ifail, info);
}

} // extern C
