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

#include "rocauxiliary_larft.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_larft_impl(rocblas_handle handle,
                                    const rocblas_direct direct,
                                    const rocblas_storev storev,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    T* V,
                                    const rocblas_int ldv,
                                    T* tau,
                                    T* F,
                                    const rocblas_int ldf)
{
    ROCSOLVER_ENTER_TOP("larft", "--direct", direct, "--storev", storev, "-n", n, "-k", k, "--ldv",
                        ldv, "--ldt", ldf);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_larft_argCheck(handle, direct, storev, n, k, ldv, ldf, V, tau, F);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftV = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride stridev = 0;
    rocblas_stride stridet = 0;
    rocblas_stride stridef = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    rocsolver_workspace_helper work_helper;
    rocsolver_larft_getMemorySize<false, T>(n, k, batch_count, &work_helper);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, work_helper.get_total_size<T>());

    // memory workspace allocation
    rocblas_device_malloc mem(handle, work_helper.get_total_size<T>());
    if(!mem)
        return rocblas_status_memory_error;

    ROCBLAS_CHECK(work_helper.assign_buffer<T>(handle, mem[0]));

    // execution
    return rocsolver_larft_template<T>(handle, direct, storev, n, k, V, shiftV, ldv, stridev, tau,
                                       stridet, F, ldf, stridef, batch_count, &work_helper);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slarft(rocblas_handle handle,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* V,
                                const rocblas_int ldv,
                                float* tau,
                                float* T,
                                const rocblas_int ldt)
{
    return rocsolver::rocsolver_larft_impl<float>(handle, direct, storev, n, k, V, ldv, tau, T, ldt);
}

rocblas_status rocsolver_dlarft(rocblas_handle handle,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* V,
                                const rocblas_int ldv,
                                double* tau,
                                double* T,
                                const rocblas_int ldt)
{
    return rocsolver::rocsolver_larft_impl<double>(handle, direct, storev, n, k, V, ldv, tau, T, ldt);
}

rocblas_status rocsolver_clarft(rocblas_handle handle,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* V,
                                const rocblas_int ldv,
                                rocblas_float_complex* tau,
                                rocblas_float_complex* T,
                                const rocblas_int ldt)
{
    return rocsolver::rocsolver_larft_impl<rocblas_float_complex>(handle, direct, storev, n, k, V,
                                                                  ldv, tau, T, ldt);
}

rocblas_status rocsolver_zlarft(rocblas_handle handle,
                                const rocblas_direct direct,
                                const rocblas_storev storev,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* V,
                                const rocblas_int ldv,
                                rocblas_double_complex* tau,
                                rocblas_double_complex* T,
                                const rocblas_int ldt)
{
    return rocsolver::rocsolver_larft_impl<rocblas_double_complex>(handle, direct, storev, n, k, V,
                                                                   ldv, tau, T, ldt);
}

} // extern C
