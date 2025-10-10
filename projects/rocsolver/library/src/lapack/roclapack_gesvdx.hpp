/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "auxiliary/rocauxiliary_bdsvdx.hpp"
#include "auxiliary/rocauxiliary_ormbr_unmbr.hpp"
#include "auxiliary/rocauxiliary_ormlq_unmlq.hpp"
#include "auxiliary/rocauxiliary_ormqr_unmqr.hpp"
#include "rocblas.hpp"
#include "roclapack_gebrd.hpp"
#include "roclapack_gelqf.hpp"
#include "roclapack_geqrf.hpp"
#include "roclapack_gesvd.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_workspace_helper.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** Argument checking **/
template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvdx_argCheck(rocblas_handle handle,
                                         const rocblas_svect left_svect,
                                         const rocblas_svect right_svect,
                                         const rocblas_srange srange,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         W A,
                                         const rocblas_int lda,
                                         const TT vl,
                                         const TT vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         rocblas_int* nsv,
                                         TT* S,
                                         T* U,
                                         const rocblas_int ldu,
                                         T* V,
                                         const rocblas_int ldv,
                                         rocblas_int* ifail,
                                         rocblas_int* info,
                                         const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((left_svect != rocblas_svect_singular && left_svect != rocblas_svect_none)
       || (right_svect != rocblas_svect_singular && right_svect != rocblas_svect_none))
        return rocblas_status_invalid_value;
    if(srange != rocblas_srange_all && srange != rocblas_srange_value
       && srange != rocblas_srange_index)
        return rocblas_status_invalid_value;

    // 2. invalid size
    const rocblas_int nsv_max = (srange == rocblas_srange_index ? iu - il + 1 : std::min(m, n));
    if(n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if(left_svect == rocblas_svect_singular && ldu < m)
        return rocblas_status_invalid_size;
    if(right_svect == rocblas_svect_singular && ldv < nsv_max)
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_value && (vl < 0 || vl >= vu))
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_index && (il < 1 || iu < 0))
        return rocblas_status_invalid_size;
    if(srange == rocblas_srange_index && (iu > std::min(m, n) || (std::min(m, n) > 0 && il > iu)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && m && !A) || (nsv_max && !S) || (batch_count && !info) || (batch_count && !nsv))
        return rocblas_status_invalid_pointer;
    if((left_svect == rocblas_svect_singular || right_svect == rocblas_svect_singular)
       && std::min(m, n) && !ifail)
        return rocblas_status_invalid_pointer;
    if(left_svect == rocblas_svect_singular && m && nsv_max && !U)
        return rocblas_status_invalid_pointer;
    if(right_svect == rocblas_svect_singular && nsv_max && n && !V)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_gesvdx_getMemorySize(const rocblas_svect left_svect,
                                    const rocblas_svect right_svect,
                                    const rocblas_srange srange,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int il,
                                    const rocblas_int iu,
                                    const rocblas_int bc,
                                    rocsolver_workspace_helper* work_helper)
{
    // if quick return, set workspace to zero
    if(n == 0 || m == 0 || bc == 0)
        return;

    work_helper->set_nested_capacity(7);
    auto bdsvdx_work = work_helper->add_nested();
    auto gebrd_work = work_helper->add_nested();
    auto gexxf_work = work_helper->add_nested();
    auto ormqr_work = work_helper->add_nested();
    auto ormlq_work = work_helper->add_nested();
    auto ormbr1_work = work_helper->add_nested();
    auto ormbr2_work = work_helper->add_nested();

    const bool row = (m >= n);
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
    const rocblas_int k = std::min(m, n);
    const rocblas_int nsv_max = (srange == rocblas_srange_index ? iu - il + 1 : k);

    // general requirements for bdsvdx and gebrd
    size_t size_tau = 0;
    size_t size_tauqp = 2 * k * sizeof(T) * bc;
    size_t size_tmpDE = 2 * k * sizeof(S) * bc;
    size_t size_tmpZ = 2 * k * nsv_max * sizeof(S) * bc;
    size_t size_tmpT = 0;
    rocsolver_bdsvdx_getMemorySize<S>(k, bc, bdsvdx_work);

    if(thinSVD)
    {
        // requirements for column/row compression
        size_tau = k * sizeof(T) * bc;
        size_tmpT = k * k * sizeof(T) * bc;
        if(row)
            rocsolver_geqrf_getMemorySize<BATCHED, T>(m, n, bc, gexxf_work);
        else
            rocsolver_gelqf_getMemorySize<BATCHED, T>(m, n, bc, gexxf_work);

        // extra requirements for gebrd
        rocsolver_gebrd_getMemorySize<false, T>(k, k, bc, gebrd_work);
    }
    else
    {
        // extra requirements for gebrd
        rocsolver_gebrd_getMemorySize<BATCHED, T>(m, n, bc, gebrd_work);
    }

    if(leftvS)
    {
        if(thinSVD)
        {
            // requirements for ormqr
            if(row)
                rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(
                    rocblas_side_left, rocblas_operation_none, m, nsv_max, k, bc, ormqr_work);

            // requirements for ormbr
            rocsolver_ormbr_unmbr_getMemorySize<false, T>(rocblas_column_wise, rocblas_side_left,
                                                          rocblas_operation_none, k, nsv_max, k, bc,
                                                          ormbr1_work);
        }
        else
        {
            // requirements for ormbr
            rocblas_int mm = row ? m : k;
            rocblas_int kk = row ? k : n;
            rocsolver_ormbr_unmbr_getMemorySize<BATCHED, T>(rocblas_column_wise, rocblas_side_left,
                                                            rocblas_operation_none, mm, nsv_max, kk,
                                                            bc, ormbr1_work);
        }
    }

    if(rightvS)
    {
        if(thinSVD)
        {
            // requirements for ormlq
            if(!row)
                rocsolver_ormlq_unmlq_getMemorySize<BATCHED, T>(
                    rocblas_side_right, rocblas_operation_none, nsv_max, n, k, bc, ormlq_work);

            // requirements for ormbr
            rocsolver_ormbr_unmbr_getMemorySize<false, T>(rocblas_row_wise, rocblas_side_right,
                                                          rocblas_operation_transpose, nsv_max, k,
                                                          k, bc, ormbr2_work);
        }
        else
        {
            // requirements for ormbr
            rocblas_int nn = row ? k : n;
            rocblas_int kk = row ? m : k;
            rocsolver_ormbr_unmbr_getMemorySize<BATCHED, T>(rocblas_row_wise, rocblas_side_right,
                                                            rocblas_operation_transpose, nsv_max,
                                                            nn, kk, bc, ormbr2_work);
        }
    }

    work_helper->assign_sizes({size_tau, size_tauqp, size_tmpDE, size_tmpZ, size_tmpT});
}

template <bool BATCHED, bool STRIDED, typename T, typename TT, typename W>
rocblas_status rocsolver_gesvdx_template(rocblas_handle handle,
                                         const rocblas_svect left_svect,
                                         const rocblas_svect right_svect,
                                         const rocblas_srange srange,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         W A,
                                         const rocblas_stride shiftA,
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
                                         const rocblas_int batch_count,
                                         rocsolver_workspace_helper* work_helper)
{
    ROCSOLVER_ENTER("gesvdx", "leftsv:", left_svect, "rightsv:", right_svect, "srange:", srange,
                    "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "vl:", vl, "vu:", vu,
                    "il:", il, "iu:", iu, "ldu:", ldu, "ldv:", ldv, "bc:", batch_count);

    // quick return (no batch items)
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // set info = 0 and nsv = 0
    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(blocksReset, 1, 1), dim3(BS1, 1, 1), 0, stream, info,
                            batch_count, 0);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(blocksReset, 1, 1), dim3(BS1, 1, 1), 0, stream, nsv,
                            batch_count, 0);

    // quick return (no dimensions)
    if(n == 0 || m == 0)
        return rocblas_status_success;

    // prepare workspace
    auto bdsvdx_work = work_helper->get_nested(0);
    auto gebrd_work = work_helper->get_nested(1);
    auto gexxf_work = work_helper->get_nested(2);
    auto ormqr_work = work_helper->get_nested(3);
    auto ormlq_work = work_helper->get_nested(4);
    auto ormbr1_work = work_helper->get_nested(5);
    auto ormbr2_work = work_helper->get_nested(6);
    T* tau = (T*)(*work_helper)[0];
    T* tauqp = (T*)(*work_helper)[1];
    TT* tmpDE = (TT*)(*work_helper)[2];
    TT* tmpZ = (TT*)(*work_helper)[3];
    T* tmpT = (T*)(*work_helper)[4];

    // booleans used to determine the path that the execution will follow:
    const bool row = (m >= n);
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);

    // auxiliary sizes and variables
    rocblas_fill uplo = row ? rocblas_fill_upper : rocblas_fill_lower;
    const rocblas_svect svect = (leftvS || rightvS) ? rocblas_svect_singular : rocblas_svect_none;
    const rocblas_int thread_count = BS2;
    const rocblas_int k = std::min(m, n);
    const rocblas_int nsv_max = (srange == rocblas_srange_index ? iu - il + 1 : k);
    const rocblas_int ldt = k;
    const rocblas_int ldx = thinSVD ? k : m;
    const rocblas_int ldy = thinSVD ? k : n;
    const rocblas_int ldz = 2 * k;
    const rocblas_stride strideD = k;
    const rocblas_stride strideE = k;
    const rocblas_stride strideT = k * k;
    const rocblas_stride strideX = ldx * GEBRD_GEBD2_SWITCHSIZE;
    const rocblas_stride strideY = ldy * GEBRD_GEBD2_SWITCHSIZE;
    const rocblas_stride strideZ = 2 * k * nsv_max;
    T* UV;
    rocblas_stride strideUV;
    rocblas_int lduv;
    rocblas_int offZ;
    bool trans;

    // common block sizes and number of threads for internal kernels
    const rocblas_int blocks_m = (m - 1) / thread_count + 1;
    const rocblas_int blocks_n = (n - 1) / thread_count + 1;
    const rocblas_int blocks_k = (k - 1) / thread_count + 1;
    const rocblas_int blocks_nsv = (nsv_max - 1) / thread_count + 1;

    /***** 1. bidiagonalization *****/
    /********************************/
    if(thinSVD)
    {
        // apply qr/lq factorization
        local_geqrlq_template<BATCHED>(handle, m, n, A, shiftA, lda, strideA, tau, k, batch_count,
                                       gexxf_work, row);

        // copy triangular factor
        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream, k, k, A, shiftA,
                                lda, strideA, tmpT, 0, ldt, strideT, no_mask{}, uplo);

        // clean triangular factor
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream, k, k, tmpT, 0, ldt,
                                strideT, uplo);

        // apply gebrd to triangular factor
        rocsolver_gebrd_template<false>(handle, k, k, tmpT, 0, ldt, strideT, tmpDE, strideD,
                                        (tmpDE + k * batch_count), strideE, tauqp, k,
                                        (tauqp + k * batch_count), k, batch_count, gebrd_work);
    }
    else
    {
        // apply gebrd to matrix A
        rocsolver_gebrd_template<BATCHED>(handle, m, n, A, shiftA, lda, strideA, tmpDE, strideD,
                                          (tmpDE + k * batch_count), strideE, tauqp, k,
                                          (tauqp + k * batch_count), k, batch_count, gebrd_work);
    }

    /***** 2. solve bidiagonal problem *****/
    /***************************************/
    // compute SVD of bidiagonal matrix
    uplo = thinSVD ? rocblas_fill_upper : uplo;
    rocsolver_bdsvdx_template(handle, uplo, svect, srange, k, tmpDE, strideD,
                              (tmpDE + k * batch_count), strideE, vl, vu, il, iu, nsv, S, strideS,
                              tmpZ, 0, ldz, strideZ, ifail, strideF, info, batch_count, bdsvdx_work);

    /***** 3. compute/update left vectors *****/
    /******************************************/
    if(leftvS)
    {
        // For now we work with the nsv_max columns in tmpZ, i.e. nsv_max vectors
        // (TODO: explore other options like transfering nsv to the host, or having
        //  template functions accepting dimensions as pointers)

        // initialize matrix U
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_m, blocks_nsv, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream, m, nsv_max, U, 0,
                                ldu, strideU);

        // copy left vectors to matrix U
        ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<TT, T>), dim3(blocks_k, blocks_nsv, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream,
                                rocblas_operation_none, k, nsv_max, tmpZ, 0, ldz, strideZ, U, 0,
                                ldu, strideU);

        if(thinSVD)
        {
            // apply ormbr (update with tranformation from bidiagonalization)
            rocsolver_ormbr_unmbr_template<false>(
                handle, rocblas_column_wise, rocblas_side_left, rocblas_operation_none, k, nsv_max,
                k, tmpT, 0, ldt, strideT, tauqp, k, U, 0, ldu, strideU, batch_count, ormbr1_work);

            if(row)
            {
                // apply ormqr (update with transformation from row compression)
                rocsolver_ormqr_unmqr_template<BATCHED>(
                    handle, rocblas_side_left, rocblas_operation_none, m, nsv_max, k, A, shiftA,
                    lda, strideA, tau, k, U, 0, ldu, strideU, batch_count, ormqr_work);
            }
        }
        else
        {
            // apply ormbr (update with tranformation from bidiagonalization)
            rocblas_int mm = row ? m : k;
            rocblas_int kk = row ? k : n;
            rocsolver_ormbr_unmbr_template<BATCHED>(handle, rocblas_column_wise, rocblas_side_left,
                                                    rocblas_operation_none, mm, nsv_max, kk, A,
                                                    shiftA, lda, strideA, tauqp, k, U, 0, ldu,
                                                    strideU, batch_count, ormbr1_work);
        }
    }

    /***** 4. compute/update right vectors *****/
    /**********************************************/
    if(rightvS)
    {
        // For now we work with the k columns in tmpZ, i.e. k vectors
        // (TODO: explore other options like transfering nsv to the host, or having
        //  template functions accepting dimensions as pointers)

        // initialize matrix V
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_nsv, blocks_n, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream, nsv_max, n, V, 0,
                                ldv, strideV);

        // copy right vectors to matrix V
        ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<TT, T>), dim3(blocks_k, blocks_nsv, batch_count),
                                dim3(thread_count, thread_count, 1), 0, stream,
                                rocblas_operation_transpose, k, nsv_max, tmpZ, k, ldz, strideZ, V,
                                0, ldv, strideV);

        if(thinSVD)
        {
            // apply ormbr (update with tranformation from bidiagonalization)
            rocsolver_ormbr_unmbr_template<false>(handle, rocblas_row_wise, rocblas_side_right,
                                                  rocblas_operation_transpose, nsv_max, k, k, tmpT,
                                                  0, ldt, strideT, (tauqp + k * batch_count), k, V,
                                                  0, ldv, strideV, batch_count, ormbr2_work);

            if(!row)
            {
                // apply ormlq (update with transformation from column compression)
                rocsolver_ormlq_unmlq_template<BATCHED>(
                    handle, rocblas_side_right, rocblas_operation_none, nsv_max, n, k, A, shiftA,
                    lda, strideA, tau, k, V, 0, ldv, strideV, batch_count, ormlq_work);
            }
        }
        else
        {
            // apply ormbr (update with tranformation from bidiagonalization)
            rocblas_int nn = row ? k : n;
            rocblas_int kk = row ? m : k;
            rocsolver_ormbr_unmbr_template<BATCHED>(handle, rocblas_row_wise, rocblas_side_right,
                                                    rocblas_operation_transpose, nsv_max, nn, kk, A,
                                                    shiftA, lda, strideA, (tauqp + k * batch_count),
                                                    k, V, 0, ldv, strideV, batch_count, ormbr2_work);
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
