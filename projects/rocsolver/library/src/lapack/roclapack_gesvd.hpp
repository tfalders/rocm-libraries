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

#include "auxiliary/rocauxiliary_bdsqr.hpp"
#include "auxiliary/rocauxiliary_orgbr_ungbr.hpp"
#include "auxiliary/rocauxiliary_ormbr_unmbr.hpp"
#include "rocblas.hpp"
#include "roclapack_gebrd.hpp"
#include "roclapack_gelqf.hpp"
#include "roclapack_geqrf.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** wrapper to xxGQR/xxGLQ_TEMPLATE **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void local_orgqrlq_ungqrlq_template(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    U A,
                                    const rocblas_int shiftA,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    T* ipiv,
                                    const rocblas_stride strideP,
                                    const rocblas_int batch_count,
                                    T* scalars,
                                    T* work,
                                    T* Abyx_tmptr,
                                    T* trfact,
                                    T** workArr,
                                    const bool row)
{
    if(row)
        rocsolver_orgqr_ungqr_template<BATCHED, STRIDED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                         ipiv, strideP, batch_count, scalars, work,
                                                         Abyx_tmptr, trfact, workArr);

    else
        rocsolver_orglq_unglq_template<BATCHED, STRIDED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                         ipiv, strideP, batch_count, scalars, work,
                                                         Abyx_tmptr, trfact, workArr);
}

/** wrapper to xxGQR/xxGLQ_TEMPLATE **/
template <bool BATCHED, typename T, typename U>
void local_orgqrlq_ungqrlq_template(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    U A,
                                    const rocblas_int shiftA,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    T* ipiv,
                                    const rocblas_stride strideP,
                                    const rocblas_int batch_count,
                                    rocsolver_workspace_helper* work_helper,
                                    const bool row)
{
    if(row)
        rocsolver_orgqr_ungqr_template<BATCHED>(handle, m, n, k, A, shiftA, lda, strideA, ipiv,
                                                strideP, batch_count, work_helper);

    else
        rocsolver_orglq_unglq_template<BATCHED>(handle, m, n, k, A, shiftA, lda, strideA, ipiv,
                                                strideP, batch_count, work_helper);
}

/** wrapper to GEQRF/GELQF_TEMPLATE **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void local_geqrlq_template(rocblas_handle handle,
                           const rocblas_int m,
                           const rocblas_int n,
                           U A,
                           const rocblas_int shiftA,
                           const rocblas_int lda,
                           const rocblas_stride strideA,
                           T* ipiv,
                           const rocblas_stride strideP,
                           const rocblas_int batch_count,
                           T* scalars,
                           void* work_workArr,
                           T* Abyx_norms_trfact,
                           T* diag_tmptr,
                           T** workArr,
                           const bool row)
{
    if(row)
        rocsolver_geqrf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                   strideP, batch_count, scalars, work_workArr,
                                                   Abyx_norms_trfact, diag_tmptr, workArr);
    else
        rocsolver_gelqf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                   strideP, batch_count, scalars, work_workArr,
                                                   Abyx_norms_trfact, diag_tmptr, workArr);
}

/** wrapper to GEQRF/GELQF_TEMPLATE **/
template <bool BATCHED, typename T, typename U>
void local_geqrlq_template(rocblas_handle handle,
                           const rocblas_int m,
                           const rocblas_int n,
                           U A,
                           const rocblas_int shiftA,
                           const rocblas_int lda,
                           const rocblas_stride strideA,
                           T* ipiv,
                           const rocblas_stride strideP,
                           const rocblas_int batch_count,
                           rocsolver_workspace_helper* work_helper,
                           const bool row)
{
    if(row)
        rocsolver_geqrf_template<BATCHED>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
                                          batch_count, work_helper);
    else
        rocsolver_gelqf_template<BATCHED>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
                                          batch_count, work_helper);
}

/** Argument checking **/
template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_argCheck(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int lda,
                                        TT* S,
                                        T* U,
                                        const rocblas_int ldu,
                                        T* V,
                                        const rocblas_int ldv,
                                        TT* E,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((left_svect != rocblas_svect_all && left_svect != rocblas_svect_singular
        && left_svect != rocblas_svect_overwrite && left_svect != rocblas_svect_none)
       || (right_svect != rocblas_svect_all && right_svect != rocblas_svect_singular
           && right_svect != rocblas_svect_overwrite && right_svect != rocblas_svect_none)
       || (left_svect == rocblas_svect_overwrite && right_svect == rocblas_svect_overwrite))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if((left_svect == rocblas_svect_all || left_svect == rocblas_svect_singular) && ldu < m)
        return rocblas_status_invalid_size;
    if((right_svect == rocblas_svect_all && ldv < n)
       || (right_svect == rocblas_svect_singular && ldv < std::min(m, n)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && m && !A) || (std::min(m, n) > 1 && !E) || (std::min(m, n) && !S)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if((left_svect == rocblas_svect_all && m && !U)
       || (left_svect == rocblas_svect_singular && std::min(m, n) && !U))
        return rocblas_status_invalid_pointer;
    if((right_svect == rocblas_svect_all || right_svect == rocblas_svect_singular) && n && !V)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_gesvd_getMemorySize(const rocblas_svect left_svect,
                                   const rocblas_svect right_svect,
                                   const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   const rocblas_workmode fast_alg,
                                   rocsolver_workspace_helper* work_helper)
{
    // if quick return, set workspace to zero
    if(n == 0 || m == 0 || batch_count == 0)
        return;

    work_helper->set_nested_capacity(7);
    auto gebrd_work = work_helper->add_nested();
    auto bdsqr_work = work_helper->add_nested();
    auto gexxf_work = work_helper->add_nested();
    auto ormbr_work = work_helper->add_nested();
    auto orgbr1_work = work_helper->add_nested();
    auto orgbr2_work = work_helper->add_nested();
    auto orgxx_work = work_helper->add_nested();

    // booleans used to determine the path that the execution will follow:
    const bool row = (m >= n);
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool leftvO = (left_svect == rocblas_svect_overwrite);
    const bool leftvA = (left_svect == rocblas_svect_all);
    const bool leftvN = (left_svect == rocblas_svect_none);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool rightvO = (right_svect == rocblas_svect_overwrite);
    const bool rightvA = (right_svect == rocblas_svect_all);
    const bool rightvN = (right_svect == rocblas_svect_none);
    //const bool leadvS = row ? leftvS : rightvS;
    const bool leadvO = row ? leftvO : rightvO;
    const bool leadvA = row ? leftvA : rightvA;
    const bool leadvN = row ? leftvN : rightvN;
    //const bool othervS = !row ? leftvS : rightvS;
    const bool othervO = !row ? leftvO : rightvO;
    //const bool othervA = !row ? leftvA : rightvA;
    const bool othervN = !row ? leftvN : rightvN;
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
    const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

    // auxiliary sizes and variables
    const rocblas_int k = std::min(m, n);
    const rocblas_int kk = std::max(m, n);
    const rocblas_int nu = leftvN ? 0 : ((fast_thinSVD || (thinSVD && leadvN)) ? k : m);
    const rocblas_int nv = rightvN ? 0 : ((fast_thinSVD || (thinSVD && leadvN)) ? k : n);
    const rocblas_storev storev_lead = row ? rocblas_column_wise : rocblas_row_wise;
    const rocblas_storev storev_other = row ? rocblas_row_wise : rocblas_column_wise;
    const rocblas_side side = row ? rocblas_side_right : rocblas_side_left;
    const rocblas_operation trans
        = row ? rocblas_operation_none : rocblas_operation_conjugate_transpose;
    rocblas_int mn;

    // workspace required for the bidiagonalization
    if(thinSVD)
        rocsolver_gebrd_getMemorySize<BATCHED, T>(k, k, batch_count, gebrd_work);
    else
        rocsolver_gebrd_getMemorySize<BATCHED, T>(m, n, batch_count, gebrd_work);

    // workspace required for the SVD of the bidiagonal form
    rocsolver_bdsqr_getMemorySize<T, S>(k, nv, nu, 0, batch_count, bdsqr_work);

    // extra requirements for QR/LQ factorization
    if(thinSVD)
    {
        if(row)
            rocsolver_geqrf_getMemorySize<BATCHED, T>(m, n, batch_count, gexxf_work);
        else
            rocsolver_gelqf_getMemorySize<BATCHED, T>(m, n, batch_count, gexxf_work);
    }

    // extra requirements for orthonormal/unitary matrix generation
    // ormbr
    if(thinSVD && !fast_thinSVD && !leadvN)
        rocsolver_ormbr_unmbr_getMemorySize<BATCHED, T>(storev_lead, side, trans, m, n, k,
                                                        batch_count, ormbr_work);
    // orgbr
    if(thinSVD)
    {
        if(!othervN)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(storev_other, k, k, k, batch_count,
                                                            orgbr1_work);

        if(fast_thinSVD && !leadvN)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(storev_lead, k, k, k, batch_count,
                                                            orgbr2_work);
    }
    else
    {
        mn = (row && leftvS) ? n : m;
        if(leftvS || leftvA)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(rocblas_column_wise, m, mn, n,
                                                            batch_count, orgbr1_work);
        else if(leftvO)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(rocblas_column_wise, m, k, n,
                                                            batch_count, orgbr1_work);

        mn = (!row && rightvS) ? m : n;
        if(rightvS || rightvA)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(rocblas_row_wise, mn, n, m, batch_count,
                                                            orgbr2_work);
        else if(rightvO)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(rocblas_row_wise, k, n, m, batch_count,
                                                            orgbr2_work);
    }
    // orgqr/orglq
    if(thinSVD && !leadvN)
    {
        if(leadvA)
        {
            if(row)
                rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(kk, kk, k, batch_count, orgxx_work);
            else
                rocsolver_orglq_unglq_getMemorySize<BATCHED, T>(kk, kk, k, batch_count, orgxx_work);
        }
        else
        {
            if(row)
                rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(m, n, k, batch_count, orgxx_work);
            else
                rocsolver_orglq_unglq_getMemorySize<BATCHED, T>(m, n, k, batch_count, orgxx_work);
        }
    }

    // size of array tau to store householder scalars on intermediate
    // orthonormal/unitary matrices
    size_t size_tau = 2 * sizeof(T) * std::min(m, n) * batch_count;

    // size of arrays to store temporary copies
    size_t size_tempArrayT
        = (fast_thinSVD || (thinSVD && leadvO && othervN)) ? sizeof(T) * k * k * batch_count : 0;
    size_t size_tempArrayC
        = (fast_thinSVD && (othervN || othervO || leadvO)) ? sizeof(T) * m * n * batch_count : 0;

    // size of array of pointers to workspace
    size_t size_workArr = 0;
    if(BATCHED)
        size_workArr = sizeof(T*) * batch_count;

    work_helper->assign_sizes({size_tau, size_tempArrayT, size_tempArrayC}, {size_workArr});
}

template <bool BATCHED, bool STRIDED, typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_template(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_stride shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        TT* S,
                                        const rocblas_stride strideS,
                                        T* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        T* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        TT* E,
                                        const rocblas_stride strideE,
                                        const rocblas_workmode fast_alg,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        rocsolver_workspace_helper* work_helper)
{
    ROCSOLVER_ENTER("gesvd", "leftsv:", left_svect, "rightsv:", right_svect, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "ldu:", ldu, "ldv:", ldv, "mode:", fast_alg,
                    "bc:", batch_count);

    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // quick return
    if(n == 0 || m == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // prepare workspace
    auto gebrd_work = work_helper->get_nested(0);
    auto bdsqr_work = work_helper->get_nested(1);
    auto gexxf_work = work_helper->get_nested(2);
    auto ormbr_work = work_helper->get_nested(3);
    auto orgbr1_work = work_helper->get_nested(4);
    auto orgbr2_work = work_helper->get_nested(5);
    auto orgxx_work = work_helper->get_nested(6);
    T* tau = (T*)(*work_helper)[0];
    T* tempArrayT = (T*)(*work_helper)[1];
    T* tempArrayC = (T*)(*work_helper)[2];
    T** workArr = (T**)(*work_helper)[3];

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants to use when calling rocablas functions
    T one = 1;
    T zero = 0;

    // booleans used to determine the path that the execution will follow:
    const bool row = (m >= n);
    const bool leftvS = (left_svect == rocblas_svect_singular);
    const bool leftvO = (left_svect == rocblas_svect_overwrite);
    const bool leftvA = (left_svect == rocblas_svect_all);
    const bool leftvN = (left_svect == rocblas_svect_none);
    const bool rightvS = (right_svect == rocblas_svect_singular);
    const bool rightvO = (right_svect == rocblas_svect_overwrite);
    const bool rightvA = (right_svect == rocblas_svect_all);
    const bool rightvN = (right_svect == rocblas_svect_none);
    const bool leadvS = row ? leftvS : rightvS;
    const bool leadvO = row ? leftvO : rightvO;
    const bool leadvA = row ? leftvA : rightvA;
    const bool leadvN = row ? leftvN : rightvN;
    const bool othervS = !row ? leftvS : rightvS;
    const bool othervO = !row ? leftvO : rightvO;
    const bool othervA = !row ? leftvA : rightvA;
    const bool othervN = !row ? leftvN : rightvN;
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
    const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

    // auxiliary sizes and variables
    const rocblas_int k = std::min(m, n);
    const rocblas_int kk = std::max(m, n);
    const rocblas_int shiftX = 0;
    const rocblas_int shiftY = 0;
    const rocblas_int shiftUV = 0;
    const rocblas_int shiftT = 0;
    const rocblas_int shiftC = 0;
    const rocblas_int shiftU = 0;
    const rocblas_int shiftV = 0;
    const rocblas_int ldx = thinSVD ? k : m;
    const rocblas_int ldy = thinSVD ? k : n;
    const rocblas_stride strideX = ldx * GEBRD_BLOCKSIZE;
    const rocblas_stride strideY = ldy * GEBRD_BLOCKSIZE;
    T* bufferT = tempArrayT;
    rocblas_int ldt = k;
    rocblas_stride strideT = k * k;
    T* bufferC = tempArrayC;
    rocblas_int ldc = m;
    rocblas_stride strideC = m * n;

    T* UV;
    rocblas_int lduv, mn, nu, nv;
    rocblas_int offset_other, offset_lead;
    rocblas_storev storev_other, storev_lead;
    rocblas_stride strideUV;
    rocblas_fill uplo;
    rocblas_side side;
    rocblas_operation trans;

    nu = leftvN ? 0 : m;
    nv = rightvN ? 0 : n;
    if(row)
    {
        UV = U;
        lduv = ldu;
        strideUV = strideU;
        uplo = rocblas_fill_upper;
        storev_other = rocblas_row_wise;
        storev_lead = rocblas_column_wise;
        offset_other = k * batch_count;
        offset_lead = 0;
        if(othervS || othervA)
        {
            bufferC = V;
            ldc = ldv;
            strideC = strideV;
            if(!fast_thinSVD)
            {
                bufferT = V;
                ldt = ldv;
                strideT = strideV;
            }
        }
        side = rocblas_side_right;
        trans = rocblas_operation_none;
    }
    else
    {
        UV = V;
        lduv = ldv;
        strideUV = strideV;
        uplo = rocblas_fill_lower;
        storev_other = rocblas_column_wise;
        storev_lead = rocblas_row_wise;
        offset_other = 0;
        offset_lead = k * batch_count;
        if(othervS || othervA)
        {
            bufferC = U;
            ldc = ldu;
            strideC = strideU;
            if(!fast_thinSVD)
            {
                bufferT = U;
                ldt = ldu;
                strideT = strideU;
            }
        }
        side = rocblas_side_left;
        trans = COMPLEX ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;
    }

    // common block sizes and number of threads for internal kernels
    constexpr rocblas_int thread_count = 32;
    const rocblas_int blocks_m = (m - 1) / thread_count + 1;
    const rocblas_int blocks_n = (n - 1) / thread_count + 1;
    const rocblas_int blocks_k = (k - 1) / thread_count + 1;

    /** A thin SVD could be computed for matrices with sufficiently more rows than
        columns (or columns than rows) by starting with a QR factorization (or LQ
        factorization) and working with the triangular factor afterwards. When
        computing a thin SVD, a fast algorithm could be executed by doing some
        computations out-of-place. **/

    if(thinSVD)
    /*******************************************/
    /********** CASE: CHOOSE THIN-SVD **********/
    /*******************************************/
    {
        if(leadvN)
        /***** SUB-CASE: USE THIN-SVD WITH NO LEAD-DIMENSION VECTORS *****/
        /*****************************************************************/
        {
            nu = leftvN ? 0 : k;
            nv = rightvN ? 0 : k;

            //*** STAGE 1: Row (or column) compression ***//
            local_geqrlq_template<BATCHED>(handle, m, n, A, shiftA, lda, strideA, tau, k,
                                           batch_count, gexxf_work, row);

            //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
            // N/A

            //*** STAGE 3: Bidiagonalization ***//
            // clean triangular factor
            ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                    dim3(thread_count, thread_count, 1), 0, stream, k, k, A, shiftA,
                                    lda, strideA, uplo);

            rocsolver_gebrd_template<BATCHED>(handle, k, k, A, shiftA, lda, strideA, S, strideS, E,
                                              strideE, tau, k, (tau + k * batch_count), k,
                                              batch_count, gebrd_work);

            //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
            if(!othervN)
                rocsolver_orgbr_ungbr_template<BATCHED>(handle, storev_other, k, k, k, A, shiftA,
                                                        lda, strideA, (tau + offset_other), k,
                                                        batch_count, orgbr1_work);

            //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
            if(row)
                rocsolver_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, 0, S, strideS, E,
                                            strideE, A, shiftA, lda, strideA, U, shiftU, ldu, strideU,
                                            (W) nullptr, 0, 1, 1, info, batch_count, bdsqr_work);
            else
                rocsolver_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, 0, S, strideS, E,
                                            strideE, V, shiftV, ldv, strideV, A, shiftA, lda, strideA,
                                            (W) nullptr, 0, 1, 1, info, batch_count, bdsqr_work);

            //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
            if(othervS || othervA)
            {
                mn = row ? n : m;
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, mn, mn, A,
                                        shiftA, lda, strideA, bufferC, shiftC, ldc, strideC);
            }
        }

        else if(fast_thinSVD)
        /***** SUB-CASE: USE FAST (OUT-OF-PLACE) THIN-SVD ALGORITHM *****/
        /****************************************************************/
        {
            nu = leftvN ? 0 : k;
            nv = rightvN ? 0 : k;

            //*** STAGE 1: Row (or column) compression ***//
            local_geqrlq_template<BATCHED>(handle, m, n, A, shiftA, lda, strideA, tau, k,
                                           batch_count, gexxf_work, row);

            if(leadvA)
                // copy factorization to U or V when needed
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, m, n, A,
                                        shiftA, lda, strideA, UV, shiftUV, lduv, strideUV);

            // copy the triangular part to be used in the bidiagonalization
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                    dim3(thread_count, thread_count, 1), 0, stream, k, k, A, shiftA,
                                    lda, strideA, bufferT, shiftT, ldt, strideT, no_mask{}, uplo);

            //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
            if(leadvA)
                local_orgqrlq_ungqrlq_template<false>(handle, kk, kk, k, UV, shiftUV, lduv, strideUV,
                                                      tau, k, batch_count, orgxx_work, row);
            else
                local_orgqrlq_ungqrlq_template<BATCHED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                        tau, k, batch_count, orgxx_work, row);

            //*** STAGE 3: Bidiagonalization ***//
            // clean triangular factor
            ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                    dim3(thread_count, thread_count, 1), 0, stream, k, k, bufferT,
                                    shiftT, ldt, strideT, uplo);

            rocsolver_gebrd_template<false>(handle, k, k, bufferT, shiftT, ldt, strideT, S, strideS,
                                            E, strideE, tau, k, (tau + k * batch_count), k,
                                            batch_count, gebrd_work);

            if(!othervN)
                // copy results to generate non-lead vectors if required
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                        bufferT, shiftT, ldt, strideT, bufferC, shiftC, ldc, strideC);

            //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
            // for lead-dimension vectors
            rocsolver_orgbr_ungbr_template<false>(handle, storev_lead, k, k, k, bufferT, shiftT,
                                                  ldt, strideT, (tau + offset_lead), k, batch_count,
                                                  orgbr2_work);

            // for the other-side vectors
            if(!othervN)
                rocsolver_orgbr_ungbr_template<false>(handle, storev_other, k, k, k, bufferC,
                                                      shiftC, ldc, strideC, (tau + offset_other), k,
                                                      batch_count, orgbr1_work);

            //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
            if(row)
                rocsolver_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, 0, S, strideS, E,
                                            strideE, bufferC, shiftC, ldc, strideC, bufferT, shiftT,
                                            ldt, strideT, (T*)nullptr, 0, 1, 1, info, batch_count,
                                            bdsqr_work);
            else
                rocsolver_bdsqr_template<T>(handle, rocblas_fill_upper, k, nv, nu, 0, S, strideS, E,
                                            strideE, bufferT, shiftT, ldt, strideT, bufferC, shiftC,
                                            ldc, strideC, (T*)nullptr, 0, 1, 1, info, batch_count,
                                            bdsqr_work);

            //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
            if(leadvO)
            {
                bufferC = tempArrayC;
                ldc = m;
                strideC = m * n;

                // update
                if(row)
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, A, shiftA, lda, strideA, bufferT, shiftT, ldt, strideT,
                                   &zero, bufferC, shiftC, ldc, strideC, batch_count, workArr);
                else
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, bufferT, shiftT, ldt, strideT, A, shiftA, lda, strideA,
                                   &zero, bufferC, shiftC, ldc, strideC, batch_count, workArr);

                // copy to overwrite A
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, m, n,
                                        bufferC, shiftC, ldc, strideC, A, shiftA, lda, strideA);
            }
            else if(leadvS)
            {
                // update
                if(row)
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, A, shiftA, lda, strideA, bufferT, shiftT, ldt, strideT,
                                   &zero, UV, shiftUV, lduv, strideUV, batch_count, workArr);
                else
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, bufferT, shiftT, ldt, strideT, A, shiftA, lda, strideA,
                                   &zero, UV, shiftUV, lduv, strideUV, batch_count, workArr);

                // overwrite A if required
                if(othervO)
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                            bufferC, shiftC, ldc, strideC, A, shiftA, lda, strideA);
            }
            else
            {
                // update
                if(row)
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, UV, shiftUV, lduv, strideUV, bufferT, shiftT, ldt, strideT,
                                   &zero, A, shiftA, lda, strideA, batch_count, workArr);
                else
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                                   &one, bufferT, shiftT, ldt, strideT, UV, shiftUV, lduv, strideUV,
                                   &zero, A, shiftA, lda, strideA, batch_count, workArr);

                // copy back to U/V
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, m, n, A,
                                        shiftA, lda, strideA, UV, shiftUV, lduv, strideUV);

                // overwrite A if required
                if(othervO)
                    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                            dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                            bufferC, shiftC, ldc, strideC, A, shiftA, lda, strideA);
            }
        }

        else
        /************ SUB-CASE: USE IN-PLACE THIN-SVD ALGORITHM *******/
        /**************************************************************/
        {
            /** (Note: A compression is not required when leadvO and othervN. We are
                compressing matrix A here, albeit requiring extra workspace for the purpose of
                testing. -See corresponding unit test for more details-) **/

            //*** STAGE 1: Row (or column) compression ***//
            local_geqrlq_template<BATCHED>(handle, m, n, A, shiftA, lda, strideA, tau, k,
                                           batch_count, gexxf_work, row);

            if(!leadvO)
                // copy factorization to U or V when needed
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_n, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, m, n, A,
                                        shiftA, lda, strideA, UV, shiftUV, lduv, strideUV);

            if(othervS || othervA || (leadvO && othervN))
                // copy the triangular part
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k, A,
                                        shiftA, lda, strideA, bufferT, shiftT, ldt, strideT,
                                        no_mask{}, uplo);

            //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
            if(leadvO)
                local_orgqrlq_ungqrlq_template<BATCHED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                        tau, k, batch_count, orgxx_work, row);
            else if(leadvA)
                local_orgqrlq_ungqrlq_template<false>(handle, kk, kk, k, UV, shiftUV, lduv, strideUV,
                                                      tau, k, batch_count, orgxx_work, row);
            else
                local_orgqrlq_ungqrlq_template<false>(handle, m, n, k, UV, shiftUV, lduv, strideUV,
                                                      tau, k, batch_count, orgxx_work, row);

            //*** STAGE 3: Bidiagonalization ***//
            if(othervS || othervA || (leadvO && othervN))
            {
                // clean triangular factor
                ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k,
                                        bufferT, shiftT, ldt, strideT, uplo);

                rocsolver_gebrd_template<false>(handle, k, k, bufferT, shiftT, ldt, strideT, S,
                                                strideS, E, strideE, tau, k,
                                                (tau + k * batch_count), k, batch_count, gebrd_work);

                uplo = rocblas_fill_upper;
            }
            else
            {
                // clean triangular factor
                ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks_k, blocks_k, batch_count),
                                        dim3(thread_count, thread_count, 1), 0, stream, k, k, A,
                                        shiftA, lda, strideA, uplo);

                rocsolver_gebrd_template<BATCHED>(handle, k, k, A, shiftA, lda, strideA, S, strideS,
                                                  E, strideE, tau, k, (tau + k * batch_count), k,
                                                  batch_count, gebrd_work);

                uplo = rocblas_fill_upper;
            }

            //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
            // for lead-dimension vectors
            if(othervS || othervA || (leadvO && othervN))
            {
                if(leadvO)
                    rocsolver_ormbr_unmbr_template<BATCHED>(
                        handle, storev_lead, side, trans, m, n, k, bufferT, shiftT, ldt, strideT,
                        (tau + offset_lead), k, A, shiftA, lda, strideA, batch_count, ormbr_work);
                else
                    rocsolver_ormbr_unmbr_template<false>(handle, storev_lead, side, trans, m, n, k,
                                                          bufferT, shiftT, ldt, strideT,
                                                          (tau + offset_lead), k, UV, shiftUV, lduv,
                                                          strideUV, batch_count, ormbr_work);
            }
            else
                rocsolver_ormbr_unmbr_template<BATCHED>(
                    handle, storev_lead, side, trans, m, n, k, A, shiftA, lda, strideA,
                    (tau + offset_lead), k, UV, shiftUV, lduv, strideUV, batch_count, ormbr_work);

            // for the other-side vectors
            if(othervS || othervA)
                rocsolver_orgbr_ungbr_template<false>(handle, storev_other, k, k, k, bufferT,
                                                      shiftT, ldt, strideT, (tau + offset_other), k,
                                                      batch_count, orgbr1_work);
            else if(othervO)
                rocsolver_orgbr_ungbr_template<BATCHED>(handle, storev_other, k, k, k, A, shiftA,
                                                        lda, strideA, (tau + offset_other), k,
                                                        batch_count, orgbr1_work);

            //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
            uplo = rocblas_fill_upper;
            if(!leftvO && !rightvO)
            {
                rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V,
                                            shiftV, ldv, strideV, U, shiftU, ldu, strideU,
                                            (T*)nullptr, 0, 1, 1, info, batch_count, bdsqr_work);
            }
            else if(leftvO && !rightvO)
            {
                rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V,
                                            shiftV, ldv, strideV, A, shiftA, lda, strideA,
                                            (W) nullptr, 0, 1, 1, info, batch_count, bdsqr_work);
            }
            else
            {
                rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, A,
                                            shiftA, lda, strideA, U, shiftU, ldu, strideU,
                                            (W) nullptr, 0, 1, 1, info, batch_count, bdsqr_work);
            }

            //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
            // N/A
        }
    }

    else
    /*********************************************/
    /********** CASE: CHOOSE NORMAL SVD **********/
    /*********************************************/
    {
        //*** STAGE 1: Row (or column) compression ***//
        // N/A

        //*** STAGE 2: generate orthonormal/unitary matrix from row/column compression ***//
        // N/A

        //*** STAGE 3: Bidiagonalization ***//
        rocsolver_gebrd_template<BATCHED>(handle, m, n, A, shiftA, lda, strideA, S, strideS, E,
                                          strideE, tau, k, (tau + k * batch_count), k, batch_count,
                                          gebrd_work);

        //*** STAGE 4: generate orthonormal/unitary matrices from bidiagonalization ***//
        if(leftvS || leftvA)
        {
            // copy data to matrix U where orthogonal matrix will be generated
            mn = (row && leftvS) ? n : m;
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_k, batch_count),
                                    dim3(thread_count, thread_count, 1), 0, stream, m, k, A, shiftA,
                                    lda, strideA, U, shiftU, ldu, strideU);

            rocsolver_orgbr_ungbr_template<false>(handle, rocblas_column_wise, m, mn, n, U, shiftU,
                                                  ldu, strideU, tau, k, batch_count, orgbr1_work);
        }

        if(rightvS || rightvA)
        {
            // copy data to matrix V where othogonal matrix will be generated
            mn = (!row && rightvS) ? m : n;
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_n, batch_count),
                                    dim3(thread_count, thread_count, 1), 0, stream, k, n, A, shiftA,
                                    lda, strideA, V, shiftV, ldv, strideV);

            rocsolver_orgbr_ungbr_template<false>(handle, rocblas_row_wise, mn, n, m, V, shiftV,
                                                  ldv, strideV, (tau + k * batch_count), k,
                                                  batch_count, orgbr2_work);
        }

        if(leftvO)
        {
            rocsolver_orgbr_ungbr_template<BATCHED>(handle, rocblas_column_wise, m, k, n, A, shiftA,
                                                    lda, strideA, tau, k, batch_count, orgbr1_work);
        }

        if(rightvO)
        {
            rocsolver_orgbr_ungbr_template<BATCHED>(handle, rocblas_row_wise, k, n, m, A, shiftA,
                                                    lda, strideA, (tau + k * batch_count), k,
                                                    batch_count, orgbr2_work);
        }

        //*** STAGE 5: Compute singular values and vectors from the bidiagonal form ***//
        if(!leftvO && !rightvO)
        {
            rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V,
                                        shiftV, ldv, strideV, U, shiftU, ldu, strideU, (T*)nullptr,
                                        0, 1, 1, info, batch_count, bdsqr_work);
        }

        else if(leftvO && !rightvO)
        {
            rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V,
                                        shiftV, ldv, strideV, A, shiftA, lda, strideA, (W) nullptr,
                                        0, 1, 1, info, batch_count, bdsqr_work);
        }

        else
        {
            rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, A,
                                        shiftA, lda, strideA, U, shiftU, ldu, strideU, (W) nullptr,
                                        0, 1, 1, info, batch_count, bdsqr_work);
        }

        //*** STAGE 6: update vectors with orthonormal/unitary matrices ***//
        // N/A
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
