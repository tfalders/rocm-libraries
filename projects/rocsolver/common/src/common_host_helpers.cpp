/* **************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>

#include "common_host_helpers.hpp"

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_BEGIN_NAMESPACE
#endif

// Return the path to the currently running executable
std::string rocsolver_exepath()
{
#ifdef _WIN32

    std::vector<TCHAR> result(MAX_PATH + 1);
    DWORD length = 0;
    for(;;)
    {
        length = GetModuleFileNameA(nullptr, result.data(), result.size());
        if(length < result.size() - 1)
        {
            result.resize(length + 1);
            break;
        }
        result.resize(result.size() * 2);
    }

    fs::path exepath(result.begin(), result.end());
    exepath = exepath.remove_filename();
    exepath += exepath.empty() ? "" : "/";
    return exepath.string();

#else
    std::string pathstr;
    char* path = realpath("/proc/self/exe", 0);
    if(path)
    {
        char* p = strrchr(path, '/');
        if(p)
        {
            p[1] = 0;
            pathstr = path;
        }
        free(path);
    }
    return pathstr;
#endif
}

fs::path get_sparse_data_dir()
{
    // first check an environment variable
    if(const char* datadir = std::getenv("ROCSOLVER_TEST_DATA"))
        return fs::path{datadir};

    std::vector<std::string> considered;

    // check relative to the running executable
    fs::path exe_path = fs::path(rocsolver_exepath());
    std::vector<fs::path> candidates = {"../share/rocsolver/test", "sparsedata"};
    for(const fs::path& candidate : candidates)
    {
        std::error_code ec;
        fs::path exe_relative = fs::canonical(exe_path / candidate, ec);
        if(!ec)
            return exe_relative;
        considered.push_back(exe_relative.string());
    }

    fmt::print(stderr,
               "Warning: default sparse data directories not found. "
               "Defaulting to current working directory.\nExecutable location: {}\n"
               "Paths considered:\n{}\n",
               exe_path.string(), fmt::join(considered, "\n"));

    return fs::current_path();
}

/***********************************************************************
 * timing functions                                                    *
 ***********************************************************************/

/* CPU Timer (in microseconds): no GPU synchronization
 */
double get_time_us_no_sync()
{
    namespace sc = std::chrono;
    const sc::steady_clock::time_point t = sc::steady_clock::now();
    return double(sc::duration_cast<sc::microseconds>(t.time_since_epoch()).count());
}

/* CPU Timer (in microseconds): synchronize with the default device and return wall time
 */
double get_time_us()
{
    hipError_t status = hipDeviceSynchronize();
#ifdef ROCSOLVER_LIBRARY
    if(status != hipSuccess)
        fmt::print(std::cerr, "{}: [{}] {}\n", __PRETTY_FUNCTION__, hipGetErrorName(status),
                   hipGetErrorString(status));
#else
    THROW_IF_HIP_ERROR(status);
#endif
    return get_time_us_no_sync();
}

/* CPU Timer (in microseconds): synchronize with given queue/stream and return wall time
 */
double get_time_us_sync(hipStream_t stream)
{
    hipError_t status = hipStreamSynchronize(stream);
#ifdef ROCSOLVER_LIBRARY
    if(status != hipSuccess)
        fmt::print(std::cerr, "{}: [{}] {}\n", __PRETTY_FUNCTION__, hipGetErrorName(status),
                   hipGetErrorString(status));
#else
    THROW_IF_HIP_ERROR(status);
#endif
    return get_time_us_no_sync();
}

/*! \brief Get warp size of the current device */
int get_device_warp_size()
{
    int warp_size;
    int device_id;
    hipError_t err;

    err = hipGetDevice(&device_id);

    if(err != hipSuccess)
    {
        return 0;
    }

    err = hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device_id);

    if(err != hipSuccess)
    {
        return 0;
    }

    return warp_size;
}

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_END_NAMESPACE
#endif
