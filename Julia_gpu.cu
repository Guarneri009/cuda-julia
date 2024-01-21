#include <iostream>
#include <cstddef>
#include <thrust/complex.h>
#include <cuda.h>

constexpr auto threads_perblock = 1024;

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << "  " << file << "  " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

inline __device__ float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline __device__ float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline __device__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

__device__ void hsv2rgb_gpu(int h, int s, int v, int *r, int *g, int *b)
{
    if (h < 0)
        h += (1 - h / 360) * 360;
    if (360 < h)
        h %= 360;
    auto h1 = (h * 4096 + 50) / 120;
    auto s1 = (s * 4096 + 50) / 100;
    auto v1 = (v * 4096 + 50) / 100;
    auto h2 = h1 % 4096;
    auto a1{0}, a2{0};
    if (h2 < 2048)
    {
        a1 = (4096 - (2048 - h2) * s1 / 2048) * v1 / 4096;
        a2 = v1;
    }
    else
    {
        a2 = (4096 - (h2 - 2048) * s1 / 2048) * v1 / 4096;
        a1 = v1;
    }

    auto b1 = clamp((a2 * 255 + 2048) / 4096, 0, 255);
    auto b2 = clamp((a1 * 255 + 2048) / 4096, 0, 255);
    auto b3 = clamp(((4096 - s1) * v1 / 4096 * 255 + 2048) / 4096, 0, 255);

    switch (h1 / 4096)
    {
    case 1:
        *g = b1;
        *b = b2;
        *r = b3;
        break;
    case 2:
        *b = b1;
        *r = b2;
        *g = b3;
        break;
    default:
        *r = b1;
        *g = b2;
        *b = b3;
        break;
    }
}

__device__ int julia_cuda(int x, int y, int view_size)
{
    const auto scale = 1.5f;
    auto jx = scale * static_cast<float>(view_size / 2 - x) / (view_size / 2);
    auto jy = scale * static_cast<float>(view_size / 2 - y) / (view_size / 2);

    thrust::complex<float> c(-0.8f, 0.156f);
    thrust::complex<float> z(jx, jy);

    auto i = 0;
    for (i = 0; i < 360; i++)
    {
        z = z * z + c;
        if (thrust::norm(z) > 1000)
            break;
    }

    return i;
}

__global__ void kernel(unsigned char *ptr, int view_size)
{
    auto r{0}, g{0}, b{0};
    auto x = blockIdx.x;
    auto y = threadIdx.x;
    auto offset = x + y * gridDim.x;
    auto value = julia_cuda(x, y, view_size);

    hsv2rgb_gpu(value, 100, 100, &r, &g, &b);

    if (value >= 0 && value <= 20)
    {
        ptr[offset * 4 + 0] = 255;
        ptr[offset * 4 + 1] = 255;
        ptr[offset * 4 + 2] = 255;
        ptr[offset * 4 + 3] = 255;
    }
    else
    {
        ptr[offset * 4 + 0] = r;
        ptr[offset * 4 + 1] = g;
        ptr[offset * 4 + 2] = b;
        ptr[offset * 4 + 3] = 255;
    }
}

extern "C" unsigned char *JuliaGPU(std::size_t size, int view_size)
{
    unsigned char *ptr_gpu;

    HANDLE_ERROR(cudaMalloc((void **)&ptr_gpu, size));

    auto *ptr = new unsigned char[size];
    HANDLE_ERROR(cudaMemcpy(ptr_gpu, ptr, size, cudaMemcpyHostToDevice));

    auto blocks_per_grid = ((view_size * view_size) + threads_perblock - 1) / threads_perblock;
    std::cout << "CUDA kernel [" << blocks_per_grid << "] blocks [" << threads_perblock << "] threads" << std::endl;
    kernel<<<blocks_per_grid, threads_perblock>>>(ptr_gpu, view_size);

    HANDLE_ERROR(cudaMemcpy(ptr, ptr_gpu, size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(ptr_gpu));
    return ptr;
}
