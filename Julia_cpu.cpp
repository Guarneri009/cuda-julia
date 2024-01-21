#include <new>
#include <complex>
#include <cstddef>
#include <iostream>

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

// https://scrapbox.io/ePi5131/%E6%8B%A1%E5%BC%B5%E7%B7%A8%E9%9B%86%E3%81%AEHSV%E9%96%A2%E6%95%B0%E3%81%AEHSV%3ERGB%E5%A4%89%E6%8F%9B%E3%81%AE%E5%86%85%E5%AE%B9
void hsv2rgb_cpu(int h, int s, int v, int *r, int *g, int *b)
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

int julia_CPU(int x, int y, int view_size)
{
    const auto scale = 1.5f;
    auto jx = scale * (float)(view_size / 2 - x) / (view_size / 2);
    auto jy = scale * (float)(view_size / 2 - y) / (view_size / 2);

    std::complex<float> c(-0.8f, 0.156f);
    std::complex<float> z(jx, jy);

    auto i = 0;
    for (i = 0; i < 360; i++)
    {
        z = z * z + c;
        if (std::norm(z) > 1000)
            break;
    }
    return i;
}

extern "C" unsigned char *JuliaCPU(std::size_t size, int view_size)
{
    auto *ptr = new unsigned char[size];
    auto r{0}, g{0}, b{0};

    for (auto y = 0; y < view_size; y++)
    {
        for (auto x = 0; x < view_size; x++)
        {
            auto offset = x + y * view_size;
            auto value = julia_CPU(x, y, view_size);

            if (value >= 0 && value <= 20)
            {
                ptr[offset * 4 + 0] = 255;
                ptr[offset * 4 + 1] = 255;
                ptr[offset * 4 + 2] = 255;
                ptr[offset * 4 + 3] = 255;
                continue;
            }

            hsv2rgb_cpu(value, 100, 100, &r, &g, &b);

            ptr[offset * 4 + 0] = r;
            ptr[offset * 4 + 1] = g;
            ptr[offset * 4 + 2] = b;
            ptr[offset * 4 + 3] = 255;
        }
    }
    return ptr;
}
