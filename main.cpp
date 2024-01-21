#include <iostream>
#include <string>
#include <cstddef>
#include <memory>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TIME_ELAPSED_START(x) auto x##0 = std::chrono::system_clock::now();
#define TIME_ELAPSED_END(x, mess)                                      \
   auto x##1 = std::chrono::system_clock::now();                       \
   std::chrono::duration<double, std::milli> x##elapsed = x##1 - x##0; \
   std::cout << mess << x##elapsed.count() << " ms" << std::endl;

constexpr auto view_size = 1024;

extern "C" unsigned char *JuliaCPU(std::size_t, int);
extern "C" unsigned char *JuliaGPU(std::size_t, int);

struct RGBA
{
   unsigned char r, g, b, a;
};

int draw(unsigned char *ptr, const std::string &filename)
{
   constexpr std::size_t width{view_size}, height{view_size};
   std::unique_ptr<RGBA[][width]> rgba(new (std::nothrow) RGBA[height][width]);
   if (!rgba)
      return -1;

   for (std::size_t row{}; row < height; ++row)
      for (std::size_t col{}; col < width; ++col)
      {
         auto offset = col + row * view_size;
         rgba[row][col].r = ptr[offset * 4 + 0];
         rgba[row][col].g = ptr[offset * 4 + 1];
         rgba[row][col].b = ptr[offset * 4 + 2];
         rgba[row][col].a = ptr[offset * 4 + 3];
      }

   stbi_write_png(filename.c_str(), static_cast<int>(width), static_cast<int>(height),
                  static_cast<int>(sizeof(RGBA)), rgba.get(), 0);
   return 0;
}

int main()
{

   auto size = view_size * view_size * 4 * sizeof(unsigned char);

   // run on CPU
   TIME_ELAPSED_START(CPU);
   auto *ptr_cpu = JuliaCPU(size, view_size);
   TIME_ELAPSED_END(CPU, "CPU total result..  ");
   draw(ptr_cpu, "picture_CPU.png");
   delete (ptr_cpu);

   // run on CPU
   TIME_ELAPSED_START(GPU);
   auto *ptr_gpu = JuliaGPU(size, view_size);
   TIME_ELAPSED_END(GPU, "GPU total result..  ");
   draw(ptr_gpu, "picture_GPU.png");
   delete (ptr_gpu);
}
