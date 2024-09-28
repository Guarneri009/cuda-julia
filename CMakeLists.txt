cmake_minimum_required(VERSION 3.19.0)
message(STATUS "CMake version: ${CMAKE_VERSION}")

enable_language(CXX)
# 必要に応じて、enable_language(CUDA) は CMake 3.8 以降のバージョンで使うとよいです。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(main CXX)

option(USE_CUDA "Use CUDA" ON)

# ソースファイルの取得
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/*.cu")

# USE_CUDAがONならばCUDAを有効にし、関連するソースファイルをコンパイルに含める
if(USE_CUDA)
    enable_language(CUDA)  # ここでCUDAを有効にする
    add_executable(${PROJECT_NAME} ${cpu_source_files} ${gpu_source_files})
else()
    add_executable(${PROJECT_NAME} ${cpu_source_files})
endif()

# オプション情報の出力
message(STATUS "USE_CUDA: ${USE_CUDA}")

# コンパイル時にUSE_CUDAが定義されるように設定
if(USE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA)
endif()

# CUDA固有のコンパイル設定
if(USE_CUDA)
    # CUDAのターゲットアーキテクチャを設定 (CUDA_ARCHITECTURESはCMake 3.18以上推奨)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "61")

    # C++17とCUDA17を両方サポートするためのコンパイルオプション
    target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17 cuda_std_17)
endif()
