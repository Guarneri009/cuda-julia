cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(main CXX)

option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/*.cu")

if(USE_CUDA)
    enable_language("CUDA")
    add_executable(${PROJECT_NAME} ${cpu_source_files} ${gpu_source_files})
else(USE_CUDA)
    add_executable(${PROJECT_NAME} ${cpu_source_files})
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

# Compile Info
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "52")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_features(${PROJECT_NAME} PUBLIC cuda_std_17)