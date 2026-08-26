cmake_minimum_required(VERSION 3.18)
project(smolvla_trt_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

add_library(smolvla_fused_rope_layout SHARED fused_rope_layout.cu)
target_include_directories(
  smolvla_fused_rope_layout
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/x86_64-linux-gnu
)
target_link_libraries(
  smolvla_fused_rope_layout
  PRIVATE
    CUDA::cudart
    nvinfer
    nvinfer_plugin
)

set_target_properties(
  smolvla_fused_rope_layout
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "smolvla_fused_rope_layout"
)

add_executable(smolvla_fused_attention_v3_smoke fused_attention_v3_smoke.cpp)
target_include_directories(
  smolvla_fused_attention_v3_smoke
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/x86_64-linux-gnu
)
target_link_libraries(
  smolvla_fused_attention_v3_smoke
  PRIVATE
    smolvla_fused_rope_layout
    CUDA::cudart
    nvinfer
    nvinfer_plugin
)
set_target_properties(
  smolvla_fused_attention_v3_smoke
  PROPERTIES
    BUILD_RPATH "$ORIGIN"
)

add_executable(smolvla_fused_attention_v3_real fused_attention_v3_real.cpp)
target_include_directories(
  smolvla_fused_attention_v3_real
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/x86_64-linux-gnu
)
target_link_libraries(
  smolvla_fused_attention_v3_real
  PRIVATE
    smolvla_fused_rope_layout
    CUDA::cudart
    nvinfer
    nvinfer_plugin
)
set_target_properties(
  smolvla_fused_attention_v3_real
  PROPERTIES
    BUILD_RPATH "$ORIGIN"
)

add_executable(smolvla_unfused_attention_trt_real unfused_attention_trt_real.cpp)
target_include_directories(
  smolvla_unfused_attention_trt_real
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/x86_64-linux-gnu
)
target_link_libraries(
  smolvla_unfused_attention_trt_real
  PRIVATE
    CUDA::cudart
    nvinfer
)
set_target_properties(
  smolvla_unfused_attention_trt_real
  PROPERTIES
    BUILD_RPATH "$ORIGIN"
)

add_executable(smolvla_smoothquant_linear_epilogue smoothquant_linear_epilogue.cu)
target_include_directories(
  smolvla_smoothquant_linear_epilogue
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
  smolvla_smoothquant_linear_epilogue
  PRIVATE
    CUDA::cudart
)

add_executable(smolvla_smoothquant_linear_cublaslt smoothquant_linear_cublaslt.cu)
target_include_directories(
  smolvla_smoothquant_linear_cublaslt
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
  smolvla_smoothquant_linear_cublaslt
  PRIVATE
    CUDA::cudart
    CUDA::cublasLt
)

add_executable(smolvla_smoothquant_qkv_cublaslt smoothquant_qkv_cublaslt.cu)
target_include_directories(
  smolvla_smoothquant_qkv_cublaslt
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
  smolvla_smoothquant_qkv_cublaslt
  PRIVATE
    CUDA::cudart
    CUDA::cublasLt
)

add_executable(smolvla_smoothquant_qkv_plugin_v3_smoke smoothquant_qkv_plugin_v3_smoke.cu)
target_include_directories(
  smolvla_smoothquant_qkv_plugin_v3_smoke
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/x86_64-linux-gnu
)
target_link_libraries(
  smolvla_smoothquant_qkv_plugin_v3_smoke
  PRIVATE
    CUDA::cudart
    CUDA::cublasLt
    nvinfer
    nvinfer_plugin
)
set_target_properties(
  smolvla_smoothquant_qkv_plugin_v3_smoke
  PROPERTIES
    BUILD_RPATH "$ORIGIN"
)

add_executable(smolvla_smoothquant_qkv_plugin_serialized_smoke smoothquant_qkv_plugin_serialized_smoke.cu)
target_include_directories(
  smolvla_smoothquant_qkv_plugin_serialized_smoke
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/x86_64-linux-gnu
)
target_link_libraries(
  smolvla_smoothquant_qkv_plugin_serialized_smoke
  PRIVATE
    CUDA::cudart
    CUDA::cublasLt
    nvinfer
    nvinfer_plugin
)
set_target_properties(
  smolvla_smoothquant_qkv_plugin_serialized_smoke
  PROPERTIES
    BUILD_RPATH "$ORIGIN"
)
