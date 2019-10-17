cmake_minimum_required(VERSION 3.7)

file(GLOB TORCH_SPARSE_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/torch_sparse/*.cpp
)
set(TORCH_SPARSE_SRCS
	${TORCH_SPARSE_SRCS}
)

file(GLOB TORCH_SPARSE_CUDA_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/torch_sparse/*.cu
)
set(TORCH_SPARSE_CUDA_SRCS
  ${TORCH_SPARSE_CUDA_SRCS}
)

set(CMAKE_CXX_STANDARD 11)
SET(CXX11_ABI 0)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}")

set(CUDA_LINK_LIBRARIES_KEYWORD "PUBLIC")
find_package(CUDA REQUIRED)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} 
	-DGOOGLE_CUDA=1 
  -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}
	-O3 -Xcompiler -fPIC -std=c++11 --prec-div=false --prec-sqrt=false \
 	-gencode=arch=compute_35,code=sm_35 \
	-gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_52,code=sm_52 \
 	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
 	-gencode=arch=compute_70,code=sm_70 \
 	-gencode=arch=compute_70,code=compute_70
" )
message("TORCH SPARSE CUDA IS ${TORCH_SPARSE_CUDA_SRCS}")
message("TORCH SPARSE IS ${TORCH_SPARSE_SRCS}")
cuda_add_library(torch_sparse_cuda SHARED ${TORCH_SPARSE_CUDA_SRCS})
target_include_directories(torch_sparse_cuda PUBLIC
    ${PYTORCH_DIR}/include
)
message("-- Cuda Libs ${CUDA_LIBRARIES}")

# PYTORCH_DIR
IF(DEFINED ENV{PYTORCH_DIR})
  SET(PYTORCH_DIR $ENV{PYTORCH_DIR})
ENDIF()
IF("${PYTORCH_DIR}" STREQUAL "")
	message(FATAL_ERROR "Please pass in a Pytorch directory with -DPYTORCH_DIR=")
ENDIF()
message("-- Using pytorch dir ${PYTORCH_DIR}")

IF(DEFINED ENV{TVM_DIR})
  set(TVM_DIR $ENV{TVM_DIR})
  message("-- Using TVM dir ${TVM_DIR}")
  set(TVM_INCLUDE_DIRS
    ${TVM_DIR}/include/
    ${TVM_DIR}/3rdparty/dlpack/include/
    ${TVM_DIR}/3rdparty/dmlc-core/include/
  )
  set(TVM_LINK_DIRS ${TVM_DIR}/build)
  set(TVM_LIBRARIES
    tvm_runtime
  )
  execute_process(COMMAND python tvm_kernels.py WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  file(GLOB TVM_GEN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tvm_gen/*.cpp
  )
  set(TORCH_SPARSE_SRCS ${TORCH_SPARSE_SRCS}
    ${TVM_GEN_SRCS}
  )
ENDIF()
link_directories(${PYTORCH_DIR}/lib ${TVM_LINK_DIRS})

add_subdirectory(pybind11)

pybind11_add_module(_torch_sparse SHARED ${TORCH_SPARSE_SRCS})
target_link_libraries(_torch_sparse PUBLIC c10 c10_cuda torch pybind11 cuda ${CUDA_LIBRARIES} torch_sparse_cuda ${TVM_LIBRARIES})

target_include_directories(_torch_sparse PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_sparse
    ${CMAKE_CURRENT_SOURCE_DIR}/blocksparse/src
    ${PYTORCH_DIR}/include
    ${PYBIND11_INCLUDE_DIR}
		${CUDA_INCLUDE_DIRS}
    ${TVM_INCLUDE_DIRS}
)


