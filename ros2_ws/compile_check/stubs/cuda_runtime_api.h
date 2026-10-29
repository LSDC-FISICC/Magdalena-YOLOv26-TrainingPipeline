// Minimal CUDA runtime API surface used by model.cpp, for compile checks without CUDA.
#pragma once
#include <cstddef>
typedef int cudaError_t; typedef struct CUstream_st* cudaStream_t;
enum { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };
cudaError_t cudaMalloc(void**, std::size_t); cudaError_t cudaFree(void*);
cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaStreamCreate(cudaStream_t*); cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
