// Minimal TensorRT 10 API surface used by model.cpp, for compile checks without TensorRT.
#pragma once
#include <cstddef>
#include <cstdint>
namespace nvinfer1 {
class ILogger { public: enum class Severity : int32_t { kINTERNAL_ERROR, kERROR, kWARNING, kINFO, kVERBOSE };
  virtual void log(Severity severity, const char* msg) noexcept = 0; virtual ~ILogger() = default; };
struct Dims { int32_t nbDims; int64_t d[8]; };
class Dims4 : public Dims { public: Dims4(int64_t a, int64_t b, int64_t c, int64_t e) { nbDims = 4; d[0]=a; d[1]=b; d[2]=c; d[3]=e; } };
class IExecutionContext { public: virtual ~IExecutionContext() = default;
  bool setTensorAddress(const char*, void*) noexcept; bool setInputShape(const char*, const Dims&) noexcept;
  bool enqueueV3(void* stream) noexcept; };
class ICudaEngine { public: virtual ~ICudaEngine() = default; IExecutionContext* createExecutionContext() noexcept; };
class IRuntime { public: virtual ~IRuntime() = default; ICudaEngine* deserializeCudaEngine(const void*, std::size_t) noexcept; };
IRuntime* createInferRuntime(ILogger&) noexcept;
}
