#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

// Logger TensorRT
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

// Struct to save the results 
struct InferenceResult {
    int predictedClass;
    float confidence;
};

class model {
    public: 
    model(const std::string& model_path, int batch_size = 5); 
    ~model();

    //Function to analyse a Dataset
    std::vector<InferenceResult> analyzeBatch_from_path(const std::vector<std::string>& imagePaths);
    //Function to analyse a batch by doing the prepocessing+
    std::vector<InferenceResult> analyzeBatch(const std::vector<cv::Mat>& images);
    //Function to analyse and already preprocessed batch
    std::vector<InferenceResult> analyzeBatch_preprocessed(const std::vector<cv::Mat>& images);

    int get_batch_size() const {return m_maxBatchSize;}


    private :

    //Function to load the Engine for its path
    std::vector<char> loadEnginePath(const std::string& filename);
    //Function to preprocess an image from a batch of paths
    void preprocessImage_from_path(const std::string& imgpath, int batchIdx);
    //Function to preprocess a cv::mat image from a batch
    void preprocessImage(const cv::Mat& img, int batchIdx);

    //parameters
    const int m_maxBatchSize = 5;
    const int m_imgW = 320; 
    const int m_imgH = 96;
    const int m_numClasses = 2;
    const int m_channels = 3;

    //TensorRT Logging System
    TRTLogger m_logger;
    // Runtime instance used to deserialize the CUDA engine (.engine file)
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> m_runtime{nullptr, [](nvinfer1::IRuntime* p) { delete p; }};
    // The core TensorRT network engine containing the optimized model architecture
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> m_engine{nullptr, [](nvinfer1::ICudaEngine* p) { delete p; }};
    // Execution context that holds intermediate activation values to run inference on the GPU
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> m_context{nullptr, [](nvinfer1::IExecutionContext* p) { delete p; }};

    //CUDA Stream & Device (GPU) Memory Allocations
    cudaStream_t m_stream;
    void* m_deviceInput = nullptr;
    void* m_deviceOutput = nullptr;
    
    //Host (CPU) Memory Buffers
    std::vector<float> m_hostInput;
    std::vector<float> m_hostOutput;


};