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

// Logger TensorRT interne à la classe ou partagé
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

// Structure pour stocker le résultat d'une image
struct InferenceResult {
    int predictedClass;
    float confidence;
};

class model {
    public: 
    model(const std::string& model_path, int batch_size = 5); 
    ~model();

    std::vector<InferenceResult> analyzeBatch_from_path(const std::vector<std::string>& imagePaths);
    std::vector<InferenceResult> analyzeBatch(const std::vector<cv::Mat>& images);
    std::vector<InferenceResult> analyzeBatch_preprocessed(const std::vector<cv::Mat>& images);
    int get_batch_size() const {return m_maxBatchSize;}


    private :


    std::vector<char> loadEnginePath(const std::string& filename);
    void preprocessImage_from_path(const std::string& imgpath, int batchIdx);
    void preprocessImage(const cv::Mat& img, int batchIdx);

    const int m_maxBatchSize = 5;
    const int m_imgW = 320; 
    const int m_imgH = 96;
    const int m_numClasses = 2;
    const int m_channels = 3;

    TRTLogger m_logger;
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> m_runtime{nullptr, [](nvinfer1::IRuntime* p) { delete p; }};
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> m_engine{nullptr, [](nvinfer1::ICudaEngine* p) { delete p; }};
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> m_context{nullptr, [](nvinfer1::IExecutionContext* p) { delete p; }};

    cudaStream_t m_stream;
    void* m_deviceInput = nullptr;
    void* m_deviceOutput = nullptr;
    
    std::vector<float> m_hostInput;
    std::vector<float> m_hostOutput;


};