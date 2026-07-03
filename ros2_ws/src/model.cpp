# include "model.hpp"

model::model(const std::string& model_path, int batch_size) : m_maxBatchSize(batch_size) {
    //pour debug visuel
    cv::namedWindow("Debug Preprocess", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Debug Preprocess2", cv::WINDOW_AUTOSIZE);

    std::vector<char> engineData = loadEnginePath(model_path);

    // 2. Initialisation des composants d'inférence TensorRT
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime) throw std::runtime_error("Failed to create TensorRT InferRuntime");

    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!m_engine) throw std::runtime_error("Failed to deserialize CUDA Engine");

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) throw std::runtime_error("Failed to create TensorRT ExecutionContext");

    // 3. Allocation des buffers sur le GPU et le CPU à la taille maximale (ex: batch de 5)
    size_t maxInputSize = m_maxBatchSize * m_channels * m_imgH * m_imgW * sizeof(float);
    size_t maxOutputSize = m_maxBatchSize * m_numClasses * sizeof(float);

    if (cudaMalloc(&m_deviceInput, maxInputSize) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA device input memory");
    }
    if (cudaMalloc(&m_deviceOutput, maxOutputSize) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA device output memory");
    }

    m_hostInput.resize(maxInputSize / sizeof(float));
    m_hostOutput.resize(maxOutputSize / sizeof(float));

    // 4. Création du stream CUDA
    if (cudaStreamCreate(&m_stream) != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }
}


model::~model() {
    if (m_stream) cudaStreamDestroy(m_stream);
    if (m_deviceInput) cudaFree(m_deviceInput);
    if (m_deviceOutput) cudaFree(m_deviceOutput);
}


std::vector<char> model::loadEnginePath(const std::string& filename){

    std::ifstream file(filename, std::ios::binary); 
    if (!file.good()){throw std::runtime_error("Failed to open .engine path :" + filename);}

    //std::streamsize size = file.tellg();
    //file.seekg(0, std::ios::beg);

    std::cout << "is_open: " << file.is_open() << std::endl;
    std::cout << "good: " << file.good() << std::endl;

    uint32_t jsonLength = 0;
    file.read(reinterpret_cast<char*>(&jsonLength), sizeof(jsonLength));

    std::cout << "after read -> good: " << file.good()
              << " eof: " << file.eof()
              << " fail: " << file.fail()
              << " bad: " << file.bad()
              << " gcount: " << file.gcount()
              << " jsonLength: " << jsonLength << std::endl;

    if (!file){
        throw std::runtime_error("Failed to read JSON header length: " + filename);
    }

    // Optionnel : lire et afficher le JSON pour vérifier (stride, names, imgsz...)
    std::string jsonHeader(jsonLength, '\0');
    file.read(jsonHeader.data(), jsonLength);
    if (!file){
        throw std::runtime_error("Failed to read JSON header content: " + filename);
    }
    std::cout << "[Engine metadata] " << jsonHeader << std::endl;

    //std::vector<char> buffer(size); 
    //if (!file.read(buffer.data(), size)){
    //    throw std::runtime_error("Failed to read .engine file" + filename);
    //}
    std::vector<char> buffer(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    if (buffer.empty()){
        throw std::runtime_error("No TensorRT plan data found after JSON header: " + filename);
    }

    return buffer;
}

void model::preprocessImage_from_path(const std::string& imgpath, int batchIdx){
    cv::Mat img = cv::imread(imgpath, cv::IMREAD_COLOR);
    if (img.empty()){
        throw std::runtime_error("Failed to read the image" + imgpath);
    }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(320, 96));

    cv::Mat imgFloat; 
    resized.convertTo(imgFloat, CV_32FC3, 1.0f/ 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(imgFloat, channels); 

    size_t channelLength = 96*320;
    size_t batchOffset = batchIdx * m_channels * channelLength;
    std::memcpy(m_hostInput.data() + batchOffset, channels[2].data, channelLength * sizeof(float)); 
    std::memcpy(m_hostInput.data()+ batchOffset + channelLength, channels[1].data, channelLength * sizeof(float));
    std::memcpy(m_hostInput.data() + batchOffset + 2*channelLength, channels[0].data, channelLength * sizeof(float));
}



void model::preprocessImage(const cv::Mat& img, int batchIdx){
    if (img.empty()) {
        throw std::runtime_error("Empty cv::Mat received during preprocessing at index " + std::to_string(batchIdx));
    }

    int height = img.rows;
    int weights = img.cols;
    int x_start = weights/2 - 160;
    int x_end = weights/2 +160;
    int y_start = height/2 -48;
    int y_end = height/2 +48;
    cv::Rect rectangle(x_start, y_start, x_end, y_end);

    cv::Mat img_rect = img(rectangle);

    //pour le debug visuel
    cv::imshow("Debug Preprocess", img_rect);
    cv::waitKey(1);

    cv::Mat resized;
    cv::resize(img_rect, resized, cv::Size(320, 96));

    cv::Mat imgFloat; 
    resized.convertTo(imgFloat, CV_32FC3, 1.0f/ 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(imgFloat, channels); 

    size_t channelLength = 96*320;
    size_t batchOffset = batchIdx * m_channels * channelLength;
    std::memcpy(m_hostInput.data() + batchOffset, channels[2].data, channelLength * sizeof(float)); 
    std::memcpy(m_hostInput.data()+ batchOffset + channelLength, channels[1].data, channelLength * sizeof(float));
    std::memcpy(m_hostInput.data() + batchOffset + 2*channelLength, channels[0].data, channelLength * sizeof(float));
}

std::vector<InferenceResult> model::analyzeBatch(const std::vector<cv::Mat>& images){
    
    int currentBatchSize = static_cast<int>(images.size());
    if (currentBatchSize == 0) return {};
    if (currentBatchSize > m_maxBatchSize) {
        throw std::runtime_error("Requested batch size (" + std::to_string(currentBatchSize) + 
                                 ") exceeds maximum configured batch size (" + std::to_string(m_maxBatchSize) + ")");
    }

    // 1. Prétraitement séquentiel de chaque image reçue dans le vecteur
    for (int i = 0; i < currentBatchSize; ++i) {
        preprocessImage(images[i], i);
    }

    // 2. Calcul des tailles de données exactes pour la taille du batch actuel
    size_t actualInputSize = currentBatchSize * m_channels * m_imgH * m_imgW * sizeof(float);
    size_t actualOutputSize = currentBatchSize * m_numClasses * sizeof(float);

    // 3. Copie asynchrone Host -> Device (CPU vers GPU)
    cudaMemcpyAsync(m_deviceInput, m_hostInput.data(), actualInputSize, cudaMemcpyHostToDevice, m_stream);

    // 4. Configuration des tenseurs d'entrée/sortie pour l'API TensorRT V3
    m_context->setTensorAddress("images", m_deviceInput);
    m_context->setTensorAddress("output0", m_deviceOutput);

    // Ajustement dynamique de la forme d'entrée selon le nombre d'images (1 à 5)
    nvinfer1::Dims4 inputDims{currentBatchSize, m_channels, m_imgH, m_imgW};
    if (!m_context->setInputShape("images", inputDims)) {
        throw std::runtime_error("TensorRT failed to set input shape for batch size: " + std::to_string(currentBatchSize));
    }

    // Execution de l'inférence sur le GPU
    m_context->enqueueV3(m_stream);

    // 5. Copie asynchrone Device -> Host (GPU vers CPU) & Synchronisation du stream
    cudaMemcpyAsync(m_hostOutput.data(), m_deviceOutput, actualOutputSize, cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    // 6. Post-processing et extraction des scores de classes pour chaque image du batch
    std::vector<InferenceResult> results;
    results.reserve(currentBatchSize);

    for (int i = 0; i < currentBatchSize; ++i) {
        // Décalage pour atteindre le début des outputs de l'image 'i'
        int offset = i * m_numClasses;
        float class0_conf = m_hostOutput[offset];     // Ex: "no_plant"
        float class1_conf = m_hostOutput[offset + 1]; // Ex: "plant"

        int predictClass = (class1_conf > class0_conf) ? 1 : 0;
        float confidence = (predictClass == 1) ? class1_conf : class0_conf;

        results.push_back({predictClass, confidence});
    }

    return results;

}

std::vector<InferenceResult> model::analyzeBatch_preprocessed(const std::vector<cv::Mat>& images){
    
    int currentBatchSize = static_cast<int>(images.size());
    if (currentBatchSize == 0) return {};
    if (currentBatchSize > m_maxBatchSize) {
        throw std::runtime_error("Requested batch size (" + std::to_string(currentBatchSize) + 
                                 ") exceeds maximum configured batch size (" + std::to_string(m_maxBatchSize) + ")");
    }

    // 1. Prétraitement séquentiel de chaque image reçue dans le vecteur
    for (int i = 0; i < currentBatchSize; ++i) {
            cv::imshow("Debug Preprocess", images[i]);
            cv::waitKey(1);

            std::vector<cv::Mat> channels(3);
            cv::split(images[i], channels); 

            size_t channelLength = 96*320;
            size_t batchOffset = i * m_channels * channelLength;
            std::memcpy(m_hostInput.data() + batchOffset, channels[2].data, channelLength * sizeof(float)); 
            std::memcpy(m_hostInput.data()+ batchOffset + channelLength, channels[1].data, channelLength * sizeof(float));
            std::memcpy(m_hostInput.data() + batchOffset + 2*channelLength, channels[0].data, channelLength * sizeof(float));
    }



    // 2. Calcul des tailles de données exactes pour la taille du batch actuel
    size_t actualInputSize = currentBatchSize * m_channels * m_imgH * m_imgW * sizeof(float);
    size_t actualOutputSize = currentBatchSize * m_numClasses * sizeof(float);

    // 3. Copie asynchrone Host -> Device (CPU vers GPU)
    cudaMemcpyAsync(m_deviceInput, m_hostInput.data(), actualInputSize, cudaMemcpyHostToDevice, m_stream);

    // 4. Configuration des tenseurs d'entrée/sortie pour l'API TensorRT V3
    m_context->setTensorAddress("images", m_deviceInput);
    m_context->setTensorAddress("output0", m_deviceOutput);

    // Ajustement dynamique de la forme d'entrée selon le nombre d'images (1 à 5)
    nvinfer1::Dims4 inputDims{currentBatchSize, m_channels, m_imgH, m_imgW};
    if (!m_context->setInputShape("images", inputDims)) {
        throw std::runtime_error("TensorRT failed to set input shape for batch size: " + std::to_string(currentBatchSize));
    }

    // Execution de l'inférence sur le GPU
    m_context->enqueueV3(m_stream);

    // 5. Copie asynchrone Device -> Host (GPU vers CPU) & Synchronisation du stream
    cudaMemcpyAsync(m_hostOutput.data(), m_deviceOutput, actualOutputSize, cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    // 6. Post-processing et extraction des scores de classes pour chaque image du batch
    std::vector<InferenceResult> results;
    results.reserve(currentBatchSize);

    for (int i = 0; i < currentBatchSize; ++i) {
        // Décalage pour atteindre le début des outputs de l'image 'i'
        int offset = i * m_numClasses;
        float class0_conf = m_hostOutput[offset];     // Ex: "no_plant"
        float class1_conf = m_hostOutput[offset + 1]; // Ex: "plant"

        int predictClass = (class1_conf > class0_conf) ? 1 : 0;
        float confidence = (predictClass == 1) ? class1_conf : class0_conf;

        results.push_back({predictClass, confidence});
    }

    return results;

}










std::vector<InferenceResult> model::analyzeBatch_from_path(const std::vector<std::string>& imagePaths){

    int currentBatchSize = static_cast<int>(imagePaths.size());
    if (currentBatchSize == 0) return {};
    if (currentBatchSize > m_maxBatchSize) {
        throw std::runtime_error("Batch size exceeds maximum allowed size");
    }

    // 1. Prétraiter chaque image et l'injecter au bon endroit dans le buffer d'entrée
    for (int i = 0; i < currentBatchSize; ++i) {
        preprocessImage_from_path(imagePaths[i], i);
    }

    // 2. Recalculer les tailles réelles pour ce batch précis
    size_t actualInputSize = currentBatchSize * m_channels * m_imgH * m_imgW * sizeof(float);
    size_t actualOutputSize = currentBatchSize * m_numClasses * sizeof(float);

    // 3. Transfert Host -> Device
    cudaMemcpyAsync(m_deviceInput, m_hostInput.data(), actualInputSize, cudaMemcpyHostToDevice, m_stream);

    // 4. Configuration TensorRT (Execution en V3)
    m_context->setTensorAddress("images", m_deviceInput);
    m_context->setTensorAddress("output0", m_deviceOutput);

    nvinfer1::Dims4 inputDims{currentBatchSize, m_channels, m_imgH, m_imgW};
    if (!m_context->setInputShape("images", inputDims)) {
        throw std::runtime_error("Failed to set input shape");
    }

    // Inference
    m_context->enqueueV3(m_stream);

    // 5. Transfert Device -> Host & Synchro
    cudaMemcpyAsync(m_hostOutput.data(), m_deviceOutput, actualOutputSize, cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    // 6. Post-processing des résultats pour chaque image du batch
    std::vector<InferenceResult> results;
    for (int i = 0; i < currentBatchSize; ++i) {
        int offset = i * m_numClasses;
        float class0_conf = m_hostOutput[offset];
        float class1_conf = m_hostOutput[offset + 1];

        int predictClass = (class1_conf > class0_conf) ? 1 : 0;
        float confidence = (predictClass == 1) ? class1_conf : class0_conf;

        results.push_back({predictClass, confidence});
    }

    return results;
}
