#include "yolov5.h"

#include <sys/stat.h>
#include <unistd.h>

#include "mmcv_nms.h"

using namespace ppl::common;
using namespace ppl::nn;

Yolov5Impl::Yolov5Impl(const ModelParams model_params){
    this->model_params.yolov5_height  = model_params.yolov5_height;
    this->model_params.yolov5_width   = model_params.yolov5_width;
    this->model_params.yolov5_channel = model_params.yolov5_channel;

    this->model_params.num_classes    = model_params.num_classes;
    this->model_params.onnx_path      = model_params.onnx_path;
    this->model_params.prob_threshold = model_params.prob_threshold;
    this->model_params.nms_threshold  = model_params.nms_threshold;

    memcpy(this->model_params.mean, model_params.mean, 3*sizeof(float));
    memcpy(this->model_params.std, model_params.std, 3*sizeof(float));
}

RetCode Yolov5Impl::yolov5_network_detect_init(){
    
    if (access( model_params.onnx_path, F_OK ) == -1){
        fprintf(stderr, "Not found %s \n", model_params.onnx_path);
        return RC_INVALID_VALUE;
    }

    in_data = (float*)malloc(model_params.yolov5_height * model_params.yolov5_width * model_params.yolov5_channel*sizeof(float));
    if (in_data == NULL)
        return RC_INVALID_VALUE;

    // create runtime builder onnx model
    Engine* x86_engine = X86EngineFactory::Create(X86EngineOptions());
    RuntimeBuilder* builder = OnnxRuntimeBuilderFactory::Create(model_params.onnx_path, &x86_engine, 1);

    if (!builder){
        fprintf(stderr, "create RuntimeBuilder from onnx model %s failed!\n", model_params.onnx_path);
        return RC_INVALID_VALUE;
    }

    printf("successfully create runtime builder!\n");

    context.reset(builder->CreateRuntime());
    if (!context) {
        fprintf(stderr, "build runtime failed!\n");
        return RC_INVALID_VALUE;
    }
    
    printf("successfully build runtime!\n");

    input_tensor = std::shared_ptr<Tensor>(context->GetInputTensor(0));
    const std::vector<int64_t> input_shape{1, model_params.yolov5_channel, model_params.yolov5_height, model_params.yolov5_width};
    input_tensor->GetShape().Reshape(input_shape);
    auto status = input_tensor->ReallocBuffer();
    if (status != RC_SUCCESS){
        fprintf(stderr, "ReallocBuffer for tensor [%s] failed: %s\n", input_tensor->GetName(), GetRetCodeStr(status));

        return RC_INVALID_VALUE;
    }

    return RC_SUCCESS;
}

RetCode Yolov5Impl::preprocess(cv::Mat& src, float* in_data){
    if (src.empty() || in_data == NULL)
        return RC_INVALID_VALUE;


    // split 3 channel to change HWC to CHW
    std::vector<cv::Mat> rgb_channels(3);
    cv::split(src, rgb_channels);

    // by this constructor, when cv::Mat r_channel_fp32 changed, in_data will also change
    cv::Mat r_channel_fp32(640, 640, CV_32FC1, in_data + 0 * 640 * 640);
    cv::Mat g_channel_fp32(640, 640, CV_32FC1, in_data + 1 * 640 * 640);
    cv::Mat b_channel_fp32(640, 640, CV_32FC1, in_data + 2 * 640 * 640);
    std::vector<cv::Mat> rgb_channels_fp32{r_channel_fp32, g_channel_fp32, b_channel_fp32};

    // convert uint8 to fp32, y = (x - mean) / std
    // const float mean[3] = {0, 0, 0}; // change mean & std according to your dataset & training param
    std::cout << model_params.std[0] << std::endl; 
    for (uint32_t i = 0; i < rgb_channels.size(); ++i) {
        rgb_channels[i].convertTo(rgb_channels_fp32[i], CV_32FC1, 1.0f / model_params.std[i], -model_params.mean[i] / model_params.std[i]);
    }

    return RC_SUCCESS;
}

RetCode Yolov5Impl::yolov5_network_detect(cv::Mat& src, std::vector<DetectRes>& detect_res) {
    if (src.empty() || context == NULL)
        return RC_INVALID_VALUE;

    const int width   = model_params.yolov5_width;
    const int height  = model_params.yolov5_width;
    const int channel = model_params.yolov5_channel;

    // float* in_data = (float*)malloc(width*height*channel*sizeof(float));
    if (in_data == NULL)
        return RC_INVALID_VALUE;
    
    RetCode retcode = preprocess(src, in_data);

    // set model input data
    // set input data descriptor
    TensorShape src_desc = input_tensor->GetShape(); // description of your prepared data, not input tensor's description
    src_desc.SetDataType(DATATYPE_FLOAT32);
    src_desc.SetDataFormat(DATAFORMAT_NDARRAY); // for 4-D Tensor, NDARRAY == NCHW


    retcode = input_tensor->ConvertFromHost(in_data, src_desc); // convert data type & format from src_desc to input_tensor & fill data
    if (retcode != RC_SUCCESS) {
        fprintf(stderr, "set input data to tensor [%s] failed: %s\n", input_tensor->GetName(), GetRetCodeStr(retcode));
        return RC_INVALID_VALUE;
    }

    printf("successfully set input data to tensor [%s]!\n", input_tensor->GetName());

    // forward
    retcode = context->Run(); // forward
    if (retcode != RC_SUCCESS) {
        fprintf(stderr, "run network failed: %s\n", GetRetCodeStr(retcode));
        return RC_INVALID_VALUE;
    }

    retcode = context->Sync(); // wait for all ops run finished, not implemented yet.
    if (retcode != RC_SUCCESS) { // now sync is done by runtime->Run() function.
        fprintf(stderr, "runtime sync failed: %s\n", GetRetCodeStr(retcode));
        return RC_INVALID_VALUE;
    }

    printf("successfully run network!\n");

    postprecess(detect_res);
}

RetCode Yolov5Impl::postprecess(std::vector<DetectRes>& detect_res){
    if (context == NULL)
        return RC_INVALID_VALUE; 

    std::vector<DetectRes> proposals;

    // stride 8
    {
        auto output_tensor = context->GetOutputTensor(0);
        uint64_t output_size = output_tensor->GetShape().GetElementsExcludingPadding();
        std::vector<float> output_data_(output_size);
        float* output_data = output_data_.data();

        // set output data descriptor
        TensorShape dst_desc = output_tensor->GetShape(); // description of your output data buffer, not output_tensor's description
        dst_desc.SetDataType(DATATYPE_FLOAT32);
        dst_desc.SetDataFormat(DATAFORMAT_NDARRAY); // output is 1-D Tensor, NDARRAY == vector

        auto status = output_tensor->ConvertToHost(output_data, dst_desc); // convert data type & format from output_tensor to dst_desc
        if (status != RC_SUCCESS) {
            fprintf(stderr, "get output data from tensor [%s] failed: %s\n", output_tensor->GetName(),
                    GetRetCodeStr(status));
            return -1;
        }

        printf("successfully get outputs!\n");

        std::cout << output_size << std::endl;
        
        std::vector<float> anchor = {10.f, 13.f, 16.f, 30.f, 33.f, 23.f};
        std::vector<DetectRes> proposals8;
        generate_proposals(anchor, 80, 80, 8, (float*)(output_tensor->GetBufferPtr()), 
                          model_params.prob_threshold, model_params.num_classes, proposals8);

        proposals.insert(proposals.end(), proposals8.begin(), proposals8.end());
    }

    // stride 16
    {
        auto output_tensor = context->GetOutputTensor(1);
        uint64_t output_size = output_tensor->GetShape().GetElementsExcludingPadding();
        std::vector<float> output_data_(output_size);
        float* output_data = output_data_.data();

        // set output data descriptor
        TensorShape dst_desc = output_tensor->GetShape(); // description of your output data buffer, not output_tensor's description
        dst_desc.SetDataType(DATATYPE_FLOAT32);
        dst_desc.SetDataFormat(DATAFORMAT_NDARRAY); // output is 1-D Tensor, NDARRAY == vector

        auto status = output_tensor->ConvertToHost(output_data, dst_desc); // convert data type & format from output_tensor to dst_desc
        if (status != RC_SUCCESS) {
            fprintf(stderr, "get output data from tensor [%s] failed: %s\n", output_tensor->GetName(),
                    GetRetCodeStr(status));
            return -1;
        }

        printf("successfully get outputs!\n");

        std::cout << output_size << std::endl;
        
        std::vector<float> anchor = {30.f, 61.f, 62.f, 45.f, 59.f, 119.f};
        std::vector<DetectRes> proposals16;
        generate_proposals(anchor, 40, 40, 16, (float*)(output_tensor->GetBufferPtr()), 
                          model_params.prob_threshold, model_params.num_classes, proposals16);

        proposals.insert(proposals.end(), proposals16.begin(), proposals16.end());
    }

     // stride 32
    {
        auto output_tensor = context->GetOutputTensor(2);
        uint64_t output_size = output_tensor->GetShape().GetElementsExcludingPadding();
        std::vector<float> output_data_(output_size);
        float* output_data = output_data_.data();

        // set output data descriptor
        TensorShape dst_desc = output_tensor->GetShape(); // description of your output data buffer, not output_tensor's description
        dst_desc.SetDataType(DATATYPE_FLOAT32);
        dst_desc.SetDataFormat(DATAFORMAT_NDARRAY); // output is 1-D Tensor, NDARRAY == vector

        auto status = output_tensor->ConvertToHost(output_data, dst_desc); // convert data type & format from output_tensor to dst_desc
        if (status != RC_SUCCESS) {
            fprintf(stderr, "get output data from tensor [%s] failed: %s\n", output_tensor->GetName(),
                    GetRetCodeStr(status));
            return -1;
        }

        printf("successfully get outputs!\n");
        std::cout << output_size << std::endl;
        
        std::vector<float> anchor = {116.f, 90.f, 156.f, 198.f, 373.f, 326.f};
        std::vector<DetectRes> proposals32;
        generate_proposals(anchor, 20, 20, 32, (float*)(output_tensor->GetBufferPtr()), 
                          model_params.prob_threshold, model_params.num_classes, proposals32);

        proposals.insert(proposals.end(), proposals32.begin(), proposals32.end());
    }

    //nms
    {
        float* bbox_vec = (float*)malloc(proposals.size()*4*sizeof(float));
        float* scores_vec = (float*)malloc(proposals.size()*sizeof(float));

        for (size_t i = 0; i < proposals.size(); ++i) {
            bbox_vec[i*4 + 0] = proposals[i].x_min;
            bbox_vec[i*4 + 1] = proposals[i].y_min;
            bbox_vec[i*4 + 2] = proposals[i].x_max;
            bbox_vec[i*4 + 3] = proposals[i].y_max;

            scores_vec[i] = proposals[i].prob;
        }

        int64_t* keep_index = (int64_t*)malloc(proposals.size()*sizeof(int16_t));
        int64_t num_keep_box = 0;
        mmcv_nms_ndarray_fp32(bbox_vec, scores_vec, proposals.size(), 
                            model_params.nms_threshold, 4, 
                            keep_index, &num_keep_box);

        std::cout << "num_keep_box: " << num_keep_box << std::endl;

        for (size_t i = 0; i < num_keep_box; ++i) {
            detect_res.push_back(proposals[keep_index[i]]);
        }

        if (bbox_vec) {
            free(bbox_vec);
            bbox_vec = NULL;
        }

        if (scores_vec) {
            free(scores_vec);
            scores_vec = NULL;
        }
 
        if (keep_index) {
            free(keep_index);
            keep_index = NULL;
        }
    }

    //detect_res = proposals;
}

Yolov5Impl::~Yolov5Impl(){
    context.reset();
    input_tensor.reset();

    if (in_data) {
        free(in_data);
        in_data = NULL;
    }
}
