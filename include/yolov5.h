#ifndef __YOLOV5S_PPL_NN_YOLOV5S_H__
#define __YOLOV5S_PPL_NN_YOLOV5S_H__
/**********************************************************
* \file yolov5s.h
* \brief Implement yolov5 using ppl.nn
* \date 2021-9-29
* \author runrunrun1994
***********************************************************/

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include "ppl/nn/models/onnx/onnx_runtime_builder_factory.h"
#include "ppl/nn/engines/x86/engine_factory.h"
#include "ppl/nn/engines/x86/x86_engine_options.h"
#include "utils.h"
/**
* \brief The params of yolov5 model
*/

struct ModelParams{
    int yolov5_height;       ///< height
    int yolov5_width;        ///< width
    int yolov5_channel;      ///< channel

    int num_classes;         ///< the number of classes
    char* onnx_path;         ///< the path of onnx model

    float mean[3];           ///< The mean of input image
    float std[3];            ///< The std of input image
};

/**
* \brief Detect result struct
*/

class Yolov5Impl{
    public:
        explicit Yolov5Impl(const ModelParams model_params);
        ~Yolov5Impl();

        ppl::common::RetCode yolov5_network_detect_init();

        ppl::common::RetCode yolov5_network_detect(cv::Mat& src, std::vector<DetectRes>& detect_res);

    private:
        ModelParams model_params;
        std::unique_ptr<ppl::nn::Runtime> context;
        std::shared_ptr<ppl::nn::Tensor> input_tensor;

        ppl::common::RetCode preprocess(cv::Mat& src, float* in_data);
        ppl::common::RetCode postprecess(std::vector<DetectRes>& detect_res);
};


#endif