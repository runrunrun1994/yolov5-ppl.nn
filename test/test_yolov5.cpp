#include "yolov5.h"

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]){
    ModelParams yolov5_params;
    yolov5_params.yolov5_height  = 640;
    yolov5_params.yolov5_width   = 640;
    yolov5_params.yolov5_channel = 3;
    yolov5_params.num_classes    = 80;
    yolov5_params.onnx_path      = argv[1];
    
    yolov5_params.mean[0] = 0.0f;
    yolov5_params.mean[1] = 0.0f;
    yolov5_params.mean[2] = 0.0f;
    yolov5_params.std[0] = 255.0f;
    yolov5_params.std[1] = 255.0f;
    yolov5_params.std[2] = 255.0f;

    yolov5_params.prob_threshold = 0.5;
    yolov5_params.nms_threshold  = 0.45;


    Yolov5Impl* yolov5 = new Yolov5Impl(yolov5_params);
    yolov5->yolov5_network_detect_init();

    cv::Mat image = cv::imread(argv[2]);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(640, 640));
    cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    std::vector<DetectRes> detect_res;
    yolov5->yolov5_network_detect(resized_image, detect_res);

    for (size_t i = 0; i < detect_res.size(); ++i){
        // std::cout << "Here" << std::endl;
        int x_min = static_cast<int>(detect_res[i].x_min);
        int y_min = static_cast<int>(detect_res[i].y_min);
        int x_max = static_cast<int>(detect_res[i].x_max);
        int y_max = static_cast<int>(detect_res[i].y_max);
        float score = detect_res[i].prob;
        int label = detect_res[i].label;
        printf("%d %d %d %d %f %d\n", x_min, y_min, x_max, y_max, score, label);
        cv::rectangle(resized_image,cv::Rect(x_min, y_min, x_max, y_max), cv::Scalar(0,0,255),1,1,0);
    }

    cv::imwrite("./test.jpg", resized_image);

    return 0;
}