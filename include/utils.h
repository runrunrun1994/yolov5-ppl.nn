#ifndef __YOLOV5_PPL_NN_UTILS_H__
#define __YOLOV5_PPL_NN_UTILS_H__
#include <vector>

struct DetectRes{
    int x_min;               ///< left top x_min
    int y_min;               ///< left top y_min
    int x_max;               ///< right bottom x_max
    int y_max;               ///< right Bittom y_max
    int label;               ///< label id
    float prob;              ///< scores
};

void generate_proposals(std::vector<float> anchor,
                        const int num_grid_w,
                        const int num_grid_h,
                        const int stride,
                        const float* output,
                        float prob_threshold,
                        int num_classes,
                        std::vector<DetectRes>& detect_res);

#endif