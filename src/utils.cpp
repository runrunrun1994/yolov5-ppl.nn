#include "utils.h"

#include <cmath>
#include <float.h>
#include <iostream>

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

void generate_proposals(std::vector<float> anchor,
                        const int num_grid_w,
                        const int num_grid_h,
                        const int stride,
                        const float* output,
                        float prob_threshold,
                        const int num_classes,
                        std::vector<DetectRes>& detect_res) {

    const int num_anchor = anchor.size() / 2;
    const int offset = num_classes + 5;
    const int area_grid = num_grid_w * num_grid_h;


    for (int q = 0; q  < num_anchor; q++) {
        const float anchor_w = anchor[q * 2];
        const float anchor_h = anchor[q * 2 + 1];

        for (int i = 0; i < num_grid_h; i++) {
            for (int j = 0; j < num_grid_w; j++){
                // find class index with max class score
                // std::cout << "i: " << i << std::endl;
                int class_index = 0;
                float class_score = -FLT_MAX;
                int offset_xy = i * num_grid_h * offset + j * offset;

                for (int k = 0; k < num_classes; ++k) {
                    float score = output[q * area_grid * offset + offset_xy + 5 + k];
                    // std::cout << "score: " << score << std::endl;
                    if (score > class_score) {
                        class_index = k;
                        class_score = score;
                    }
                }

                // std::cout << "class_score: " << class_score << std::endl;

                float box_score = output[q * area_grid * offset + offset_xy + 4];
                // std::cout << "box_score: " << box_score << std::endl;
                float confidence = sigmoid(box_score) * sigmoid(class_score);
                // std::cout << "confidence: " << confidence << std::endl;
                // std::cout << "--------------------------" << std::endl;

                if (confidence >= prob_threshold) {
                    float dx = sigmoid(output[q * area_grid * offset + offset_xy + 0]);
                    float dy = sigmoid(output[q * area_grid * offset + offset_xy + 1]);
                    float dw = sigmoid(output[q * area_grid * offset + offset_xy + 2]);
                    float dh = sigmoid(output[q * area_grid * offset + offset_xy + 3]);

                    //std::cout << "dx: " << dx << " dy: " << dy << " dw: " << dw << " dh: " << dh << std::endl;
    
                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    //printf("pb_cx: %f pb_cy: %f pb_w: %f pb_h: %f\n", pb_cx, pb_cy, pb_w, pb_h);

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    DetectRes proposal;
                    proposal.x_min = x0;
                    proposal.y_min = y0;
                    proposal.x_max = x1;
                    proposal.y_max = y1;
                    proposal.label = class_index;
                    proposal.prob  = confidence;

                    detect_res.push_back(proposal);
                }

            }
        }
    }
}