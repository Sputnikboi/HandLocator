#pragma once
#include "types.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class PalmDetector {
public:
    // gpu_device_id: pass -1 for CPU, 0/1/... for a specific CUDA device.
    explicit PalmDetector(const std::string& model_path,
                          int   gpu_device_id    = 1,
                          float score_threshold  = 0.2f,
                          float nms_threshold    = 0.3f);

    std::vector<PalmDetection> detect(const cv::Mat& bgr_frame, int max_hands = 2);

private:
    Ort::Env           env_;
    Ort::Session       session_{nullptr};
    Ort::MemoryInfo    mem_info_;

    float score_thresh_;
    float nms_thresh_;

    static constexpr int kInputW = 192;
    static constexpr int kInputH = 192;

    std::vector<cv::Point2f> anchors_;
    std::vector<float>       input_buf_; // reused across frames

    void generate_anchors();
    std::vector<PalmDetection> decode(const float* scores, const float* boxes,
                                      int n_anchors, int box_stride,
                                      int frame_w, int frame_h);
    static float sigmoid(float x);
};
