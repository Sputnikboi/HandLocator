#pragma once
#include "types.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// YOLOv8-based hand detector — drop-in replacement for PalmDetector.
// Produces PalmDetection structs so the downstream HandLandmarkDetector
// continues to work unchanged.
class HandDetector {
public:
    // gpu_device_id: pass -1 for CPU, 0/1/... for a specific CUDA device.
    explicit HandDetector(const std::string& model_path,
                          int   gpu_device_id    = 1,
                          float score_threshold  = 0.5f,
                          float nms_threshold    = 0.45f);

    std::vector<PalmDetection> detect(const cv::Mat& bgr_frame, int max_hands = 2);

private:
    Ort::Env           env_;
    Ort::Session       session_{nullptr};
    Ort::MemoryInfo    mem_info_;

    float score_thresh_;
    float nms_thresh_;

    int   input_w_ = 0;
    int   input_h_ = 0;

    std::vector<float> input_buf_;

    // Pre-processing produces letterboxed image. Store the transform params.
    float scale_   = 1.f;
    float pad_x_   = 0.f;
    float pad_y_   = 0.f;

    void preprocess(const cv::Mat& bgr_frame);
    std::vector<PalmDetection> postprocess(const float* data, int n_detections,
                                            int frame_w, int frame_h);
};
