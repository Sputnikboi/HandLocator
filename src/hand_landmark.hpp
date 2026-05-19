#pragma once
#include "types.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

class HandLandmarkDetector {
public:
    explicit HandLandmarkDetector(const std::string& model_path,
                                   int   gpu_device_id      = 1,
                                   float presence_threshold = 0.6f);

    HandLandmarks detect(const cv::Mat& bgr_frame, const PalmDetection& palm);

private:
    Ort::Env           env_;
    Ort::Session       session_{nullptr};
    Ort::MemoryInfo    mem_info_;

    float presence_thresh_;
    static constexpr int kInputW = 224;
    static constexpr int kInputH = 224;

    std::vector<float> input_buf_;

    cv::Mat build_transform(const PalmDetection& palm);
};
