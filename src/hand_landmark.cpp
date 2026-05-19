#include "hand_landmark.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>

HandLandmarkDetector::HandLandmarkDetector(const std::string& model_path,
                                             int   gpu_device_id,
                                             float presence_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "HandLandmark")
    , mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , presence_thresh_(presence_threshold)
{
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetIntraOpNumThreads(1);

    if (gpu_device_id >= 0) {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = gpu_device_id;
        opts.AppendExecutionProvider_CUDA(cuda);
    }

    session_ = Ort::Session(env_, model_path.c_str(), opts);
    input_buf_.resize(3 * kInputH * kInputW);
}

cv::Mat HandLandmarkDetector::build_transform(const PalmDetection& palm) {
    float scale = static_cast<float>(kInputW) / palm.roi_size;
    float ca = std::cos(-palm.rotation);
    float sa = std::sin(-palm.rotation);

    double m[6] = {
        scale * ca, scale * (-sa),
        kInputW/2.0 - scale*(ca*palm.roi_cx - sa*palm.roi_cy),
        scale * sa, scale * ca,
        kInputH/2.0 - scale*(sa*palm.roi_cx + ca*palm.roi_cy)
    };
    return cv::Mat(2, 3, CV_64F, m).clone();
}

HandLandmarks HandLandmarkDetector::detect(const cv::Mat& bgr_frame,
                                            const PalmDetection& palm)
{
    HandLandmarks result{};
    result.valid = false;

    // Warp ROI to 224×224
    cv::Mat M = build_transform(palm);
    cv::Mat crop;
    cv::warpAffine(bgr_frame, crop, M, {kInputW, kInputH},
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    // Normalize [0,1], BGR→RGB, NCHW (tflite2onnx adds a transpose op)
    cv::Mat fcrop;
    cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
    crop.convertTo(fcrop, CV_32FC3, 1.0 / 255.0);
    cv::Mat ch[3];
    cv::split(fcrop, ch);
    float* dst = input_buf_.data();
    for (int c = 0; c < 3; ++c) {
        std::memcpy(dst, ch[c].ptr<float>(), kInputH * kInputW * sizeof(float));
        dst += kInputH * kInputW;
    }

    std::array<int64_t,4> shape{1, 3, kInputH, kInputW};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_, input_buf_.data(), input_buf_.size(),
        shape.data(), shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = session_.GetInputNameAllocated(0, alloc);
    const char* in_names[] = {in_name.get()};

    int n_out = static_cast<int>(session_.GetOutputCount());
    std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
    std::vector<const char*>             out_names;
    for (int i = 0; i < n_out; ++i) {
        out_name_ptrs.push_back(session_.GetOutputNameAllocated(i, alloc));
        out_names.push_back(out_name_ptrs.back().get());
    }

    auto outs = session_.Run(Ort::RunOptions{nullptr},
        in_names, &input_tensor, 1,
        out_names.data(), out_names.size());

    // Route outputs by index (verified model layout):
    //   out[0] = Identity  : [1,63] screen landmarks in pixel coords (0–224)
    //   out[1] = Identity_1: [1,1]  presence logit
    //   out[2] = Identity_2: [1,1]  handedness logit
    //   out[3] = Identity_3: [1,63] world landmarks (near zero, unused)
    const float* lm_data    = outs[0].GetTensorData<float>();
    const float* world_data = outs[3].GetTensorData<float>();
    float presence   = 1.f / (1.f + std::exp(-outs[1].GetTensorData<float>()[0]));
    float handedness = 1.f / (1.f + std::exp(-outs[2].GetTensorData<float>()[0]));

    if (!lm_data) return result;

    result.presence  = presence;
    result.handedness = handedness;
    if (presence < presence_thresh_) return result;

    // Invert affine → original frame coordinates
    cv::Mat M_inv;
    cv::invertAffineTransform(M, M_inv);

    for (int i = 0; i < 21; ++i) {
        float lx = lm_data[i*3 + 0];   // already in 224×224 pixel space
        float ly = lm_data[i*3 + 1];
        float lz = lm_data[i*3 + 2];
        float ox = static_cast<float>(
            M_inv.at<double>(0,0)*lx + M_inv.at<double>(0,1)*ly + M_inv.at<double>(0,2));
        float oy = static_cast<float>(
            M_inv.at<double>(1,0)*lx + M_inv.at<double>(1,1)*ly + M_inv.at<double>(1,2));
        result.landmarks[i] = {ox, oy, lz};
    }

    // Copy world-space landmarks (Identity_3)
    for (int i = 0; i < 21; ++i) {
        result.world_landmarks[i] = {
            world_data[i*3 + 0],
            world_data[i*3 + 1],
            world_data[i*3 + 2]
        };
    }

    result.valid = true;
    return result;
}
