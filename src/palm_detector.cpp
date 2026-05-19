#include "palm_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <cmath>
#include <algorithm>
#include <stdexcept>

PalmDetector::PalmDetector(const std::string& model_path,
                             int   gpu_device_id,
                             float score_threshold,
                             float nms_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "PalmDetector")
    , mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , score_thresh_(score_threshold)
    , nms_thresh_(nms_threshold)
{
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetIntraOpNumThreads(1); // single-thread; pipeline parallelism is at frame level

    if (gpu_device_id >= 0) {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = gpu_device_id;
        opts.AppendExecutionProvider_CUDA(cuda);
    }

    session_ = Ort::Session(env_, model_path.c_str(), opts);
    generate_anchors();
    input_buf_.resize(3 * kInputH * kInputW);
}

void PalmDetector::generate_anchors() {
    // BlazePalm-lite: strides [8,16,16,16], 2 anchors/location, 192×192 → 2016 anchors
    const int strides[] = {8, 16, 16, 16};
    anchors_.reserve(2016);
    for (int stride : strides) {
        int cols = (kInputW + stride - 1) / stride;
        int rows = (kInputH + stride - 1) / stride;
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < cols; ++x)
                for (int a = 0; a < 2; ++a) // 2 anchors per location
                    anchors_.push_back({(x + 0.5f) / cols, (y + 0.5f) / rows});
    }
}

float PalmDetector::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<PalmDetection> PalmDetector::detect(const cv::Mat& bgr_frame, int max_hands) {
    // Resize → RGB float32, normalize to [-1, 1], lay out as NHWC
    cv::Mat resized;
    cv::resize(bgr_frame, resized, {kInputW, kInputH});
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32FC3, 1.0 / 127.5, -1.0);

    // NHWC: [B=1, H=192, W=192, C=3] — contiguous after convertTo
    std::memcpy(input_buf_.data(), resized.ptr<float>(),
                kInputH * kInputW * 3 * sizeof(float));

    std::array<int64_t,4> shape{1, kInputH, kInputW, 3};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_, input_buf_.data(), input_buf_.size(),
        shape.data(), shape.size());

    // Query input/output names from the model
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name  = session_.GetInputNameAllocated(0, alloc);
    const char* in_names[]  = {in_name.get()};

    int n_out = static_cast<int>(session_.GetOutputCount());
    std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
    std::vector<const char*>             out_names;
    for (int i = 0; i < n_out; ++i) {
        out_name_ptrs.push_back(session_.GetOutputNameAllocated(i, alloc));
        out_names.push_back(out_name_ptrs.back().get());
    }

    auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
        in_names, &input_tensor, 1,
        out_names.data(), out_names.size());

    // Identify scores [N,1] and boxes [N,18] by output shape
    const float* scores_ptr = nullptr;
    const float* boxes_ptr  = nullptr;
    int n_anchors = 0;
    int box_stride = 18; // values per anchor in box tensor

    static bool first_run = true;
    for (auto& t : output_tensors) {
        auto info  = t.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        int last_dim = static_cast<int>(shape.back());
        if (first_run) {
            std::cerr << "[palm debug] output shape:";
            for (auto d : shape) std::cerr << " " << d;
            std::cerr << "\n";
        }
        if (last_dim == 1) {
            scores_ptr = t.GetTensorData<float>();
            n_anchors  = static_cast<int>(info.GetElementCount());
        } else {
            boxes_ptr  = t.GetTensorData<float>();
            box_stride = last_dim;
        }
    }

    if (first_run && scores_ptr) {
        float max_raw = *std::max_element(scores_ptr, scores_ptr + n_anchors);
        std::cerr << "[palm debug] anchors=" << n_anchors
                  << "  max_raw_score=" << max_raw
                  << "  sigmoid(max)=" << sigmoid(max_raw) << "\n";
        first_run = false;
    }

    if (!scores_ptr || !boxes_ptr)
        throw std::runtime_error("Palm detector: unexpected output shapes.");

    auto dets = decode(scores_ptr, boxes_ptr, n_anchors, box_stride, bgr_frame.cols, bgr_frame.rows);

    // NMS via OpenCV helper
    std::vector<cv::Rect> rects;
    std::vector<float>    scores;
    rects.reserve(dets.size());
    scores.reserve(dets.size());
    for (auto& d : dets) {
        rects.push_back({
            static_cast<int>(d.box.x      * bgr_frame.cols),
            static_cast<int>(d.box.y      * bgr_frame.rows),
            static_cast<int>(d.box.width  * bgr_frame.cols),
            static_cast<int>(d.box.height * bgr_frame.rows)});
        scores.push_back(d.score);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(rects, scores, score_thresh_, nms_thresh_, keep);

    std::vector<PalmDetection> result;
    for (int idx : keep) {
        if (static_cast<int>(result.size()) >= max_hands) break;
        result.push_back(dets[idx]);
    }
    return result;
}

std::vector<PalmDetection> PalmDetector::decode(const float* scores,
                                                  const float* boxes,
                                                  int n_anchors,
                                                  int box_stride,
                                                  int frame_w, int frame_h)
{
    std::vector<PalmDetection> dets;
    for (int i = 0; i < n_anchors; ++i) {
        float score = sigmoid(scores[i]);
        if (score < score_thresh_) continue;

        const float* b = boxes + i * box_stride;
        float ax = anchors_[i].x;
        float ay = anchors_[i].y;

        float cx = b[0] / kInputW + ax;
        float cy = b[1] / kInputH + ay;
        float w  = b[2] / kInputW;
        float h  = b[3] / kInputH;

        PalmDetection d;
        d.score = score;
        d.box   = {cx - w/2, cy - h/2, w, h};

        for (int k = 0; k < 7; ++k) {
            d.keypoints[k].x = b[4 + k*2 + 0] / kInputW + ax;
            d.keypoints[k].y = b[4 + k*2 + 1] / kInputH + ay;
        }

        // Rotation: wrist (kp[0]) → middle-finger MCP (kp[2]), target angle = π/2
        float dx = d.keypoints[2].x - d.keypoints[0].x;
        float dy = d.keypoints[2].y - d.keypoints[0].y;
        d.rotation = static_cast<float>(M_PI_2) - std::atan2(-dy, dx);
        while (d.rotation >  M_PI) d.rotation -= 2*M_PI;
        while (d.rotation < -M_PI) d.rotation += 2*M_PI;

        float palm_cx = (d.box.x + d.box.width/2)  * frame_w;
        float palm_cy = (d.box.y + d.box.height/2) * frame_h;
        float long_px = std::max(d.box.width * frame_w, d.box.height * frame_h) * 2.6f;
        d.roi_cx   = palm_cx + long_px * 0.5f * std::sin(d.rotation);
        d.roi_cy   = palm_cy - long_px * 0.5f * std::cos(d.rotation);
        d.roi_size = long_px;

        dets.push_back(d);
    }
    return dets;
}
