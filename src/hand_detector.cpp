#include "hand_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

HandDetector::HandDetector(const std::string& model_path,
                           int   gpu_device_id,
                           float score_threshold,
                           float nms_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "HandDetector")
    , mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , score_thresh_(score_threshold)
    , nms_thresh_(nms_threshold)
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

    // Read input shape from model: expected NCHW [1, 3, H, W]
    auto shape = session_.GetInputTypeInfo(0)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();
    input_h_ = static_cast<int>(shape[2]);
    input_w_ = static_cast<int>(shape[3]);

    input_buf_.resize(3 * input_h_ * input_w_);
}

// ── Letterbox + normalise → CHW float buffer ────────────────────────────────
void HandDetector::preprocess(const cv::Mat& bgr_frame)
{
    int fw = bgr_frame.cols, fh = bgr_frame.rows;
    scale_ = std::min(static_cast<float>(input_w_) / fw,
                      static_cast<float>(input_h_) / fh);
    int new_w = static_cast<int>(fw * scale_);
    int new_h = static_cast<int>(fh * scale_);
    pad_x_ = (input_w_ - new_w) / 2.f;
    pad_y_ = (input_h_ - new_h) / 2.f;

    cv::Mat resized;
    cv::resize(bgr_frame, resized, {new_w, new_h});

    cv::Mat canvas(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(static_cast<int>(pad_x_),
                                    static_cast<int>(pad_y_),
                                    new_w, new_h)));

    // BGR→RGB, [0,1], HWC→CHW
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    cv::Mat frgb;
    rgb.convertTo(frgb, CV_32FC3, 1.0 / 255.0);

    cv::Mat ch[3];
    cv::split(frgb, ch);
    float* dst = input_buf_.data();
    for (int c = 0; c < 3; ++c) {
        std::memcpy(dst, ch[c].ptr<float>(), input_h_ * input_w_ * sizeof(float));
        dst += input_h_ * input_w_;
    }
}

// ── Decode YOLOv8 output & NMS ──────────────────────────────────────────────
// YOLOv8 detection output: [1, 4+num_classes, N_detections]  (transposed)
// Row layout: [cx, cy, w, h, cls0_score, cls1_score, ...]
std::vector<PalmDetection> HandDetector::postprocess(const float* data,
                                                      int n_detections,
                                                      int frame_w, int frame_h)
{
    std::vector<cv::Rect2f> boxes;
    std::vector<float>      scores;

    for (int i = 0; i < n_detections; ++i) {
        // YOLOv8 output is [4+nc, N] — columns are detections.
        // After transpose in ultralytics export it's [N, 4+nc].
        // onnxruntime returns [1, 4+nc, N] so we index as data[row * N + i].
        float cx    = data[0 * n_detections + i];
        float cy    = data[1 * n_detections + i];
        float w     = data[2 * n_detections + i];
        float h     = data[3 * n_detections + i];
        float score = data[4 * n_detections + i];  // single class (hand)

        if (score < score_thresh_) continue;

        // Remove letterbox padding and scale to original frame
        float x1 = (cx - w / 2.f - pad_x_) / scale_;
        float y1 = (cy - h / 2.f - pad_y_) / scale_;
        float bw  = w / scale_;
        float bh  = h / scale_;

        boxes.emplace_back(x1, y1, bw, bh);
        scores.push_back(score);
    }

    // NMS via OpenCV
    std::vector<int> indices;
    if (!boxes.empty()) {
        // cv::dnn::NMSBoxes wants cv::Rect2d or Rect, convert
        std::vector<cv::Rect> int_boxes;
        int_boxes.reserve(boxes.size());
        for (auto& b : boxes)
            int_boxes.emplace_back(static_cast<int>(b.x), static_cast<int>(b.y),
                                   static_cast<int>(b.width), static_cast<int>(b.height));
        cv::dnn::NMSBoxes(int_boxes, scores, score_thresh_, nms_thresh_, indices);
    }

    std::vector<PalmDetection> results;
    for (int idx : indices) {
        if (static_cast<int>(results.size()) >= 2) break;  // at most 2

        auto& b = boxes[idx];
        float nx = b.x / frame_w;
        float ny = b.y / frame_h;
        float nw = b.width  / frame_w;
        float nh = b.height / frame_h;

        PalmDetection pd;
        pd.score = scores[idx];
        pd.box   = {nx, ny, nw, nh};

        // No keypoints from YOLO, so set rotation = 0 and compute ROI
        // from the bounding box centre. The landmark model tolerates
        // zero rotation — it's just a tighter initial crop.
        pd.rotation = 0.f;
        float roi_cx = b.x + b.width  / 2.f;
        float roi_cy = b.y + b.height / 2.f;
        float long_side = std::max(b.width, b.height) * 2.0f;

        pd.roi_cx   = roi_cx;
        pd.roi_cy   = roi_cy;
        pd.roi_size = long_side;

        // Zero-fill keypoints (unused downstream when rotation = 0)
        for (auto& kp : pd.keypoints) kp = {0, 0};

        results.push_back(pd);
    }
    return results;
}

// ── Public API ──────────────────────────────────────────────────────────────
std::vector<PalmDetection> HandDetector::detect(const cv::Mat& bgr_frame,
                                                 int max_hands)
{
    preprocess(bgr_frame);

    std::array<int64_t, 4> shape{1, 3, input_h_, input_w_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_, input_buf_.data(), input_buf_.size(),
        shape.data(), shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name  = session_.GetInputNameAllocated(0, alloc);
    auto out_name = session_.GetOutputNameAllocated(0, alloc);
    const char* in_names[]  = {in_name.get()};
    const char* out_names[] = {out_name.get()};

    auto outs = session_.Run(Ort::RunOptions{nullptr},
        in_names, &input_tensor, 1,
        out_names, 1);

    // Output shape: [1, 4+nc, N]
    auto out_shape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    int n_det = static_cast<int>(out_shape[2]);
    const float* data = outs[0].GetTensorData<float>();

    auto results = postprocess(data, n_det, bgr_frame.cols, bgr_frame.rows);

    // Respect max_hands
    if (static_cast<int>(results.size()) > max_hands)
        results.resize(max_hands);

    return results;
}
