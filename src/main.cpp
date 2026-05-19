#include "hand_detector.hpp"
#include "hand_landmark.hpp"
#include "viewer3d.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

// ── 2D drawing helpers ──────────────────────────────────────────────
static void draw_landmarks(cv::Mat& frame, const HandLandmarks& hand,
                            cv::Scalar color = {0, 200, 0}) {
    for (auto [a, b] : HAND_CONNECTIONS) {
        cv::Point pa(static_cast<int>(hand.landmarks[a].x),
                     static_cast<int>(hand.landmarks[a].y));
        cv::Point pb(static_cast<int>(hand.landmarks[b].x),
                     static_cast<int>(hand.landmarks[b].y));
        cv::line(frame, pa, pb, color, 2, cv::LINE_AA);
    }
    for (int i = 0; i < 21; ++i) {
        cv::Point p(static_cast<int>(hand.landmarks[i].x),
                    static_cast<int>(hand.landmarks[i].y));
        cv::circle(frame, p, 4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
}

static void draw_box(cv::Mat& frame, const PalmDetection& det) {
    int W = frame.cols, H = frame.rows;
    cv::Rect r(
        static_cast<int>(det.box.x      * W),
        static_cast<int>(det.box.y      * H),
        static_cast<int>(det.box.width  * W),
        static_cast<int>(det.box.height * H));
    cv::rectangle(frame, r, cv::Scalar(255, 128, 0), 2);
    cv::putText(frame, cv::format("%.2f", det.score), r.tl() + cv::Point(0, -4),
                cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(255, 128, 0), 1);
}

static PalmDetection palm_from_landmarks(const HandLandmarks& prev,
                                          int frame_w, int frame_h)
{
    float xmin = 1e9f, ymin = 1e9f, xmax = -1e9f, ymax = -1e9f;
    for (int i = 0; i < 21; ++i) {
        float nx = prev.landmarks[i].x / frame_w;
        float ny = prev.landmarks[i].y / frame_h;
        xmin = std::min(xmin, nx);
        ymin = std::min(ymin, ny);
        xmax = std::max(xmax, nx);
        ymax = std::max(ymax, ny);
    }

    float wx = prev.landmarks[WRIST].x;
    float wy = prev.landmarks[WRIST].y;
    float mx = prev.landmarks[MIDDLE_MCP].x;
    float my = prev.landmarks[MIDDLE_MCP].y;
    float dx = mx - wx;
    float dy = my - wy;
    float rotation = static_cast<float>(M_PI_2) - std::atan2(-dy, dx);
    while (rotation >  M_PI) rotation -= 2 * M_PI;
    while (rotation < -M_PI) rotation += 2 * M_PI;

    float cx = (xmin + xmax) / 2.0f;
    float cy = (ymin + ymax) / 2.0f;
    float w  = xmax - xmin;
    float h  = ymax - ymin;
    float long_px = std::max(w * frame_w, h * frame_h) * 1.6f;
    long_px = std::max(long_px, 80.f);

    PalmDetection pd;
    pd.score    = 1.0f;
    pd.box      = {cx - w/2, cy - h/2, w, h};
    pd.rotation = rotation;
    pd.roi_cx   = cx * frame_w;
    pd.roi_cy   = cy * frame_h;
    pd.roi_size = long_px;
    return pd;
}

static bool landmarks_sane(const HandLandmarks& hand, int frame_w, int frame_h) {
    float xmin = 1e9f, ymin = 1e9f, xmax = -1e9f, ymax = -1e9f;
    for (int i = 0; i < 21; ++i) {
        xmin = std::min(xmin, hand.landmarks[i].x);
        ymin = std::min(ymin, hand.landmarks[i].y);
        xmax = std::max(xmax, hand.landmarks[i].x);
        ymax = std::max(ymax, hand.landmarks[i].y);
    }
    float w = xmax - xmin;
    float h = ymax - ymin;
    // Use max dimension — fists/side views can be very narrow in one axis
    float span = std::max(w, h);
    if (span < 5.f) return false;  // truly collapsed
    if (xmax < 0 || ymax < 0 || xmin > frame_w || ymin > frame_h) return false;
    return true;
}

// ── Per-hand tracking state ─────────────────────────────────────────
struct TrackedHand {
    HandLandmarks landmarks{};
    int           lost_count = 0;
    bool          active     = false;
    float         smooth_roi_size = 0.f;  // EMA-smoothed ROI size
};

constexpr int kMaxHands         = 2;
constexpr int kMaxLostFrames    = 5;
constexpr int kRedetectInterval = 15;

int main(int argc, char** argv) {
    std::string det_model = "models/hand_yolov8n.onnx";
    std::string lm_model  = "models/hand_landmark_lite.onnx";
    int camera_id = 0;

    if (argc >= 3) { det_model = argv[1]; lm_model = argv[2]; }
    if (argc >= 4) { camera_id = std::stoi(argv[3]); }

    HandDetector  hand_det(det_model);
    HandLandmarkDetector lm_det(lm_model);

    cv::VideoCapture cap(camera_id, cv::CAP_V4L2);
    if (!cap.isOpened()) cap.open(camera_id ^ 1, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open any camera.\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::namedWindow("HandLocator", cv::WINDOW_GUI_NORMAL);

    Viewer3D viewer(500, 500);

    double fps = 0.0;
    TrackedHand tracked[kMaxHands];
    int frame_count = 0;

    const cv::Scalar hand_colors[2] = {{0, 200, 0}, {200, 200, 0}};

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto t0 = std::chrono::steady_clock::now();

        int n_tracked = 0;

        // Step 1: Track existing hands
        for (int h = 0; h < kMaxHands; ++h) {
            if (!tracked[h].active) continue;
            PalmDetection synthetic = palm_from_landmarks(tracked[h].landmarks,
                                                           frame.cols, frame.rows);
            // Smooth ROI size — can't shrink more than 20% per frame
            if (tracked[h].smooth_roi_size > 0.f) {
                float min_size = tracked[h].smooth_roi_size * 0.8f;
                synthetic.roi_size = std::max(synthetic.roi_size, min_size);
            }
            tracked[h].smooth_roi_size = synthetic.roi_size;
            auto hand = lm_det.detect(frame, synthetic);
            if (hand.valid && landmarks_sane(hand, frame.cols, frame.rows)) {
                draw_landmarks(frame, hand, hand_colors[h]);
                tracked[h].landmarks = hand;
                tracked[h].lost_count = 0;
                ++n_tracked;
            } else {
                ++tracked[h].lost_count;
                if (tracked[h].lost_count >= kMaxLostFrames)
                    tracked[h].active = false;
            }
        }

        // Step 2: Detect new hands periodically or when slots free
        bool need_detect = (n_tracked < kMaxHands) ||
                           (frame_count % kRedetectInterval == 0);
        if (need_detect) {
            auto dets = hand_det.detect(frame, kMaxHands);
            for (auto& det : dets) {
                bool duplicate = false;
                for (int h = 0; h < kMaxHands; ++h) {
                    if (!tracked[h].active) continue;
                    float tcx = 0, tcy = 0;
                    for (int i = 0; i < 21; ++i) {
                        tcx += tracked[h].landmarks.landmarks[i].x;
                        tcy += tracked[h].landmarks.landmarks[i].y;
                    }
                    tcx /= 21.f; tcy /= 21.f;
                    if (std::hypot(det.roi_cx - tcx, det.roi_cy - tcy) < det.roi_size * 0.5f) {
                        duplicate = true; break;
                    }
                }
                if (duplicate) continue;

                int slot = -1;
                for (int h = 0; h < kMaxHands; ++h)
                    if (!tracked[h].active) { slot = h; break; }
                if (slot < 0) break;

                draw_box(frame, det);
                auto hand = lm_det.detect(frame, det);
                if (hand.valid && landmarks_sane(hand, frame.cols, frame.rows)) {
                    draw_landmarks(frame, hand, hand_colors[slot]);
                    tracked[slot].landmarks = hand;
                    tracked[slot].lost_count = 0;
                    tracked[slot].active = true;
                    tracked[slot].smooth_roi_size = det.roi_size;
                }
            }
        }

        n_tracked = 0;
        for (int h = 0; h < kMaxHands; ++h)
            if (tracked[h].active) ++n_tracked;

        // ── 3D viewer ───────────────────────────────────────────────
        HandLandmarks view_hands[kMaxHands];
        for (int h = 0; h < kMaxHands; ++h)
            view_hands[h] = tracked[h].active ? tracked[h].landmarks : HandLandmarks{};
        if (!viewer.render(view_hands, kMaxHands, hand_colors))
            break;  // 3D window closed

        // ── HUD ─────────────────────────────────────────────────────
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fps = 0.9 * fps + 0.1 * (1000.0 / ms);

        cv::putText(frame,
            cv::format("%.1f fps  |  %d hand(s)  |  q=quit", fps, n_tracked),
            {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2, cv::LINE_AA);

        cv::imshow("HandLocator", frame);
        if (cv::waitKey(1) == 'q') break;
        ++frame_count;
    }
    return 0;
}
