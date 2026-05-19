#pragma once
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>

struct PalmDetection {
    cv::Rect2f box;                       // normalized [0,1] in original frame
    float score;
    std::array<cv::Point2f, 7> keypoints; // normalized [0,1]
    float rotation;                       // radians, 0 = hand pointing up
    float roi_cx, roi_cy, roi_size;       // pixels
};

struct HandLandmarks {
    std::array<cv::Point3f, 21> landmarks;       // x,y in original image pixels, z relative depth
    std::array<cv::Point3f, 21> world_landmarks;  // 3D world-space (meters, relative to wrist)
    float presence;
    float handedness; // sigmoid → >0.5 means right hand
    bool valid;
};

// MediaPipe landmark indices
enum Landmark {
    WRIST = 0,
    THUMB_CMC=1, THUMB_MCP=2, THUMB_IP=3, THUMB_TIP=4,
    INDEX_MCP=5, INDEX_PIP=6, INDEX_DIP=7, INDEX_TIP=8,
    MIDDLE_MCP=9, MIDDLE_PIP=10, MIDDLE_DIP=11, MIDDLE_TIP=12,
    RING_MCP=13, RING_PIP=14, RING_DIP=15, RING_TIP=16,
    PINKY_MCP=17, PINKY_PIP=18, PINKY_DIP=19, PINKY_TIP=20
};

static const std::vector<std::pair<int,int>> HAND_CONNECTIONS = {
    {0,1},{1,2},{2,3},{3,4},
    {0,5},{5,6},{6,7},{7,8},
    {0,9},{9,10},{10,11},{11,12},
    {0,13},{13,14},{14,15},{15,16},
    {0,17},{17,18},{18,19},{19,20},
    {5,9},{9,13},{13,17}
};
