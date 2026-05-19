#pragma once
#include "types.hpp"
#include <array>

class Viewer3D {
public:
    Viewer3D(int width = 500, int height = 500);
    ~Viewer3D();

    // Call each frame with currently tracked hands. Returns false if window closed.
    bool render(const HandLandmarks* hands, int n_hands,
                const cv::Scalar* colors);

    bool should_close() const;

    struct Impl;
    Impl* impl_;
};
