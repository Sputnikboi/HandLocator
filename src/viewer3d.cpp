#include "viewer3d.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdio>

// ── Impl ────────────────────────────────────────────────────────────
struct Viewer3D::Impl {
    GLFWwindow* window = nullptr;
    int width, height;

    // Camera orbit
    float rotX   = -20.f;
    float rotY   =  30.f;
    float zoom   =  1.0f;

    // Mouse state
    bool  dragging = false;
    double lastMx = 0, lastMy = 0;
};

// ── GLFW callbacks ──────────────────────────────────────────────────
static void scroll_cb(GLFWwindow* w, double /*dx*/, double dy) {
    auto* impl = static_cast<Viewer3D::Impl*>(glfwGetWindowUserPointer(w));
    impl->zoom *= (dy > 0) ? 1.1f : 0.9f;
    impl->zoom = std::clamp(impl->zoom, 0.2f, 5.0f);
}

static void mouse_button_cb(GLFWwindow* w, int button, int action, int /*mods*/) {
    auto* impl = static_cast<Viewer3D::Impl*>(glfwGetWindowUserPointer(w));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        impl->dragging = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &impl->lastMx, &impl->lastMy);
    }
}

static void cursor_cb(GLFWwindow* w, double x, double y) {
    auto* impl = static_cast<Viewer3D::Impl*>(glfwGetWindowUserPointer(w));
    if (impl->dragging) {
        impl->rotY += static_cast<float>(x - impl->lastMx) * 0.4f;
        impl->rotX += static_cast<float>(y - impl->lastMy) * 0.4f;
        impl->rotX = std::clamp(impl->rotX, -89.f, 89.f);
        impl->lastMx = x;
        impl->lastMy = y;
    }
}

// ── Constructor / Destructor ────────────────────────────────────────
Viewer3D::Viewer3D(int width, int height) : impl_(new Impl) {
    impl_->width  = width;
    impl_->height = height;

    if (!glfwInit()) {
        std::fprintf(stderr, "GLFW init failed\n");
        return;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    impl_->window = glfwCreateWindow(width, height, "3D Hand Viewer", nullptr, nullptr);
    if (!impl_->window) {
        std::fprintf(stderr, "GLFW window creation failed\n");
        return;
    }

    glfwMakeContextCurrent(impl_->window);
    glfwSwapInterval(0);  // no vsync — main loop is camera-rate

    glfwSetWindowUserPointer(impl_->window, impl_);
    glfwSetScrollCallback(impl_->window, scroll_cb);
    glfwSetMouseButtonCallback(impl_->window, mouse_button_cb);
    glfwSetCursorPosCallback(impl_->window, cursor_cb);

    glewExperimental = GL_TRUE;
    glewInit();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
}

Viewer3D::~Viewer3D() {
    if (impl_->window) glfwDestroyWindow(impl_->window);
    glfwTerminate();
    delete impl_;
}

bool Viewer3D::should_close() const {
    return !impl_->window || glfwWindowShouldClose(impl_->window);
}

// ── Drawing helpers ─────────────────────────────────────────────────
static void draw_grid() {
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    float extent = 0.15f;
    int divisions = 10;
    float step = extent * 2.f / divisions;
    for (int i = 0; i <= divisions; ++i) {
        float t = -extent + i * step;
        float bright = (i == divisions / 2) ? 0.35f : 0.18f;
        glColor4f(bright, bright, bright, 0.6f);
        glVertex3f(t, 0, -extent);
        glVertex3f(t, 0,  extent);
        glVertex3f(-extent, 0, t);
        glVertex3f( extent, 0, t);
    }
    glEnd();
}

static void draw_axes() {
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // X = red
    glColor3f(0.9f, 0.2f, 0.2f); glVertex3f(0,0,0); glVertex3f(0.06f, 0, 0);
    // Y = green
    glColor3f(0.2f, 0.9f, 0.2f); glVertex3f(0,0,0); glVertex3f(0, 0.06f, 0);
    // Z = blue
    glColor3f(0.3f, 0.3f, 0.9f); glVertex3f(0,0,0); glVertex3f(0, 0, 0.06f);
    glEnd();
}

static void draw_sphere(float x, float y, float z, float r, int segs = 8) {
    // Simple circle-based sphere approximation
    for (int i = 0; i < segs; ++i) {
        float lat0 = M_PI * (-0.5f + (float)i / segs);
        float lat1 = M_PI * (-0.5f + (float)(i+1) / segs);
        float cy0 = std::cos(lat0), sy0 = std::sin(lat0);
        float cy1 = std::cos(lat1), sy1 = std::sin(lat1);
        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= segs; ++j) {
            float lng = 2.f * M_PI * (float)j / segs;
            float cx = std::cos(lng), sz = std::sin(lng);
            glVertex3f(x + r*cx*cy0, y + r*sy0, z + r*sz*cy0);
            glVertex3f(x + r*cx*cy1, y + r*sy1, z + r*sz*cy1);
        }
        glEnd();
    }
}

static void draw_hand(const HandLandmarks& hand, float cr, float cg, float cb) {
    auto& wl = hand.world_landmarks;

    // Center on centroid
    float cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < 21; ++i) {
        cx += wl[i].x; cy += wl[i].y; cz += wl[i].z;
    }
    cx /= 21.f; cy /= 21.f; cz /= 21.f;

    float pts[21][3];
    for (int i = 0; i < 21; ++i) {
        pts[i][0] = wl[i].x - cx;
        pts[i][1] = -(wl[i].y - cy);  // flip Y so fingers point up
        pts[i][2] = wl[i].z - cz;
    }

    // Bones
    glLineWidth(3.0f);
    glColor3f(cr, cg, cb);
    glBegin(GL_LINES);
    for (auto [a, b] : HAND_CONNECTIONS) {
        glVertex3f(pts[a][0], pts[a][1], pts[a][2]);
        glVertex3f(pts[b][0], pts[b][1], pts[b][2]);
    }
    glEnd();

    // Joints — fingertips brighter
    for (int i = 0; i < 21; ++i) {
        bool tip = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20);
        if (tip)
            glColor3f(std::min(1.f, cr+0.4f), std::min(1.f, cg+0.4f), std::min(1.f, cb+0.4f));
        else
            glColor3f(cr * 0.8f, cg * 0.8f, cb * 0.8f);
        float radius = tip ? 0.005f : 0.004f;
        draw_sphere(pts[i][0], pts[i][1], pts[i][2], radius);
    }
}

// ── Main render call ────────────────────────────────────────────────
bool Viewer3D::render(const HandLandmarks* hands, int n_hands,
                       const cv::Scalar* colors)
{
    if (!impl_->window) return false;
    glfwPollEvents();
    if (glfwWindowShouldClose(impl_->window)) return false;

    glfwGetFramebufferSize(impl_->window, &impl_->width, &impl_->height);
    glViewport(0, 0, impl_->width, impl_->height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Projection
    float aspect = static_cast<float>(impl_->width) / std::max(1, impl_->height);
    float fov = 45.f;
    float near = 0.001f, far = 10.f;
    float top = near * std::tan(fov * 0.5f * M_PI / 180.f);
    float right = top * aspect;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-right, right, -top, top, near, far);

    // Camera: orbit around origin
    float dist = 0.35f / impl_->zoom;
    float radX = impl_->rotX * M_PI / 180.f;
    float radY = impl_->rotY * M_PI / 180.f;
    float eyeX = dist * std::cos(radX) * std::sin(radY);
    float eyeY = dist * std::sin(radX);
    float eyeZ = dist * std::cos(radX) * std::cos(radY);

    // lookAt: eye → origin, up = (0,1,0)
    // forward = normalize(target - eye) = normalize(-eye)
    float fx = -eyeX, fy = -eyeY, fz = -eyeZ;
    float flen = std::sqrt(fx*fx + fy*fy + fz*fz);
    fx /= flen; fy /= flen; fz /= flen;

    // side = normalize(forward × up)
    // cross(f, (0,1,0)) = (f.y*0 - f.z*1, f.z*0 - f.x*0, f.x*1 - f.y*0)
    float sx = -fz, sy = 0.f, sz = fx;
    float slen = std::sqrt(sx*sx + sz*sz);
    if (slen > 1e-6f) { sx /= slen; sz /= slen; }

    // up = side × forward
    float ux = sy*fz - sz*fy;
    float uy = sz*fx - sx*fz;
    float uz = sx*fy - sy*fx;

    // OpenGL column-major view matrix
    float view[16] = {
         sx,  ux, -fx, 0,
         sy,  uy, -fy, 0,
         sz,  uz, -fz, 0,
        -(sx*eyeX + sy*eyeY + sz*eyeZ),
        -(ux*eyeX + uy*eyeY + uz*eyeZ),
         (fx*eyeX + fy*eyeY + fz*eyeZ),
         1
    };

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(view);

    draw_grid();
    draw_axes();

    for (int i = 0; i < n_hands; ++i) {
        if (!hands[i].valid) continue;
        float r = static_cast<float>(colors[i][2]) / 255.f;  // cv::Scalar is BGR
        float g = static_cast<float>(colors[i][1]) / 255.f;
        float b = static_cast<float>(colors[i][0]) / 255.f;
        draw_hand(hands[i], r, g, b);
    }

    glfwSwapBuffers(impl_->window);
    return true;
}
