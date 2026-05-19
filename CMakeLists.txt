cmake_minimum_required(VERSION 3.16)
project(HandLocator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ── OpenCV (camera capture, image ops, NMS helper) ──────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core dnn highgui imgproc videoio)

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
# Expects the prebuilt ORT package extracted to <project>/onnxruntime/
set(ORT_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")
if(NOT EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime not found at ${ORT_DIR}.\n"
        "Run:  bash setup_onnxruntime.sh")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${ORT_DIR}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include")

# ── GLFW / OpenGL (3D viewer) ────────────────────────────────────────────────
find_package(OpenGL REQUIRED)
find_package(GLEW   REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(hand_locator
    src/main.cpp
    src/hand_detector.cpp
    src/hand_landmark.cpp
    src/viewer3d.cpp
)

target_include_directories(hand_locator PRIVATE
    src
    ${OpenCV_INCLUDE_DIRS}
    ${GLFW3_INCLUDE_DIRS}
)
target_link_libraries(hand_locator PRIVATE
    onnxruntime
    ${OpenCV_LIBS}
    OpenGL::GL
    GLEW::GLEW
    ${GLFW3_LIBRARIES}
)

# Embed the ORT lib path so the binary finds it without LD_LIBRARY_PATH
set_target_properties(hand_locator PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib")

message(STATUS "OpenCV : ${OpenCV_VERSION}")
message(STATUS "ORT dir: ${ORT_DIR}")
message(STATUS "Build  : ${CMAKE_BUILD_TYPE}")
