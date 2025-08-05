#include <open3d/Open3D.h>
#include <iostream>

int main() {
    auto vis = std::make_shared<open3d::visualization::Visualizer>();
    bool success = vis->CreateVisualizerWindow("Test", 640, 480, 0, 0, true);
    std::cout << "Window creation: " << (success ? "SUCCESS" : "FAILED")
              << std::endl;
    return 0;
}