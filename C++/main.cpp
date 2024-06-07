#include "plane_fill.h"

int main(int, char**) {
    PlaneFill fills;

    fills.cx = 633.66333;
    fills.cy = 365.10729;
    fills.fx = 540.58892;
    fills.grad_size = 10;
    fills.x_seeds = 8;
    fills.y_seeds = 4;

    std::string path = "../data_files";

    std::vector<fs::path> color_files = Load_datas(path, "color");
    std::vector<fs::path> xyz_files = Load_datas(path, "xyz");
    std::vector<fs::path> depth_files = Load_datas(path, "depth");

    for(size_t i=0; i<xyz_files.size(); i++)
    {
        cv::Mat depth = NpytoMat(depth_files[i].string());
        cv::Mat color = NpytoMat(color_files[i].string());
        cv::Mat xyz = NpytoMat(xyz_files[i].string());
        cv::Mat normal = fills.calc_Depth_fill(depth);
        cv::Mat cross = fills.calc_Cross(xyz);
        cv::Mat label = fills.Labelling(normal);
        cv::Mat c_label = fills.Labelling(cross);
        std::cout << std::endl;

        cv::imshow("color", color);
        cv::imshow("normal", normal);
        cv::imshow("cross", cross);
        cv::imshow("c_label", c_label);
        cv::imshow("label", label);
        cv::waitKey(0);
    }
    return 0;
}