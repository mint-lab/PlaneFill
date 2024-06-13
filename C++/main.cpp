#include "plane_fill.h"

int main(int, char**) {
  PlaneFill fills;

  fills.cx_ = 633.66333;
  fills.cy_ = 365.10729;
  fills.fx_ = 540.58892;
  fills.grad_size_ = 10;
  fills.num_x_seeds_ = 8;
  fills.num_y_seeds_ = 4;

  std::string path = "../data_files";

  std::vector<fs::path> color_files = LoadDatas(path, "color");
  std::vector<fs::path> xyz_files = LoadDatas(path, "xyz");
  std::vector<fs::path> depth_files = LoadDatas(path, "depth");

  for (size_t i = 0; i < xyz_files.size(); i++) {
    cv::Mat depth = NpytoMat(depth_files[i].string());
    cv::Mat color = NpytoMat(color_files[i].string());
    cv::Mat xyz = NpytoMat(xyz_files[i].string());
    cv::Mat normal = fills.GetNoramlFromMINT24(depth);
    cv::Mat cross = fills.GetNormalFromCrossProduct(xyz);
    cv::Mat label = fills.Labelling(normal, 100, cv::Scalar(0.02, 0.02, 0.02));
    cv::Mat c_label = fills.Labelling(cross, 100, cv::Scalar(0.02, 0.02, 0.02));
    std::cout << std::endl;

    cv::imshow("color", color);
    cv::imshow("normal", normal);
    cv::imshow("cross", cross);
    cv::imshow("c_label", c_label);
    cv::imshow("label", label);
    int key = cv::waitKey(0);

    if (key == 'q') {
      break;
    }
  }
  return 0;
}