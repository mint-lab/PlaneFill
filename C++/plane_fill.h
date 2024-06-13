#ifndef PLANE_FILL_H
#define PLANE_FILL_H

#include <cnpy.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
// #include <thread>

namespace fs = std::filesystem;

class PlaneFill {
 public:
  cv::Mat GetNormalFromCrossProduct(const cv::Mat& pcd);
  cv::Mat GetNormalFromMINT23(const cv::Mat& depth);
  cv::Mat GetNoramlFromMINT24(const cv::Mat& depth);
  cv::Mat Labelling(const cv::Mat& normal, int th_area = 1000,
                    cv::Scalar max_diff = cv::Scalar(0.01, 0.01, 0.01));
  void DivideRegion(const cv::Mat& region);

  float fx_;
  float cx_;
  float cy_;
  int grad_size_;
  int num_x_seeds_;
  int num_y_seeds_;
  cv::Rect ccomp_;

 private:
  std::vector<cv::Mat> sub_regions_;
};

void meshgrid(int x_size, int y_size, int grad_size, cv::Mat& X, cv::Mat& Y,
              float cx, float cy);
std::vector<fs::path> LoadDatas(const std::string& path,
                                 const std::string& keyword);
cv::Mat NpytoMat(const std::string& filePath);
// void processRange(const cv::Mat& dx, const cv::Mat& dy, cv::Mat& result,
//                   int startRow, int endRow);

#endif