#ifndef PLANE_FILL_H
#define PLANE_FILL_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cnpy.h>

namespace fs = std::filesystem;

class PlaneFill{
public:
    cv::Mat calc_Cross(const cv::Mat& pcd);
    cv::Mat calc_Fast_fill(const cv::Mat& depth);
    cv::Mat calc_Depth_fill(const cv::Mat& depth);
    cv::Mat Labelling(const cv::Mat& normal);
    void devide_region(const cv::Mat& region);

    float fx;
    float cx;
    float cy;
    int grad_size;
    int x_seeds;
    int y_seeds;
    cv::Rect ccomp;

private:
    std::vector<cv::Mat> sub_regions;
};

void meshgrid(int x_size, int y_size, int grad_size, cv::Mat &X, cv::Mat &Y, float cx, float cy);
std::vector<fs::path> Load_datas(const std::string& path, const std::string& keyword);
cv::Mat NpytoMat(const std::string& filePath);

#endif