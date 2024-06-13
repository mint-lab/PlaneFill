#include "plane_fill.h"

void meshgrid(int x_size, int y_size, int grad_size, cv::Mat& X, cv::Mat& Y,
              float cx, float cy) {
  X = cv::Mat(y_size, x_size, CV_32FC1);
  Y = cv::Mat(y_size, x_size, CV_32FC1);

  for (int i = 0; i < y_size; ++i) {
    for (int j = 0; j < x_size; ++j) {
      X.at<float>(i, j) = grad_size / 2 + j - cx + 1;
      Y.at<float>(i, j) = grad_size / 2 + i - cy + 1;
    }
  }
}

std::vector<fs::path> LoadDatas(const std::string& path,
                                const std::string& keyword) {
  std::vector<fs::path> files;

  for (const auto& entry : fs::directory_iterator(path)) {
    if (fs::is_regular_file(entry)) {
      std::string filename = entry.path().filename().string();
      if (filename.find(keyword) != std::string::npos) {
        files.push_back(entry.path());
      }
    }
  }

  std::sort(files.begin(), files.end());

  return files;
}

cv::Mat NpytoMat(const std::string& filePath) {
  cnpy::NpyArray npy_data = cnpy::npy_load(filePath);
  float* data_ptr = npy_data.data<float>();
  std::vector<size_t> shape = npy_data.shape;

  int rows = shape[0];
  int cols = shape[1];
  cv::Mat mat;

  if (shape.size() == 2) {
    int rows = shape[0];
    int cols = shape[1];

    if (npy_data.word_size == sizeof(float)) {
      float* data_ptr = npy_data.data<float>();
      mat = cv::Mat(rows, cols, CV_32F, data_ptr);
    } else if (npy_data.word_size == sizeof(double)) {
      double* data_ptr = npy_data.data<double>();
      mat = cv::Mat(rows, cols, CV_64F, data_ptr);
    } else if (npy_data.word_size == sizeof(int)) {
      int* data_ptr = npy_data.data<int>();
      mat = cv::Mat(rows, cols, CV_32S, data_ptr);
    } else if (npy_data.word_size == sizeof(unsigned char)) {
      unsigned char* data_ptr = npy_data.data<unsigned char>();
      mat = cv::Mat(rows, cols, CV_8U, data_ptr);
    } else {
      std::cerr << "Unsupported data type" << std::endl;
      return cv::Mat();
    }
  }

  else if (shape.size() == 3) {
    int rows = shape[0];
    int cols = shape[1];
    int channels = shape[2];

    if (npy_data.word_size == sizeof(float)) {
      float* data_ptr = npy_data.data<float>();
      mat = cv::Mat(rows, cols, CV_32FC(channels), data_ptr);
    } else if (npy_data.word_size == sizeof(double)) {
      double* data_ptr = npy_data.data<double>();
      mat = cv::Mat(rows, cols, CV_64FC(channels), data_ptr);
    } else if (npy_data.word_size == sizeof(int)) {
      int* data_ptr = npy_data.data<int>();
      mat = cv::Mat(rows, cols, CV_32SC(channels), data_ptr);
    } else if (npy_data.word_size == sizeof(unsigned char)) {
      unsigned char* data_ptr = npy_data.data<unsigned char>();
      mat = cv::Mat(rows, cols, CV_8UC(channels), data_ptr);
    } else {
      std::cerr << "Unsupported data type" << std::endl;
      return cv::Mat();
    }
  } else {
    std::cerr << "Unsupported array shape. Only 2D and 3D arrays are supported."
              << std::endl;
    return cv::Mat();
  }

  return mat.clone();
}