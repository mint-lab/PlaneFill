#include "plane_fill.h"

// void processRange(const cv::Mat& dx, const cv::Mat& dy, cv::Mat& result, int
// startRow, int endRow) {
//     for (int y = startRow; y < endRow; y++) {
//         for (int x = 0; x < dx.cols; x++) {
//             cv::Vec3f v1 = dx.at<cv::Vec3f>(y, x);
//             cv::Vec3f v2 = dy.at<cv::Vec3f>(y, x);
//             cv::Vec3f cross = v1.cross(v2);
//             float norm = cv::norm(cross);
//             result.at<cv::Vec3f>(y, x) = (cross / norm + cv::Vec3f(1, 1, 1))
//             / 2;
//         }
//     }
// }

cv::Mat PlaneFill::GetNormalFromCrossProduct(const cv::Mat& pcd) {
  int x = pcd.cols;
  int y = pcd.rows;

  int x_size = x - grad_size_;
  int y_size = y - grad_size_;
  PlaneFill::DivideRegion(pcd);

  cv::Mat dx = (sub_regions_[1] - sub_regions_[0]) / (grad_size_);
  cv::Mat dy = (sub_regions_[3] - sub_regions_[2]) / (grad_size_);

  cv::Mat result(dx.size(), dx.type());

  for (int y = 0; y < dx.rows; y++) {
    for (int x = 0; x < dx.cols; x++) {
      cv::Vec3f v1 = dx.at<cv::Vec3f>(y, x);
      cv::Vec3f v2 = dy.at<cv::Vec3f>(y, x);
      cv::Vec3f cross = v1.cross(v2);
      float norm = cv::norm(cross);
      result.at<cv::Vec3f>(y, x) = (cross / norm + cv::Vec3f(1, 1, 1)) / 2;
    }
  }

  // const int numThreads = std::thread::hardware_concurrency();
  // std::vector<std::thread> threads;
  // int rowsperThread = dx.rows/numThreads;

  // for(int i=0;i<numThreads; i++)
  // {
  //     int startRow = i*rowsperThread;
  //     int endRow = (i==numThreads-1) ? dx.rows:startRow + rowsperThread;
  //     threads.emplace_back(processRange, std::cref(dx), std::cref(dy),
  //     std::ref(result), startRow, endRow);
  // }

  // for (auto& thread : threads) {
  //     thread.join();
  // }
  return result;
}

cv::Mat PlaneFill::GetNormalFromMINT23(const cv::Mat& depth) {
  cv::Mat amplified_depth = depth * 1000;

  PlaneFill::DivideRegion(amplified_depth);

  cv::Mat dx = (sub_regions_[1] - sub_regions_[0]) / (grad_size_);
  cv::Mat dy = (sub_regions_[3] - sub_regions_[2]) / (grad_size_);
  cv::Point minLoc, minLoc2;
  cv::Point maxLoc, maxLoc2;
  double minval, maxval, minval2, maxval2;
  cv::minMaxLoc(amplified_depth, &minval, &maxval, &minLoc, &maxLoc);
  std::vector<cv::Mat> channels(3);
  cv::Mat zz = sub_regions_[4];
  cv::Mat size = dx.mul(dx) + dy.mul(dy) + zz.mul(zz);
  cv::sqrt(size, size);
  cv::Mat normal;

  channels[0] = (-dx / size + 1) / 2;
  channels[1] = (-dy / size + 1) / 2;
  channels[2] = (zz / size + 1) / 2;
  cv::merge(channels, normal);

  return normal;
}

cv::Mat PlaneFill::GetNoramlFromMINT24(const cv::Mat& depth) {
  int x = depth.cols;
  int y = depth.rows;

  int x_size = x - grad_size_;
  int y_size = y - grad_size_;
  PlaneFill::DivideRegion(depth);

  cv::Mat dx = (sub_regions_[1] - sub_regions_[0]) / (grad_size_);
  cv::Mat dy = (sub_regions_[3] - sub_regions_[2]) / (grad_size_);
  cv::Mat X, Y;
  meshgrid(x_size, y_size, grad_size_, X, Y, cx_, cy_);
  cv::Mat d = sub_regions_[4];
  cv::Mat zz = (d + X.mul(dx) + Y.mul(dy)) / fx_;
  cv::Mat size = dx.mul(dx) + dy.mul(dy) + zz.mul(zz);
  ;

  cv::sqrt(size, size);
  cv::Mat normal;
  std::vector<cv::Mat> channels(3);
  channels[0] = (-dx / size + 1) / 2;
  channels[1] = (-dy / size + 1) / 2;
  channels[2] = (zz / size + 1) / 2;
  cv::merge(channels, normal);

  return normal;
}

cv::Mat PlaneFill::Labelling(const cv::Mat& normal, int th_area,
                             cv::Scalar max_diff) {
  int area1, area2;
  int it = 1;

  int x_size = normal.cols;
  int y_size = normal.rows;
  cv::Mat label = cv::Mat::zeros(y_size, x_size, CV_8U);

  cv::Mat mask1 = cv::Mat::zeros(y_size + 2, x_size + 2, CV_8UC1);
  cv::Mat mask2 = cv::Mat::zeros(y_size + 2, x_size + 2, CV_8UC1);

  cv::Mat isChecked = cv::Mat::zeros(y_size, x_size, CV_8UC1);

  for (int i = int(x_size / (num_x_seeds_ + 1)); i < x_size - 1;
       i += int(x_size / (num_x_seeds_ + 1))) {
    for (int j = int(y_size / (num_y_seeds_ + 1)); j < y_size - 1;
         j += int(y_size / (num_y_seeds_ + 1))) {
      if (isChecked.at<u_char>(j, i) == 0) {
        mask1.setTo(0);
        mask2.setTo(0);
        area1 =
            cv::floodFill(normal.clone(), mask1, cv::Point(i, j),
                          cv::Scalar(1, 1, 1), &ccomp_, max_diff, max_diff, 8);
        area2 = cv::floodFill(normal.clone(), mask2, cv::Point(i, j),
                              cv::Scalar(1, 1, 1), &ccomp_, max_diff * 10,
                              max_diff * 10, cv::FLOODFILL_FIXED_RANGE);

        isChecked += ((mask1.mul(mask2))(cv::Rect(1, 1, x_size, y_size)));

        if (area1 > th_area && area2 > th_area) {
          label += ((mask1.mul(mask2))(cv::Rect(1, 1, x_size, y_size))) * it;
          it += 1;
        }
      }
    }
  }
  label.convertTo(label, CV_32F);
  label /= it;

  return label;
}

void PlaneFill::DivideRegion(const cv::Mat& region) {
  sub_regions_.clear();
  int x = region.cols;
  int y = region.rows;

  int x_size = x - grad_size_;
  int y_size = y - grad_size_;

  sub_regions_.push_back(
      region.clone()(cv::Rect(0, grad_size_ / 2, x_size, y_size)));
  sub_regions_.push_back(
      region.clone()(cv::Rect(grad_size_, grad_size_ / 2, x_size, y_size)));
  sub_regions_.push_back(
      region.clone()(cv::Rect(grad_size_ / 2, 0, x_size, y_size)));
  sub_regions_.push_back(
      region.clone()(cv::Rect(grad_size_ / 2, grad_size_, x_size, y_size)));
  sub_regions_.push_back(
      region.clone()(cv::Rect(grad_size_ / 2, grad_size_ / 2, x_size, y_size)));
}