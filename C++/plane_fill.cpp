#include "plane_fill.h"

cv::Mat PlaneFill::calc_Cross(const cv::Mat& pcd)
{
    clock_t start, finish;
    double duration;
    start = clock();

    int x = pcd.cols;
    int y = pcd.rows;

    int x_size = x-grad_size;
    int y_size = y-grad_size;
    PlaneFill::devide_region(pcd);

    cv::Mat dx = (sub_regions[1]-sub_regions[0])/(grad_size);
    cv::Mat dy = (sub_regions[3]-sub_regions[2])/(grad_size);

    cv::Mat result(dx.size(), dx.type());

    for (int y = 0; y < dx.rows; y++)
    {
        for (int x = 0; x < dx.cols; x++)
        {
            cv::Vec3f v1 = dx.at<cv::Vec3f>(y, x);
            cv::Vec3f v2 = dy.at<cv::Vec3f>(y, x);
            cv::Vec3f cross = v1.cross(v2);
            float norm = cv::norm(cross);
            result.at<cv::Vec3f>(y, x) = (cross / norm + cv::Vec3f(1, 1, 1)) / 2;
        }
    }

    finish = clock();
    duration = (double)(finish-start)/CLOCKS_PER_SEC;
    std::cout << "Compute time[Cross] : " << duration << std::endl;
    return result;
}

cv::Mat PlaneFill::calc_Fast_fill(const cv::Mat& depth)
{
    cv::Mat amplified_depth = depth*1000;
    clock_t start, finish;
    double duration;

    PlaneFill::devide_region(amplified_depth);

    cv::Mat dx = (sub_regions[1]-sub_regions[0])/(grad_size);
    cv::Mat dy = (sub_regions[3]-sub_regions[2])/(grad_size);
    cv::Point minLoc, minLoc2;
    cv::Point maxLoc, maxLoc2;
    double minval, maxval, minval2, maxval2;
    cv::minMaxLoc(amplified_depth, &minval, &maxval, &minLoc, &maxLoc);
    std::vector<cv::Mat> channels(3);
    cv::Mat zz = sub_regions[4];
    cv::Mat size = dx.mul(dx)+dy.mul(dy)+zz.mul(zz);
    cv::sqrt(size, size);
    cv::Mat normal;
    
    channels[0] = (-dx/size+1)/2;
    channels[1] = (-dy/size+1)/2;
    channels[2] = (zz/size+1)/2;
    cv::merge(channels, normal);

    finish = clock();
    duration = (double)(finish-start)/CLOCKS_PER_SEC;
    std::cout << "Compute time[F_Fill] : " << duration << std::endl;

    return normal;
}

cv::Mat PlaneFill::calc_Depth_fill(const cv::Mat& depth)
{
    clock_t start, finish;
    double duration;
    start = clock();

    int x = depth.cols;
    int y = depth.rows;

    int x_size = x-grad_size;
    int y_size = y-grad_size;
    PlaneFill::devide_region(depth);

    cv::Mat dx = (sub_regions[1]-sub_regions[0])/(grad_size);
    cv::Mat dy = (sub_regions[3]-sub_regions[2])/(grad_size);
    cv::Mat X, Y;
    meshgrid(x_size, y_size, grad_size, X, Y, cx, cy);
    cv::Mat d = sub_regions[4];
    cv::Mat zz = (d + X.mul(dx) + Y.mul(dy)) / fx;
    cv::Mat size = dx.mul(dx)+dy.mul(dy)+zz.mul(zz);;

    cv::sqrt(size, size);
    cv::Mat normal;
    std::vector<cv::Mat> channels(3);
    channels[0] = (-dx / size + 1) / 2;
    channels[1] = (-dy / size + 1) / 2;
    channels[2] = (zz / size + 1) / 2;
    cv::merge(channels, normal);

    finish = clock();
    duration = (double)(finish-start)/CLOCKS_PER_SEC;
    std::cout << "Compute time[D_Fill] : " << duration << std::endl;

    return normal;
}

cv::Mat PlaneFill::Labelling(const cv::Mat& normal) {
    clock_t start, finish;
    double duration;
    start = clock();

    int area1, area2;
    int it = 1;

    int x_size = normal.cols;
    int y_size = normal.rows;
    cv::Mat label = cv::Mat::zeros(y_size, x_size, CV_8U);

    for(int i=int(x_size/(x_seeds+1)); i<x_size-1; i += int(x_size/(x_seeds+1)))
            {
                for (int j=int(y_size/(y_seeds+1)); j<y_size-1; j += int(y_size/(y_seeds+1)))
                {
                    if (label.at<u_char>(j, i) == 0)
                    {
                        cv::Mat mask1 = cv::Mat::zeros(normal.rows + 2, normal.cols + 2, CV_8UC1);
                        cv::Mat mask2 = cv::Mat::zeros(normal.rows + 2, normal.cols + 2, CV_8UC1);

                        area1 = cv::floodFill(normal.clone(), mask1, cv::Point(i, j), cv::Scalar(1, 1, 1), &ccomp, cv::Scalar(0.01, 0.01, 0.01), cv::Scalar(0.01, 0.01, 0.01), 8);
                        area2 = cv::floodFill(normal.clone(), mask2, cv::Point(i, j), cv::Scalar(1, 1, 1), &ccomp, cv::Scalar(0.2, 0.2, 0.2), cv::Scalar(0.2, 0.2, 0.2), cv::FLOODFILL_FIXED_RANGE);
                        
                        if (area1 > 3000 && area2 > 3000)
                        {
                            label += ((mask1.mul(mask2))(cv::Rect(1, 1, x_size, y_size)))*it;
                            mask1.release();
                            mask2.release();
                            it += 1;
                        }
                    }
                }
            }
    label.convertTo(label, CV_32F);
    label/=it;

    finish = clock();
    duration = (double)(finish-start)/CLOCKS_PER_SEC;
    std::cout << "Compute time[Labeling] : " << duration << std::endl;

    return label;
}

void PlaneFill::devide_region(const cv::Mat& region)
{
    sub_regions.clear();
    int x = region.cols;
    int y = region.rows;

    int x_size = x-grad_size;
    int y_size = y-grad_size;

    sub_regions.push_back(region.clone()(cv::Rect(0, grad_size/2, x_size, y_size)));
    sub_regions.push_back(region.clone()(cv::Rect(grad_size, grad_size/2, x_size, y_size)));
    sub_regions.push_back(region.clone()(cv::Rect(grad_size/2, 0, x_size, y_size)));
    sub_regions.push_back(region.clone()(cv::Rect(grad_size/2, grad_size, x_size, y_size)));
    sub_regions.push_back(region.clone()(cv::Rect(grad_size/2, grad_size/2, x_size, y_size)));
}

void meshgrid(int x_size, int y_size, int grad_size, cv::Mat &X, cv::Mat &Y, float cx, float cy) {
    
    X = cv::Mat(y_size, x_size, CV_32FC1);
    Y = cv::Mat(y_size, x_size, CV_32FC1);
    
    for (int i = 0; i < y_size; ++i) {
        for (int j = 0; j < x_size; ++j) {
            X.at<float>(i, j) = grad_size/2+j-cx+1;
            Y.at<float>(i, j) = grad_size/2+i-cy+1;
        }
    }
}

std::vector<fs::path> Load_datas(const std::string& path, const std::string& keyword) {
    std::vector<fs::path> files;

    for(const auto& entry : fs::directory_iterator(path)) {
        if(fs::is_regular_file(entry)){
            std::string filename = entry.path().filename().string();
            if(filename.find(keyword) != std::string::npos){
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
        std::cerr << "Unsupported array shape. Only 2D and 3D arrays are supported." << std::endl;
        return cv::Mat();
    }

    return mat.clone();
}