#include <iostream>

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace sl;

int main(int, char**) {
    Camera zed;
    
    sl::InitParameters init_parameters;
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL;
    init_parameters.depth_minimum_distance = 0.2;
    init_parameters.depth_maximum_distance = 20;
    // init_parameters.input.setFromSVOFile("../data/220902_Gym/short.svo");
    init_parameters.input.setFromSVOFile("../data/230116_M327/auto_v.svo");
    init_parameters.svo_real_time_mode = true;
    init_parameters.coordinate_units = UNIT::METER;

    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        return EXIT_FAILURE;
    }

    Mat zed_image;
    Mat zed_depth;

    int grad_size = 20;
    int x_seeds = 8;
    int y_seeds = 4;
    int x_size = 1280-grad_size;
    int y_size = 720-grad_size;
    cv::Rect R1(0, grad_size/2, x_size, y_size);
    cv::Rect R2(grad_size, grad_size/2, x_size, y_size);
    cv::Rect R3(grad_size/2, 0, x_size, y_size);
    cv::Rect R4(grad_size/2, grad_size, x_size, y_size);
    cv::Rect R5(grad_size/2, grad_size/2, x_size, y_size);

    int iter = 0;

    while (zed.isOpened()) {
        returned_state = zed.grab();
        if (returned_state == ERROR_CODE::SUCCESS) {
            clock_t start, finish; 
            double duration;
            
            zed.retrieveImage(zed_image, VIEW::LEFT);
            zed.retrieveMeasure(zed_depth, MEASURE::DEPTH);
            cv::Mat cvImage = cv::Mat((int) zed_image.getHeight(), (int) zed_image.getWidth(), CV_8UC4, zed_image.getPtr<sl::uchar1>(sl::MEM::CPU));
            cv::Mat cvDepth = cv::Mat((int) zed_depth.getHeight(), (int) zed_depth.getWidth(), CV_32FC1, zed_depth.getPtr<sl::uchar1>(sl::MEM::CPU));
            cvDepth *= 1000;
            start = clock();
            cv::Mat img1 = cvDepth.clone()(R1);
            cv::Mat img2 = cvDepth.clone()(R2);
            cv::Mat img3 = cvDepth.clone()(R3);
            cv::Mat img4 = cvDepth.clone()(R4);
            cv::Mat dx = (img2-img1)/(2*grad_size);
            cv::Point minLoc, minLoc2;
            cv::Point maxLoc, maxLoc2;
            double minval, maxval, minval2, maxval2;
            cv::minMaxLoc(cvDepth, &minval, &maxval, &minLoc, &maxLoc);
            cv::Mat dy = (img4-img3)/(2*grad_size);
            vector<cv::Mat> channels(3);
            cv::Mat zz = ((cvDepth.clone())/maxval)(R5);
            cv::Mat size = dx.mul(dx)+dy.mul(dy)+zz.mul(zz);
            cv::sqrt(size, size);
            cv::Mat normal;
            cv::Rect ccomp;
            channels[0] = (-dx/size+1)/2;
            channels[1] = (-dy/size+1)/2;
            channels[2] = (zz/size+1)/2;
            cv::merge(channels, normal);
            cv::minMaxLoc(channels[1], &minval2, &maxval2, &minLoc2, &maxLoc2);
            int area1, area2;
            int it = 1;
            cv::Mat label = cv::Mat::zeros(y_size, x_size, CV_32F);
            cout << "Compute Time : " << (double)(clock()-start) / CLOCKS_PER_SEC << " ";

            for(int i=int(x_size/(x_seeds+1)); i<x_size-1; i += int(x_size/(x_seeds+1)))
            {
                for (int j=int(y_size/(y_seeds+1)); j<y_size-1; j += int(y_size/(y_seeds+1)))
                {
                    if (label.at<float>(j, i) == 0)
                    {
                        cv::Mat mask1, mask2;
                        area1 = cv::floodFill(normal.clone(), mask1, cv::Point(i, j), cv::Scalar(1, 1, 1), &ccomp, cv::Scalar(0.01, 0.01, 0.01), cv::Scalar(0.01, 0.01, 0.01), 8);
                        area2 = cv::floodFill(normal.clone(), mask2, cv::Point(i, j), cv::Scalar(1, 1, 1), &ccomp, cv::Scalar(0.2, 0.2, 0.2), cv::Scalar(0.2, 0.2, 0.2), cv::FLOODFILL_FIXED_RANGE);
                        if (area1 > 3000 && area2 > 3000)
                        {
                            label += (mask1.mul(mask2))(cv::Rect(1, 1, x_size, y_size))*it;
                            mask1.release();
                            mask2.release();
                            it++;
                        }
                    }
                }
            }
            
            label /= it;
            finish = clock();
            duration = (double)(finish-start) / CLOCKS_PER_SEC;
            cv::imshow("Color", cvImage);
            cv::imshow("normal", normal);
            cv::imshow("label", label);
            cout << "Total Time : " << duration << endl;
        }
        char key = cv::waitKey(1);
        if (key == ' ')
            key = cv::waitKey(0);
        if (key == 'q')
            break;
    }

    zed.close();
    return EXIT_SUCCESS;
}