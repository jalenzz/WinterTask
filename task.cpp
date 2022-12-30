//
// Created by jalen on 22-12-30.
//

#include "task.h"
#include "opencv2/opencv.hpp"
#include "iostream"

using namespace cv;

void task1() {
    Mat img = imread("/home/jalen/code/Visual_Group/WinterTask/res/1/1-2.jpg");
    int c = img.cols, r = img.rows;


    for(int i=0; i<r; i++) {
        for(int j=0; j<c; j++) {
            int average = 0;
            for(int k=0; k<3; k++) {
                average += img.at<Vec3b>(i, j)[k] / 3;
            }
            for(int k=0; k<3; k++) {
                img.at<Vec3b>(i, j)[k] = average;
            }
        }
    }
    namedWindow("task1", 0);
    resizeWindow("task1", 400, 800);
    imshow("task1", img);
    waitKey(0);
}
