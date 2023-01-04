//
// Created by jalen on 23-1-4.
//
#include "task.h"

void task8() {
    Mat HSVMat, Red, Red1, Red2;
    Mat src = imread("../res/8.png");

    cvtColor(src, HSVMat, COLOR_BGR2HSV);
    inRange(HSVMat, Scalar(0, 43, 46), Scalar(10, 255, 255), Red);

    imshow("task8", Red);
    waitKey(0);
}
