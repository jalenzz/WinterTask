//
// Created by jalen on 23-1-4.
//

#include "task.h"

void task1() {
    Mat img = imread("../res/test/1.jpg");
    int c = img.cols, r = img.rows;


    for(int i=0; i<r; i++) {
        for(int j=0; j<c; j++) {
            int average = 0;
            for(int k=0; k<3; k++) {
                average += img.at<Vec3b>(i, j)[k] / 3;
            }

            // 整体赋值
            img.at<Vec3b>(i, j) = Vec3b (average, average, average);
//            for(int k=0; k<3; k++) {
//                img.at<Vec3b>(i, j)[k] = average;
//            }
        }
    }
    namedWindow("task1", 0);
    resizeWindow("task1", 800, 800);
    imshow("task1", img);
    waitKey(0);
}
