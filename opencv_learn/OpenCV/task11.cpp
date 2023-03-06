//
// Created by jalen on 23-1-4.
//
#include "task.h"
#include "iostream"

void task11() {
    Mat src = imread("../res/11.png");
    Mat srcGray, srcThreshold, labelsMat, stats, centroids;

    cvtColor( src, srcGray, COLOR_BGR2GRAY );
    threshold(srcGray, srcThreshold, 85, 255, THRESH_BINARY_INV);

    int cnt = connectedComponentsWithStats(srcThreshold, labelsMat, stats, centroids);

    for (int i = 1; i < cnt; ++i) {
        int *t = stats.ptr<int>(i);
        rectangle(src, Rect(*t, *(t+1), *(t+2), *(t+3)),
                  Scalar(0, 0, 0), 2);
    }

    std::cout << cnt;
    imshow("task11", src);
    waitKey(0);
}