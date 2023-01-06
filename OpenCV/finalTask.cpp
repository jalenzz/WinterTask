//
// Created by jalen on 23-1-6.
//
#include "task.h"

void f() {
    imshow("f1", f1());
    imshow("f2", f2());
    imshow("f3", f3());
    waitKey();
}

Mat f1() {
    Mat src, srcMat, labels, stats, centroids;;
    src = imread("../res/f1.jpg");

    blur(src, srcMat, Size(15, 15));
    cvtColor(srcMat, srcMat, COLOR_BGR2GRAY);
    threshold(srcMat, srcMat, 100, 255, THRESH_BINARY_INV);
    morphologyEx(srcMat, srcMat, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5, 5)));
    int cnt = connectedComponentsWithStats(srcMat, labels, stats, centroids);

    for (int i = 0; i < cnt; ++i) {
        int width = stats.at<int>(i, 2), height = stats.at<int>(i, 3);
        if (abs(width - height) < 10 && width > 10) {
            int x = (int)centroids.at<double>(i, 0), y = (int)centroids.at<double>(i, 1);
            circle(src, Point(x, y), (width / 2) + 5, Scalar(0, 255, 255), -1);
        }
    }

    return src;
}

Mat f2() {
    Mat src, srcMat, labels, stats, centroids;;
    src = imread("../res/f2.jpg");

//    blur(src, srcMat, Size(9, 9));
    cvtColor(src, srcMat, COLOR_BGR2GRAY);
    threshold(srcMat, srcMat, 120, 255, THRESH_BINARY);
    morphologyEx(srcMat, srcMat, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(20, 20)));
    connectedComponentsWithStats(srcMat, labels, stats, centroids);
    int *t = stats.ptr<int>(1);
    rectangle(src, Rect(*t, *(t+1), *(t+2), *(t+3)),
              Scalar(0, 0, 255), 2);

    return src;
}


Mat f3() {
    Mat src, hsvMat, Red1, Red2, redMat, roiMat, labels, stats, centroids;
    src = imread("../res/f3.jpg");

    cvtColor(src, hsvMat, COLOR_BGR2HSV);
    // 提取红色区域
    inRange(hsvMat, Scalar(0, 43, 46), Scalar(10, 255, 255), Red1);
    inRange(hsvMat, Scalar(170, 43, 46), Scalar(180, 255, 255), Red2);
    // 去除小块的红色区域
    morphologyEx(Red1+Red2, redMat, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(15, 15)));
    connectedComponentsWithStats(redMat, labels, stats, centroids);
    int *p = stats.ptr<int>(1);
    Rect ROI = Rect(*(p), *(p+1), *(p+2), *(p+3));
    rectangle(src, ROI, Scalar(255, 0, 0), 3);

    return src;
}
