//
// Created by jalen on 23-1-4.
//
#include "task.h"

void task14_1() {
    int cnt, g_d = 60;
    Mat src, srcMat, dstMat, maskMat, faceMat, filterMat, labelsMat, stats, centroids;
    src = imread("../res/14-1.jpg");
    srcMat = maskMat = src.clone();
    dstMat = src.clone();

    // 高斯滤波之后进行连通块检测
    GaussianBlur(srcMat, srcMat, Size(19, 19), 10, 10);
    cvtColor(srcMat, srcMat, COLOR_BGR2GRAY);
    threshold(srcMat, srcMat, 70, 255, THRESH_BINARY);
    cnt = connectedComponentsWithStats(srcMat, labelsMat, stats, centroids);

    // 制作脸部区域的 mask
    for (int i = 0; i < srcMat.cols; ++i) {
        for (int j = 0; j < srcMat.rows; ++j) {
            int t = (labelsMat.at<int>(j, i) == 6) ? 255 : 0;
            maskMat.at<Vec3b>(j, i) = Vec3b(t, t, t);
        }
    }

    // mask 和图像叠加获取脸部图像 进行双边滤波
    bitwise_and(src, maskMat, faceMat);
    bilateralFilter(faceMat, filterMat, g_d, g_d * 2, g_d / 2.0);
    // 用 mask 仅叠加脸部
    filterMat.copyTo(dstMat, maskMat);

    imshow("src", src);
    imshow("dst", dstMat);

    waitKey(0);
}

void task14_2() {
    int  cnt, g_d = 10;
    Mat src, srcMat, dstMat, maskMat, faceMat, kernel, filterMat, labelsMat, stats, centroids;
    src = imread("../res/14-2.jpg");
    srcMat = maskMat = src.clone();
    dstMat = src.clone();

    // 高斯滤波之后进行连通块检测
    GaussianBlur(srcMat, srcMat, Size(19, 19), 10, 10);
    cvtColor(srcMat, srcMat, COLOR_BGR2GRAY);
    threshold(srcMat, srcMat, 70, 255, THRESH_BINARY);
    kernel = getStructuringElement(MORPH_RECT, Size(15, 15));
    morphologyEx(srcMat, srcMat, MORPH_OPEN, kernel);
    cnt = connectedComponentsWithStats(srcMat, labelsMat, stats, centroids);

    // 制作脸部区域的 mask
    for (int i = 0; i < srcMat.cols; ++i) {
        for (int j = 0; j < srcMat.rows; ++j) {
            int t = (labelsMat.at<int>(j, i) == 2) ? 255 : 0;
            maskMat.at<Vec3b>(j, i) = Vec3b(t, t, t);
        }
    }

    // mask 和图像叠加获取脸部图像 进行双边滤波
    bitwise_and(src, maskMat, faceMat);
    kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(faceMat, faceMat, MORPH_CLOSE, kernel);
    bilateralFilter(faceMat, filterMat, g_d, g_d * 2, g_d * 2);
    // 用 mask 仅叠加脸部
    filterMat.copyTo(dstMat, maskMat);

    imshow("src", src);
    imshow("dst", dstMat);

    waitKey(0);
}
