//
// Created by jalen on 23-1-4.
//
#include "task.h"

void task12() {
    int key;
    Mat frame, blurMat, medianBlurMat, GaussianBlurMat;
    VideoCapture capture(0);

    // 判断是否打开
    if (!capture.isOpened()) {
        printf("Open failed");
        return;
    }

    while (true) {
        if (key == (int)'q') break;
        capture >> frame;

        blur(frame, blurMat, Size(5, 5));
        medianBlur(frame, medianBlurMat, 5);
        GaussianBlur(frame, GaussianBlurMat, Size(5, 5), 5, 5);

        imshow("original", frame);
        imshow("blur", blurMat);
        imshow("mediaBlur", medianBlurMat);
        imshow("GaussianBlur", GaussianBlurMat);

        key = waitKey(30);  // 延时30
    }
}
