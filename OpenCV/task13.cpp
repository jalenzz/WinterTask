//
// Created by jalen on 23-1-4.
//
#include "task.h"

void task13() {
    int key, edgeThresh = 20;
    Mat frame, blurMat, grayMat, edgeMat;
    VideoCapture capture(0);

    if (!capture.isOpened()) {
        printf("Open failed");
        return;
    }

    while (true) {
        if (key == (int)'q') break;
        else if (key == (int)'w') edgeThresh++;
        else if (key == (int)'s') edgeThresh--;
        capture >> frame;

        blur(frame, blurMat, Size(5, 5));
        cvtColor( blurMat, grayMat, COLOR_BGR2GRAY );
        Canny(grayMat, edgeMat, edgeThresh, edgeThresh * 3);

        imshow("task13", edgeMat);
        key = waitKey(30);  // 延时30
    }
}
