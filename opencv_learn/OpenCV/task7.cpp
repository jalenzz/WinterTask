//
// Created by jalen on 23-1-4.
//
#include "task.h"

uint8_t GammaTable[256];
void buildGammaTable(float fPrecompensation) {
    for(int i=0; i<256; i++) {
        float f = (float)(i + 0.5) / 256; // 归一化
        float p = pow(f, fPrecompensation); // 预补偿
        GammaTable[i] = (uint8_t)(p * 256 - 0.5); // 反归一化
    }
}
void task7() {
    float fGamma = 2.2;
    buildGammaTable(1/fGamma);
    Mat src = imread("../res/7-1.png");
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            for (int k = 0; k < 3; ++k) {
                src.at<Vec3b>(i, j)[k] = GammaTable[src.at<Vec3b>(i, j)[k]];
            }
        }
    }
    imshow("task7", src);
    waitKey(0);
}
