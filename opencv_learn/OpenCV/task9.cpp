//
// Created by jalen on 23-1-4.
//
#include "task.h"

using namespace cv;

void task9() {
    Mat src = imread("../res/9.png");
    Mat erodeStruct, dilateStruct, erodeMat, dilateMat, openMat, closeMat;

    // 结构元素
    erodeStruct = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    dilateStruct = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    // 腐蚀
    erode(src, erodeMat, erodeStruct);
    // 膨胀
    dilate(src, dilateMat, dilateStruct);
    // 开运算
    dilate(erodeMat, openMat, dilateStruct);
    // 闭运算
    erode(dilateMat, closeMat, erodeStruct);

    imshow("src", src);
    imshow("erode", erodeMat);
    imshow("dilate", dilateMat);
    imshow("open", openMat);
    imshow("close", closeMat);
    waitKey(0);
}
