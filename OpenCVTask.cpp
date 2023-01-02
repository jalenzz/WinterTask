//
// Created by jalen on 22-12-30.
//

#include "OpenCVTask.h"
#include "opencv2/opencv.hpp"
#include "iostream"

using namespace cv;

void task1() {
    Mat img = imread("/home/jalen/code/Visual_Group/WinterTask/res/test/1.jpg");
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

void task2() {
    int key = 0;
    uchar threshold = 100;
    Mat src = imread("/home/jalen/code/Visual_Group/WinterTask/res/test/1.jpg");

    while (key != (int)'q') { // q 退出
        // 深拷贝
        Mat img = src.clone();
        int c = img.cols, r = img.rows;

        // w q 加减阈值
        if(key == (int)'w') threshold += 10;
        else if(key == (int)'s') threshold -= 10;

        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int average = 0;
                for (int k = 0; k < 3; k++) {
                    average += img.at<Vec3b>(i, j)[k] / 3;
                }
                if (average > threshold) average = 255;
                else average = 0;
                img.at<Vec3b>(i, j) = Vec3b (average, average, average);
            }
        }
        namedWindow("task1", 0);
        resizeWindow("task1", 800, 800);
        imshow("task1", img);

        // 获取按键值
        key = waitKey(0);
    }
}

void task4() {
    Mat src = imread("/home/jalen/code/Visual_Group/WinterTask/res/test/1.jpg");
    std::vector<Mat> channels;
    Mat mergeIMG;

    // 通道分离
    split(src, channels);

    Mat Blue = channels.at(0);
    Mat Green = channels.at(1);
    Mat Red = channels.at(2);

    imshow("task4-Blue", Blue);
    imshow("task4-Green", Green);
    imshow("task4-Red", Red);

    // 通道合并
    merge(channels, mergeIMG);
    imshow("task4-merge", mergeIMG);

    waitKey(0);
}

void task5() {
    int key;
    Mat frame;
    VideoCapture capture(0);

    // 判断是否打开
    if (!capture.isOpened()) {
        printf("Open failed");
        return;
    }

    while (true) {
        if (key == (int)'q') break;
        capture >> frame;

        //判断是否读取图片 可能是那个 WARN 的问题，没办法判断
//        if (!frame.empty()) {
//            printf("Get img failed");
//            return;
//        }

        putText(frame,
                std::to_string(capture.get(CAP_PROP_FPS)),  // 文字
                Point(20, 20),        // 第一个字左下角位置
                FONT_HERSHEY_SIMPLEX,   // 字体类型
                1,                     // 字体大小
                CV_RGB(0, 0, 0));         // 字体颜色;
        imshow("task5", frame);
        key = waitKey(30);  // 延时30
    }
}

#define BLACK Scalar(0, 0, 0)

void task6() {
    Mat src = imread("/home/jalen/code/Visual_Group/WinterTask/res/test/1.jpg");

    circle(src, Point(100, 100), 50, BLACK);
    line(src, Point(0, 0), Point(100, 100), BLACK);
    rectangle(src, Point(200, 200), Point(300, 300), BLACK, -1);


    imshow("task6", src);
    waitKey(0);
}

uint8_t GammaTable[256];
void buildGammaTable(float fPrecompensation) {
    for(int i=0; i<256; i++) {
        float f = (i + 0.5) / 256; // 归一化
        float p = pow(f, fPrecompensation); // 预补偿
        GammaTable[i] = p * 256 - 0.5; // 反归一化
    }
}
void task7() {
    float fGamma = 2.2;
    buildGammaTable(1/fGamma);
    Mat src = imread("/home/jalen/code/Visual_Group/WinterTask/res/7-1.png");
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

void task8() {
    Mat HSVMat, Red, Red1, Red2;
    Mat src = imread("/home/jalen/code/Visual_Group/WinterTask/res/8.png");

    cvtColor(src, HSVMat, COLOR_BGR2HSV);
    inRange(HSVMat, Scalar(0, 43, 46), Scalar(10, 255, 255), Red);

    imshow("task8", Red);
    waitKey(0);
}

void task9() {
    Mat src = imread("/home/jalen/code/Visual_Group/WinterTask/res/9.png");
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

void task10() {
    Mat src = imread("/home/jalen/code/Visual_Group/WinterTask/res/10.png");
    Mat srcGray, srcThreshold, labelsMat, stats, centroids;
    // 转为灰度 二值化
    cvtColor( src, srcGray, COLOR_BGR2GRAY );
    threshold(srcGray, srcThreshold, 85, 255, THRESH_BINARY);

    // 检测连通块
    int cnt = connectedComponentsWithStats(srcThreshold, labelsMat, stats, centroids);

    for (int i = 0; i < cnt; ++i) {
        // 用指针遍历
        int *t = stats.ptr<int>(i);
        rectangle(src, Rect(*t, *(t+1), *(t+2), *(t+3)),
                  Scalar(0, 0, 0), 2);
    }

    std::cout << cnt;
    imshow("task10", src);
    waitKey(0);
}

void task11() {
    Mat src = imread("/home/jalen/code/Visual_Group/WinterTask/res/11.png");
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
