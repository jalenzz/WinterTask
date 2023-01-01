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

        //判断是否读取图片
        if (!frame.empty()) {
            printf("Get img failed");
            return;
        }

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
