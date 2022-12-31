//
// Created by jalen on 22-12-30.
//

#include "task.h"
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
