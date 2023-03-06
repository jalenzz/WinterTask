//
// Created by jalen on 23-1-4.
//
#include "task.h"

void task2and3() {
    int key = 0;
    uchar threshold = 100;
    Mat src = imread("../res/test/1.jpg");

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
