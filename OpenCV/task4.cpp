//
// Created by jalen on 23-1-4.
//
#include "task.h"

void task4() {
    Mat src = imread("../res/test/1.jpg");
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
