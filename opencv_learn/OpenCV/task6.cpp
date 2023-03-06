//
// Created by jalen on 23-1-4.
//
#include "task.h"

#define BLACK Scalar(0, 0, 0)

void task6() {
    Mat src = imread("../res/test/1.jpg");

    circle(src, Point(100, 100), 50, BLACK);
    line(src, Point(0, 0), Point(100, 100), BLACK);
    rectangle(src, Point(200, 200), Point(300, 300), BLACK, -1);


    imshow("task6", src);
    waitKey(0);
}
