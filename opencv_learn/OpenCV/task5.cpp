//
// Created by jalen on 23-1-4.
//
#include "task.h"

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
