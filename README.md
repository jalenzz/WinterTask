# WinterTask

## OpenCV

### 1

3 个通道的值统一改为 3 个通道的平均值

通道的值 `img.at<Vec3b>(i, j)[0-2]`

整体赋值
```c++
img.at<Vec3b>(i, j) = Vec3b(average, average, average);
```

图片变灰

![1-1](./res/Screenshots/1-1.png)

### 2 && 3

简单实现之后想用按键控制 `threshold` 值的增减，更直观的看出变化。
一开始没有多想就直接把修改的那一段放进循环里面，但是发现图像没变化，
后面想起来，从第一次修改之后，通道的值已经发生了变化，图像已经变成黑白的了，此时三通道的数值已经完全相同，不会再改变。

于是就想着再新建一个 `Mat src` 用来读取图片， `Mat img = src`，从 `src` 读取通道的值，修改 `img` 的通道值。
本来以为解决了，但是发现还是没变化。之后突然想到 `Mat` 本应该是没有 `=` 操作的，应该是重载过了。
了解之后发现 `a = b` 只是将 `a` 的各个地址的值改为 `b` 的，在操作的时候两者会一起改变。

感觉 `=` 相当于是引用，而真正地复制可以用 `a = b.clone()` `b.CopyTo(a)`。
`clone()` 实际上是新建一个 `Mat m` 然后执行 `CopyTo`，再返回 `m`。

![1-2](./res/Screenshots/1-2.png)

### 4

```c++
// 通道分离
split(src, channels);

channels.at(i) // 012 BGR

// 通道合并
merge(channels, mergeIMG);
```

### 5

```c++
// 读取摄像头
VideoCapture capture(0);
capture >> frame;

// 判断摄像头是否打开
capture.isOpened()

// 判断图片是否读取成功
frame.empty()
bool Mat::empty() const {
    return data == 0 || total() == 0 || dims == 0;
}

// 在图像上显示文字
putText(img, text, position, fontFace, fontScale, color);

// 获取帧率
capture.get(CAP_PROP_FPS)
```


### 6

```c++
// thickness < 0 颜色填充
void circle(InputOutputArray img, Point center, int radius,
            const Scalar& color, int thickness = 1,
            int lineType = LINE_8, int shift = 0);

void line(InputOutputArray img, Point pt1, Point pt2, const Scalar& color,
          int thickness = 1, int lineType = LINE_8, int shift = 0);


// (Point pt1, Point pt2) -> Rect(x, y, width, height)
void rectangle(InputOutputArray img, Point pt1, Point pt2,
               const Scalar& color, int thickness = 1,
               int lineType = LINE_8, int shift = 0);
```

### 7

Gamma 矫正
```c++
float f = (i + 0.5) / 256; // 归一化
float p = pow(f, fPrecompensation); // 预补偿
GammaTable[i] = p * 256 - 0.5; // 反归一化
```

从显示效果上来看可以让人眼所看到的黑白对比增加，应该是相当于增加了对比度吧，在应用中的化可以让我们看黑暗中的东西看得更清楚

### 8

HSV 转换，颜色提取

```c++
cvtColor(src, HSVMat, COLOR_BGR2HSV);

inRange(hsvMat, Scalar(minH, minS, minV), Scalar(maxH, maxS, maxV), detectMat);
```

### 9

[腐蚀、膨胀、开闭运算](https://zhuanlan.zhihu.com/p/164619939)

```c++
// 获取结构元素
// shape: MORPH_RECT  MORPH_CROSS MORPH_ELLIPSE
// Size: Size(x, y)
Mat getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1));

// erode 腐蚀  dilate 膨胀  kernel 结构元素
void erode( InputArray src, OutputArray dst, InputArray kernel,
Point anchor = Point(-1,-1), int iterations = 1,
int borderType = BORDER_CONSTANT,
const Scalar& borderValue = morphologyDefaultBorderValue() );
```

开运算：先腐蚀后膨胀

闭运算：先膨胀后腐蚀 *欲使其自闭，必先让其膨胀*

相对亮的区域来说,
开运算，把小的断断续续的亮处给断开，这就需要先通过腐蚀去掉小亮点，再通过膨胀恢复原来的大片连续亮区。
闭运算就是把原来断断续续的亮区给闭合，这就需要通过膨胀扩大亮区，之后通过腐蚀恢复大片连续暗区

### 10 && 11

[threshold](https://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/threshold/threshold.html)
```c++
// 二值化
// type: enum ThresholdTypes
double threshold( InputArray src, OutputArray dst,
                  double thresh, double maxval, int type );
```

形态学梯度：膨胀图与腐蚀图之差，提取物体边缘

顶帽：原图像-开运算图，突出原图像中比周围亮的区域

黑帽：闭运算图-原图像，突出原图像中比周围暗的区域

```c++
void morphologyEx( InputArray src, OutputArray dst,
                   int op, InputArray kernel,
                   Point anchor=Point(-1,-1), int iterations=1,
                   int borderType=BORDER_CONSTANT,
                   const Scalar& borderValue=morphologyDefaultBorderValue() );

/* op:
 * MORPH_OPEN – 开运算
 * MORPH_CLOSE – 闭运算
 * MORPH_GRADIENT - 形态学梯度
 * MORPH_TOPHAT - 顶帽
 * MORPH_BLACKHAT - 黑帽
 */
```

```c++
// 用指针遍历
int *t = stats.ptr<int>(i);
```

```c++
// 返回连通块个数
// stats nx5 四边形的 x y width height
int connectedComponentsWithStats(InputArray image, OutputArray labels,
                                 OutputArray stats, OutputArray centroids,
                                 int connectivity = 8, int ltype = CV_32S);
```

### 12

```c++
blur(frame, blurMat, Size size);
medianBlur(frame, medianBlurMat, int size);
GaussianBlur(frame, GaussianBlurMat, Size size, x, y);
```

### 13

[Canny](https://www.aiuai.cn/aifarm482.html)

```c++
Canny(grayMat, edgeMat, edgeThresh, edgeThresh * 3);
```
