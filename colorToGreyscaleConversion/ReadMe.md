# ColorToGreyscaleConversion
> 这是一个将一个彩色图片转换为灰度图的例子，读取图片像素点数据部分利用opencv处理。
> 分别使用opencv自带的灰度图转换函数和利用cuda的并行处理程序进行转换，转换结果以图片的形式存在项目目录下。

## 函数介绍
toGreyParallel：并行程序的stub函数

colorToGreyscaleConversion：图片转换的kernel函数。

## 编译执行
由于编译过程涉及到opencv，Makefile中涉及的链接路径应视自己情况进行修改。
