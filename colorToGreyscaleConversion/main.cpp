//
//  main.cpp
//  colorToGrayscaleConversion
//
//  Created by poohRui on 2018/11/22.
//  Copyright © 2018 poohRui. All rights reserved.
//

#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "toGrey.h"

using namespace std;
using namespace cv;

#define CHANNELS 3

void serial_conversion(unsigned char * Pout, 
                       unsigned,char * Pin, 
                       int             width, 
                       int             height){

    for(int Col = 0;Col < width;Col++){
        for(int Row = 0;Row < height;Row++){
            int greyOffset = Row * width + Col;
            int rgbOffset = greyOffset*CHANNELS;
            unsigned char r = Pin[rgbOffset];
            unsigned char g = Pin[rgbOffset+1];
            unsigned char b = Pin[rgbOffset+2];
            Pout[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
        }
    }
}

int main(){
    
    // Load the image into mat
    Mat srcImg = imread("images/cat.jpg",CV_LOAD_IMAGE_COLOR);
    int height = srcImg.rows;
    int width = srcImg.cols;

    // Use function of opencv, do serial conversion
    Mat greyImg;
    cvtColor(srcImg, greyImg, CV_BGR2GRAY);

    imwrite("greyImg_opencv.jpg", greyImg);

    //uchar3 imgData = srcImg.data;

    unsigned char* Pin = new unsigned char[height*width*CHANNELS];
    for(int i = 0;i<height;i++){
        const uchar* imgData = srcImg.ptr<uchar>(i);
        for(int j = 0;j<width*CHANNELS;j++){
            Pin[i*width*CHANNELS+j] = imgData[j];
        }
    }

    unsigned char* Pout = new unsigned char[height*width];

    toGreyParallel(Pout,Pin,width,height);

    Mat para_greyImg = Mat(height,width,CV_8UC1);

    memcpy(para_greyImg.data,Pout,width*height);

    imwrite("greyImg_parallel.jpg", para_greyImg);

}