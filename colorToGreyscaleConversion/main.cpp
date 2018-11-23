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
#define BLOCK_DIM 16

int main(){
    
    // Load the image into mat
    Mat srcImg = imread("images/cat.jpg",CV_LOAD_IMAGE_COLOR);
    int height = srcImg.rows;
    int width = srcImg.cols;

    // Use function of opencv, do serial conversion
    Mat greyImg;
    cvtColor(srcImg, greyImg, CV_BGR2GRAY);

    // Save the opencv convert result in "greyImg_opencv.jpg"
    imwrite("images/greyImg_opencv.jpg", greyImg);

    // Put Mat data into Pin
    unsigned char* Pin = new unsigned char[height*width*CHANNELS];
    for(int i = 0;i<height;i++){
        const uchar* imgData = srcImg.ptr<uchar>(i);
        for(int j = 0;j<width*CHANNELS;j++){
            Pin[i*width*CHANNELS+j] = imgData[j];
        }
    }

    unsigned char* Pout = new unsigned char[height*width];

    toGreyParallel(Pout,Pin,width,height);

    // Convert Pout to Mat, save the parallel result into "greyImg_parallel.jpg"
    Mat para_greyImg = Mat(height,width,CV_8UC1);
    memcpy(para_greyImg.data,Pout,width*height);

    imwrite("images/greyImg_parallel.jpg", para_greyImg);

}