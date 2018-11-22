//
//  main.cpp
//  colorToGrayscaleConversion
//
//  Created by poohRui on 2018/11/22.
//  Copyright © 2018 poohRui. All rights reserved.
//

#include<iostream>
#include <opencv2\opencv.hpp>
using namespace std;

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
    Mat srcImg = imread("images/cat.jpg");
    int height = srcImg.rows;
    int width = srcImg.cols;

    // Use function of opencv, do serial conversion
    unsigned char* Pin = srcImg.data;
    unsigned char* Pout;

    Mat grayImg(height, width, CV_8UC1, Scalar(0));
    //serial_conversion(serial_Pout, Pin, width, height);

    conversionParallel(Pout, Pin, width, height);
    imshow("test",Pout);
}