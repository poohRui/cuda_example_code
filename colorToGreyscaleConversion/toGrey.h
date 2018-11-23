#ifndef TO_GREY_H
#define TO_GREY_H

/**
 * To convert the color image to grey scale
 * 
 * @param Pout  Value of the pixel point in grey scale image
 * @param Pin   Value of the pixel point in color image
 * @width       width of image
 * @height      height of image
 */
void toGreyParallel(unsigned char * Pout, 
                    unsigned char * Pin, 
                    int             width, 
                    int             height);


#endif