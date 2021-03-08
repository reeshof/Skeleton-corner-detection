#pragma once
#include "shared/CUDASkel2D/include/field.h"
#include "include/Node.hpp" 
#include <opencv2/opencv.hpp>
#include "include/image.hpp"

FIELD<float>* readPGM(string name);
void showImage(cv::Mat& image, string directory, string name, string description, bool show, bool writeImage,bool colorCode);
void showImage(FIELD<float>& image, string directory, string name, string description, bool show, bool writeImage, bool colorCode);
void showImage(FIELD<float>& image, std::vector<cv::Point2f>& cornerPoints, string directory, string name, bool show, bool writeImage, int nCorners);
void showImage(cv::Mat& image, std::vector<cv::Point2f>& cornerPoints, string directory, string name, bool show, bool writeImage, int nCorners);

//colors the corners based on whether they are repeated, not repeated, or outside of the second image's boundary
void showCorners(cv::Mat& image, std::vector<cv::Point2f>& cornerPoints, std::vector<short>& isRepeated, std::string name, int nCorners);

void writeHomography(ostream& output, cv::Mat& h);
cv::Mat readHomography(istream& input);
cv::Mat readHomography(std::string& input);
cv::Mat readImage(std::string fileName);
cv::Mat colorCodeImage(cv::Mat& image, int rows, int cols);


template<class T>
void makeFloatArray(T* oldArray, float* newArray, int arraySize);

void makeUINTarray(float* oldarray, char* newArray, int arraySize);
void computeEndpoints(FIELD<float>* im, FIELD<std::vector<cornerData>>& skeletonEndpoints, FIELD<float>& skeletonEndpoints2, float saliency, float stepSize);
void detectSkeletonCorners(FIELD<float>& image, float tau, bool showImage, std::string directoryName, int filterWidth);

