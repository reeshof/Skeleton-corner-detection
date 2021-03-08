#include "include/cornerDetector.h"
#include "include/utility.h"
#include "include/Image.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <cmath>

float cornerDetector::distancePoints(const cv::KeyPoint& point1, const cv::KeyPoint& point2)
{
	auto p1 = point1.pt;
	auto p2 = point2.pt;

	float x = p1.x - p2.x; //calculating number to square in next step
	float y = p1.y - p2.y;
	float dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
	dist = sqrt(dist);

	return dist;
}

float cornerDetector::calculateMaximumDistance(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2) {
	float minDist = 10000;
	for (const auto& point1 : keypoints1) {
		for (const auto& point2 : keypoints2) {
			if (point1.pt != point2.pt)
				minDist = std::min(distancePoints(point1, point2), minDist);
		}
	}
	return minDist;
}

std::vector<cv::KeyPoint> cornerDetector::nms(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Size& nmsSize) {//nmsSize = size of filter used for NMS (should be uneven)
	cv::Mat dst = cv::Mat(image.size(), CV_32FC1, cv::Scalar(0));
	for (const auto& keypoint : keypoints) {
		dst.at<float>(keypoint.pt.y, keypoint.pt.x) = std::max(keypoint.response, 0.0f);//dst.at(row,collumn) = y,x
	}

	cv::Mat dilateDst = cv::Mat(image.size(), CV_32FC1);
	cv::dilate(dst, dilateDst, cv::getStructuringElement(cv::MORPH_RECT, nmsSize));//dilation with 5x5 rectangular kernel
	dilateDst -= dst;

	int index = 0;
	for (int i = 0; i < dilateDst.cols; i++)
	{
		for (int j = 0; j < dilateDst.rows; j++)
		{
			if (dilateDst.at<float>(j, i) == 0 & dst.at<float>(j, i) > 0)//accessing as (j,i) because mat is accesed by row,col
			{
				dilateDst.at<float>(j, i) += index++;
			}
			else
				dilateDst.at<float>(j, i) = 0;
		}
	}

	cv::Mat dilateDst2 = cv::Mat(image.size(), CV_32FC1);
	cv::dilate(dilateDst, dilateDst2, cv::getStructuringElement(cv::MORPH_RECT, nmsSize));//dilation with 5x5 rectangular kernel
	dilateDst2 -= dilateDst;

	std::vector<cv::KeyPoint> corners;

	for (int i = 0; i < dilateDst2.cols; i++)
	{
		for (int j = 0; j < dilateDst2.rows; j++)
		{
			//if its value in dilateDst2 is 0 then it is a local maximum. If its original response was > 0 then it was also an original point.
			if (dilateDst2.at<float>(j, i) == 0 & dst.at<float>(j, i) > 0)//accessing as (j,i) because mat is accesed by row,col
			{
				corners.push_back(cv::KeyPoint(cv::Point2f(i, j), 0, 0, dst.at<float>(j, i), 0));
			}
		}
	}

	//std::cout << "Maximum distance after NMS = " << calculateMaximumDistance(corners, corners) << std::endl;
	return corners;
}

harris::harris(int blockSize, int apertureSize, int nCorners, const std::string& name, bool showImage) : blockSize_(blockSize), apertureSize_(apertureSize), cornerDetector(name, nCorners, showImage) {
}

std::vector<cv::Point2f> harris::detectCorners(const std::string& imageFilename, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints) {
	cv::Mat image = cv::imread(imageFilename, cv::IMREAD_GRAYSCALE);

	int blockSize = blockSize_;
	int apertureSize = apertureSize_;
	double k = 0.05;//fixed parameter. value recommended to use

	cv::Mat dst = cv::Mat(image.size(), CV_32FC1);

	cv::cornerHarris(image, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

	double min, max;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(dst, &min, &max, &min_loc, &max_loc);

	dst += std::abs(min);//so that all scores are non negative (not sure if this is necessary but feels better

	//dilate the image with harris corner response scores. pixels that are local maximum in a given region do not change when dilated, thus the local maximum will be the pixels where dilateDst-dst == 0
	//important to obtain local maximum, because points around corners will generally have higher scores. Thus one corner could dominate and yield 10+ corner points.
	cv::Mat dilateDst = cv::Mat(image.size(), CV_32FC1);
	cv::dilate(dst, dilateDst, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));//dilation with 5x5 rectangular kernel
	dilateDst -= dst;

	//hold all corners (local maximum)
	//std::vector<cv::KeyPoint> corners;
	std::vector<cv::Point2f> points;//For opencv transform

	for (int i = 0; i < dilateDst.cols; i++)
	{
		for (int j = 0; j < dilateDst.rows; j++)
		{
			if (dilateDst.at<float>(j, i) == 0)//accessing as (j,i) because mat is accesed by row,col
			{
				keyPoints.push_back(cv::KeyPoint(cv::Point2f(i, j), 0, 0, dst.at<float>(j, i), 0));
			}
		}
	}

	//sort in descending order
	keyPoints = nms(image, keyPoints, nmsSize);
	std::sort(keyPoints.begin(), keyPoints.end(), compareKeypoints);

	for (int i = 0; i < nCorners_ && i < keyPoints.size(); i++) {
		points.push_back(keyPoints[i].pt);
		if (showImage_)
			cv::circle(image, cv::Point(keyPoints[i].pt.x, keyPoints[i].pt.y), 8, cv::Scalar(0), 2, 2, 0);
	}

	if (showImage_) {
		cv::imshow("destination", image);
		cv::waitKey();
	}
	//std::vector<cv::Point2f> returnCorners;
	//std::copy(corners.begin(), corners.begin() + 50, std::back_inserter(returnCorners));
	return points;
}

fast::fast(int threshold, cv::FastFeatureDetector::DetectorType type, int nCorners, const std::string& name, bool showImage) : threshold_(threshold), type_(type), cornerDetector(name, nCorners, showImage) {}

std::vector<cv::Point2f> fast::detectCorners(const std::string& imageFilename, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints) {
	//std::vector<cv::KeyPoint> corners;

	cv::Mat image = cv::imread(imageFilename, cv::IMREAD_GRAYSCALE);
	//std::cout << _threshold << std::endl;
	cv::Mat dummy;

	cv::FAST(image, keyPoints, threshold_, true, cv::FastFeatureDetector::TYPE_9_16);

	keyPoints = nms(image, keyPoints, nmsSize);
	std::sort(keyPoints.begin(), keyPoints.end(), compareKeypoints);

	std::vector<cv::Point2f> points;
	for (int i = 0; i < nCorners_ && i < keyPoints.size(); i++) {
		points.push_back(keyPoints[i].pt);
		if (showImage_)
			cv::circle(image, cv::Point(keyPoints[i].pt.x, keyPoints[i].pt.y), 8, cv::Scalar(0), 2, 2, 0);
	}

	if (showImage_) {
		cv::imshow("destination", image);
		cv::waitKey();
	}

	return points;
}

harrisLaplace::harrisLaplace(int threshold, int nCorners, const std::string& name, bool showImage) : threshold_(threshold), cornerDetector(name, nCorners, showImage) {}

std::vector<cv::Point2f> harrisLaplace::detectCorners(const std::string& image, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints)
{
	cv::Mat img = cv::imread(image, cv::IMREAD_GRAYSCALE);
	auto harris = cv::xfeatures2d::HarrisLaplaceFeatureDetector::create();
	harris->detect(img, keyPoints);

	keyPoints = nms(img, keyPoints, nmsSize);
	std::sort(keyPoints.begin(), keyPoints.end(), compareKeypoints);

	for (auto& keyPoint : keyPoints) keyPoint.octave = 0;

	std::vector<cv::Point2f> points;
	for (int i = 0; i < nCorners_ && i < keyPoints.size(); i++) points.push_back(keyPoints[i].pt);

	return points;
}

sift::sift(int nCorners, const std::string& name, bool showImage) : cornerDetector(name, nCorners, showImage) {}

std::vector<cv::Point2f> sift::detectCorners(const std::string& image, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints)
{
	cv::Mat img = cv::imread(image, cv::IMREAD_GRAYSCALE);
	auto siftDetector = cv::SIFT::create();
	siftDetector->detect(img, keyPoints);

	keyPoints = nms(img, keyPoints, nmsSize);
	std::sort(keyPoints.begin(), keyPoints.end(), compareKeypoints);

	for (auto& keyPoint : keyPoints) keyPoint.octave = 0;

	std::vector<cv::Point2f> points;
	for (int i = 0; i < nCorners_ && i < keyPoints.size(); i++) points.push_back(keyPoints[i].pt);

	return points;
}


skeletons::skeletons(float saliency, float tau, float stepSize, int nCorners, const std::string& name, bool showImage) : saliency_(saliency), tau_(tau), stepSize_(stepSize), cornerDetector(name, nCorners, showImage) {}

FIELD<float>* skeletons::convertMatField(const cv::Mat& matImage) const
{
	int size = matImage.rows * matImage.cols;//number of pixels in the image
	float* floatData = new float[size];//

	makeFloatArray(matImage.data, floatData, size);

	FIELD<float>* newImage = new FIELD<float>();
	newImage->setAll(matImage.cols, matImage.rows, floatData);

	return newImage;
}

std::vector<cv::KeyPoint> cornerDetector::findLocalMax(FIELD<float>& image)
{
	//important to obtain local maximum, because points around corners will generally have higher scores. Thus one corner could dominate and yield 10+ corner points.
	cv::Mat dilateDst = cv::Mat(image.dimY(), image.dimX(), CV_32FC1);
	cv::Mat dst = cv::Mat(image.dimY(), image.dimX(), CV_32FC1, image.data());

	cv::dilate(dst, dilateDst, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));//dilation with 5x5 rectangular kernel
	dilateDst -= dst;

	std::vector<cv::KeyPoint> corners;

	for (int i = 0; i < dilateDst.cols; i++)
	{
		for (int j = 0; j < dilateDst.rows; j++)
		{
			if (dilateDst.at<float>(j, i) == 0 && dst.at<float>(j,i)>0)//accessing as (j,i) because mat is accesed by row,col
			{
				corners.push_back(cv::KeyPoint(cv::Point2f(i, j), 0, 0, dst.at<float>(j, i), 0));
			}
		}
	}

	return corners;
}

std::vector<cv::KeyPoint> skeletons::findKeypointsScale(const cv::Mat& image, const std::string& imageFilename)
{
	fieldImage = readPGM(imageFilename);

	int x = fieldImage->dimX();
	int y = fieldImage->dimY();

	std::vector<cornerData>* dummyData = new std::vector<cornerData>[x * y];
	float* dummyData2 = new float[x * y]{ 0 };

	FIELD<std::vector<cornerData>> skeletonEndpoints{};//initialize image that holds skeleton endpoints.
	FIELD < float > skeletonEndpoints2{};

	skeletonEndpoints.setAll(x, y, dummyData);
	skeletonEndpoints2.setAll(x, y, dummyData2);

	std::vector<int> releventLayers(0);

	//il->calculateImportance();
	std::vector<std::string> setDirectories;
	computeEndpoints(fieldImage, skeletonEndpoints, skeletonEndpoints2, saliency_, stepSize_);
	FIELD<float>* imageDupe = skeletonEndpoints2.dupe();

	detectSkeletonCorners(skeletonEndpoints2, tau_, showImage_, "dummy", 1);

	std::vector<cv::KeyPoint> corners = findLocalMax(skeletonEndpoints2);

	std::sort(corners.begin(), corners.end(), compareKeypoints);

	for (int i = 0; i < corners.size(); ++i) {
		int cornerx = corners[i].pt.x;
		int cornery = corners[i].pt.y;

		auto endpointData = skeletonEndpoints.value(cornerx, cornery);

		float avgSize = 0;
		for (const auto& data : endpointData) avgSize += data.distance;
		avgSize /= endpointData.size();

		corners[i].size = avgSize;
	}

	delete fieldImage;
	delete imageDupe;

	return corners;
}

std::vector<cv::KeyPoint> skeletons::findKeypoints(const cv::Mat& image) const
{
	int x = fieldImage->dimX();
	int y = fieldImage->dimY();

	std::vector<cornerData>* dummyData = new std::vector<cornerData>[x * y];
	float* dummyData2 = new float[x * y]{ 0 };

	FIELD<std::vector<cornerData>> skeletonEndpoints{};//initialize image that holds skeleton endpoints.
	FIELD < float > skeletonEndpoints2{};

	skeletonEndpoints.setAll(x, y, dummyData);
	skeletonEndpoints2.setAll(x, y, dummyData2);

	std::vector<int> releventLayers(0);

	//il->calculateImportance();
	std::vector<std::string> setDirectories;
	computeEndpoints(fieldImage, skeletonEndpoints, skeletonEndpoints2, saliency_, stepSize_);
	
	FIELD<float>* imageDupe = skeletonEndpoints2.dupe();

	detectSkeletonCorners(skeletonEndpoints2, tau_, showImage_, "dummy", 5);
	std::vector<cv::KeyPoint> corners = findLocalMax(skeletonEndpoints2);

	delete fieldImage;
	delete imageDupe;

	return corners;
}

std::vector<cv::Point2f> skeletons::detectCorners(const std::string& imageFilename, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints)
{
	std::vector<cv::Point2f> corners;

	fieldImage = readPGM(imageFilename);

	cv::Mat image = cv::imread(imageFilename, cv::IMREAD_GRAYSCALE);
	keyPoints = findKeypoints(image);

	//Stuff for assigning a harris value to each pixel
	/*
	int blockSize = 6;
	int apertureSize = 3;
	double k = 0.04;
	cv::Mat dst = cv::Mat(image.size(), CV_32FC1);
	cv::cornerHarris(image, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
	for (auto& keypoint : cornersFirstCombined) {
		keypoint.response = dst.at<float>(keypoint.pt.y, keypoint.pt.x);
	}*/

	keyPoints = nms(image, keyPoints, nmsSize);
	std::sort(keyPoints.begin(), keyPoints.end(), compareKeypoints);

	for (int i = 0; i < keyPoints.size(); i++) {//add the sorted cornerpoints
		corners.push_back(cv::Point2f(keyPoints[i].pt.x, keyPoints[i].pt.y));
	}

	return corners;
}


