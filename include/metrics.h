#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

float euclidDistance(const cv::Point2f& p1, const cv::Point2f& p2);

class performanceMetric {
public:
	performanceMetric(const std::vector<cv::KeyPoint>& keyPoints1, const std::vector<cv::KeyPoint>& keyPoints2, cv::Mat& img1, cv::Mat& img2, const cv::Mat& H);
	std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> getRepeatedIndices() const;

	//Returns the repeatability and matching score for 1 to N number of corners. <repeatability values, matching scores>
	std::pair<std::vector<float>, std::vector<float>> computeRepeatabilityMatchingScore(const int N) ;

	//Returns average precison for N corners 
	float computeAP(const int N) const;

	//Returns the coverage score for a set of corners
	float computeCoverageScore(const std::vector<cv::KeyPoint>& corners, const int N) const;

private:
	std::vector<cv::KeyPoint> keyPointsInImage1_;

	std::vector<cv::KeyPoint> keyPointsInImage2_;
	std::vector<cv::KeyPoint> keyPointsInImage2Transformed_;

	std::vector<cv::Point2f> cornersTransformed1_;
	std::vector<cv::Point2f> cornersTransformed2_;

	cv::Mat& img1_; 
	cv::Mat& img2_;

	const int regionSize = 41;//Radius of region used for calculating descriptor
};
