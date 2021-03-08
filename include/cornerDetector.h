#pragma once
#include <opencv2/opencv.hpp>
#include "../shared/CUDASkel2D/include/field.h"


auto compareKeypoints = [](cv::KeyPoint& a, cv::KeyPoint& b) {return (a.response > b.response); };

class cornerDetector {
public:
	//detectCorners should return an ordered (descending) list of detected corner points
	virtual std::vector<cv::Point2f> detectCorners(const std::string& image, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints) =0;
	std::string getName() const { return name_; };
	
	
	int nImagesCalculated = 0;

private:
	float distancePoints(const cv::KeyPoint& point1, const cv::KeyPoint& point2);
	float calculateMaximumDistance(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2);

protected:
	cornerDetector(const std::string& name, int nCorners, bool showImage) : name_(name), nCorners_(nCorners), showImage_(showImage) {};

	static std::vector<cv::KeyPoint> findLocalMax(FIELD<float>& image);
	static std::vector<cv::KeyPoint> nms(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Size& nmsSize);

	const std::string name_;
	const int nCorners_;
	const bool showImage_;
};

class harris : public cornerDetector {
public:
	harris(int blockSize, int apertureSize, int nCorners, const std::string& name, bool showImage);
	virtual std::vector<cv::Point2f> detectCorners(const std::string& image, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints);

private:
	const int blockSize_;
	const int apertureSize_;
};

class fast : public cornerDetector {
public:
	fast(int threshold, cv::FastFeatureDetector::DetectorType type, int nCorners, const std::string& name, bool showImage);
	virtual std::vector<cv::Point2f> detectCorners(const std::string& image, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints);

private:
	const int threshold_;
	const cv::FastFeatureDetector::DetectorType type_;
};

class harrisLaplace : public cornerDetector {
public:
	harrisLaplace(int threshold,  int nCorners, const std::string& name, bool showImage);
	virtual std::vector<cv::Point2f> detectCorners(const std::string& image, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints);

private:
	const int threshold_;
};

class sift : public cornerDetector {
public:
	sift(int nCorners, const std::string& name, bool showImage);
	virtual std::vector<cv::Point2f> detectCorners(const std::string& image, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints);

private:
};

class skeletons : public cornerDetector {
public:
	FIELD<float>* fieldImage = nullptr;

	skeletons(float saliency, float tau, float stepSize, int nCorners, const std::string& name, bool showImage);

	virtual std::vector<cv::Point2f> detectCorners(const std::string& image, const cv::Size& nmsSize, std::vector<cv::KeyPoint>& keyPoints);
	std::vector<cv::KeyPoint> findKeypointsScale(const cv::Mat& image, const std::string& imageFilename);

private:
	FIELD<float>* convertMatField(const cv::Mat& matImage) const;
	std::vector<cv::KeyPoint> findKeypoints(const cv::Mat& image) const ;

	const float saliency_;
	const float tau_;
	const float stepSize_;
};
