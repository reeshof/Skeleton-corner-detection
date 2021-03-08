#include "include/metrics.h"
//#include "include/opencvtest.h"
#include <opencv2/xfeatures2d.hpp>

//distance threshold used to determine if corners are repeated
const float eps = 3.0f;

//returns true if point is outside of the image boundaries
bool outOfImage(const cv::Point2f& point, const cv::Mat& image) {
	if (point.x < 0 || point.x > image.cols)
		return true;
	if (point.y < 0 || point.y > image.rows)
		return true;
	return false;
}

float euclidDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
	float part1 = (p1.x - p2.x) * (p1.x - p2.x) +  (p1.y - p2.y) * (p1.y - p2.y);
	float distance = sqrt(part1);
	return distance;
}

float euclidDistance(const cv::Mat& vec1, const cv::Mat& vec2) {
	float sum = 0.0;
	int dim = vec1.cols;
	for (int i = 0; i < dim; i++) {
		sum += (vec1.at<float>(0, i) - vec2.at<float>(0, i)) * (vec1.at<float>(0, i) - vec2.at<float>(0, i));
	}
	return sum;
}

float calculateAP(const std::vector<cv::KeyPoint>& keyPoints1, const std::vector<cv::KeyPoint>& keyPoints2, const int N) {
	//N is the number of corners that need to be found
	std::vector<int> firstIndexMatch(keyPoints1.size(), keyPoints1.size());//the default value (nCorners) will be one more than the maximum index (max index corner = nCorners - 1)
	
	for (int i = 0; i < keyPoints1.size(); i++) {
		for (int j = 0; j < keyPoints2.size(); j++) {
			double distance = euclidDistance(keyPoints1[i].pt, keyPoints2[j].pt);
			if (distance <= eps) {//only counts as a match if the distance is less than epsilon
				firstIndexMatch[i] = j;
				break;//found the first match so no need to look further
			}
		}
	}
	
	std::vector<bool> isRepeated(keyPoints1.size());
	for (int i = 0; i < keyPoints1.size(); i++) {
		if (firstIndexMatch[i] <= i) isRepeated[i] = true; //it was repeated in the first i corners
		else isRepeated[i] = false;
	}

	if (std::count(isRepeated.begin(), isRepeated.end(), true) < N) {
		std::cout << "not enough repeated corners, returniing  0";
		return 0.0f;
	}

	std::vector<float> repeatability(keyPoints1.size());
	float lastMatches = 0.0f;
	float totalRepeatability = 0.0f;
	for (int i = 0; i < keyPoints1.size(); i++) {
		if (isRepeated[i]) ++lastMatches;//it was repeated in the first i corners
			
		float currentRepeatability = lastMatches / (float)(i + 1);
		repeatability[i] = currentRepeatability;//i + 1 is the number of corners considered for the repeatability
	}

	float AP = 0;
	for (int i = 0, nRepeatedCorners = 0; nRepeatedCorners < N; i++) {
		if (isRepeated[i]) {
			AP += repeatability[i];
			nRepeatedCorners++;
		}
	}

	AP /= (float)N;

	return AP;
}

std::pair<std::vector<float>, std::vector<float>> findMatches(const std::vector<cv::KeyPoint>& keyPoints1, const std::vector<cv::KeyPoint>& keyPoints2, const cv::Mat& descriptors1, const cv::Mat& descriptors2, const std::vector < cv::Point2f>& corners2, int N) {

	std::vector<float> nMatches(keyPoints1.size(),0);
	std::vector<float> repeatability(keyPoints1.size(), 0);

	//for each keypoint in first image, for each keypoint in the second image find if its closes neighbor in feature space is also within epsilon distance
	for (int i = 0; i < N; i++) {
		float smallestDistance = std::numeric_limits<float>::max();
		int currentCorrect = 0;
		int repeated = 0;
		cv::Mat& descr1 = descriptors1.row(i);

		//std::cout << i << ": " << std::endl;
		for (int j = 0; j < N; j++) {
			float featDistance = euclidDistance(descr1, descriptors2.row(j));
			float pointDistance = euclidDistance(keyPoints1[i].pt, corners2[j]);

			//check if its repeated
			if (pointDistance <= eps) repeated = 1;

			if (featDistance <= smallestDistance) {//If it has the smallest distance, its a match
				if (pointDistance <= eps) currentCorrect = 1;//If it also has distance below epsilon, its a correct match
				else currentCorrect = 0;

				smallestDistance = featDistance;
			}
			//If this index is not a match then the correctness of the currently closest neighbor remains true, i.e. if it was a correct match then this keyPoints[i] has a correct match when j number of corners are considered
			if (j >= i) {
				nMatches[j] += currentCorrect;
				repeatability[j] += repeated;
			}
		}
	}

	for (int i = 0; i < repeatability.size(); ++i) { repeatability[i] /= (float)(i + 1);}
	for (int i = 0; i < nMatches.size(); ++i) {	nMatches[i] /= (float)(i + 1);	}

	return { repeatability,nMatches };
}

performanceMetric::performanceMetric(const std::vector<cv::KeyPoint>& keyPoints1, const std::vector<cv::KeyPoint>& keyPoints2, cv::Mat& img1, cv::Mat& img2, const cv::Mat& H) : img1_(img1),img2_(img2){
	const int nCorners = 5000;//Number of corners that will be transformed
	
	//Get an array of the corner locations for transforming using homography
	std::vector < cv::Point2f> corners1(keyPoints1.size());
	for (int i = 0; i < corners1.size(); i++)
		corners1[i] = keyPoints1[i].pt;

	std::vector < cv::Point2f> corners2(keyPoints2.size());
	for (int i = 0; i < corners2.size(); i++)
		corners2[i] = keyPoints2[i].pt;

	//transform all the corners locations using the homography matrix
	cv::perspectiveTransform(corners1, corners1, H);
	cv::perspectiveTransform(corners2, corners2, H.inv());

	//Remain nCorners whose transformation in to the other image are within the image bounds
	keyPointsInImage1_ = std::vector<cv::KeyPoint>(nCorners);
	cornersTransformed1_ = std::vector<cv::Point2f>(nCorners);
	int index1 = 0;
	for (int i = 0; i < corners1.size() && index1 < nCorners; ++i) {
		if (!outOfImage(corners1[i], img2)) {
			cornersTransformed1_[index1] = corners1[i];
			keyPointsInImage1_[index1] = keyPoints1[i];
			++index1;
		}
	}

	keyPointsInImage2_ = std::vector<cv::KeyPoint>(nCorners);
	keyPointsInImage2Transformed_ = std::vector<cv::KeyPoint>(nCorners);
	cornersTransformed2_ = std::vector<cv::Point2f>(nCorners);
	int index2 = 0;
	for (int i = 0; i < corners2.size() && index2 < nCorners; ++i) {
		if (!outOfImage(corners2[i], img1)) {
			cornersTransformed2_[index2] = corners2[i];
			keyPointsInImage2Transformed_[index2] = cv::KeyPoint(corners2[i], regionSize, -1, keyPoints2[i].response);
			keyPointsInImage2_[index2] = keyPoints2[i];
			//keyPointsInImage2[index].pt = corners2[i];
			++index2;
		}
	}

	if (index1 < 1000 || index2 < 1000)
		std::cout << "not enough corners: " << index1 << " " << index2 << std::endl;
}

std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> performanceMetric::getRepeatedIndices() const {
	std::vector<int> firstIndexMatch(keyPointsInImage1_.size(), keyPointsInImage1_.size());//the default value (nCorners) will be one more than the maximum index (max index corner = nCorners - 1)
	const float eps = 3.0f;

	for (int i = 0; i < keyPointsInImage1_.size(); i++) {
		for (int j = 0; j < keyPointsInImage2Transformed_.size(); j++) {
			double distance = euclidDistance(keyPointsInImage1_[i].pt, keyPointsInImage2Transformed_[j].pt);
			if (distance <= eps) {//only counts as a match if the distance is less than epsilon
				firstIndexMatch[i] = j;
				break;//found the first match so no need to look further
			}
		}
	}

	int nCorners = 200;//number corners used to determine if repeated
	std::vector<cv::KeyPoint> repeatedKeypoints1;
	std::vector<cv::KeyPoint> repeatedKeypoints2;
	for (int i = 0; i < nCorners; i++) {
		if (firstIndexMatch[i] <= nCorners) {
			repeatedKeypoints1.push_back(keyPointsInImage1_[i]);
			repeatedKeypoints2.push_back(keyPointsInImage2_[firstIndexMatch[i]]);
		}
	}

	return { repeatedKeypoints1,repeatedKeypoints2 };
}

std::pair<std::vector<float>, std::vector<float>> performanceMetric::computeRepeatabilityMatchingScore(const int N) 
{
	for (auto& keyPoint : keyPointsInImage1_) {
		keyPoint.size = regionSize;//set the size of the region to calculate the desriptor over, diameter of 41 pixels
		keyPoint.angle = 0;
	}
	for (auto& keyPoint : keyPointsInImage2_) {
		keyPoint.size = regionSize;//set the size of the region to calculate the desriptor over, diameter of 41 pixels
		keyPoint.angle = 0;
	}

	//next step to calculate the descriptors of all the corners
	cv::Mat descriptors1;
	cv::Mat descriptors2;

	auto daisyDetector = cv::xfeatures2d::DAISY::create(15, 3, 8, 8, cv::xfeatures2d::DAISY::NRM_FULL, cv::noArray(), true, false);
	daisyDetector->compute(img1_, keyPointsInImage1_, descriptors1);
	daisyDetector->compute(img2_, keyPointsInImage2_, descriptors2);

	//findmatches will return the repeatability results in the first element of the pair, and matching in the second element. 
	auto result1 = findMatches(keyPointsInImage1_, keyPointsInImage2_, descriptors1, descriptors2, cornersTransformed2_, N);
	auto result2 = findMatches(keyPointsInImage2_, keyPointsInImage1_, descriptors2, descriptors1, cornersTransformed1_, N);

	//average the vectors and put in result1
	for (int i = 0; i < result1.first.size(); ++i) { result1.first[i] = (result1.first[i] + result2.first[i]) * 0.5f;}

	for (int i = 0; i < result1.second.size(); ++i) { result1.second[i] = (result1.second[i] + result2.second[i]) * 0.5f;}

	return result1;
}

float performanceMetric::computeCoverageScore(const std::vector<cv::KeyPoint>& corners, const int N) const
{
	assert(corners.size() >= N);

	float sumHaromincMeans = 0;

	for (int i = 0; i < corners.size() && i < N; i++) {
		float totalDistances = 0;

		for (int j = 0; j < corners.size() && j < N; j++) {
			if (i != j) {
				float distance = euclidDistance(corners[i].pt, corners[j].pt);
				totalDistances += (1.0f / distance);
			}
		}

		float D = (float)(N - 1) / totalDistances;
		sumHaromincMeans += (1.0f / D);
	}

	float coverage = (float)N / sumHaromincMeans;

	return coverage;
}

float performanceMetric::computeAP(const int N) const
{
	float AP = calculateAP(keyPointsInImage1_, keyPointsInImage2Transformed_, N);

	return AP;
}