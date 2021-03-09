/* main.cpp */

/**
 * Author   :   Yuri Meiburg / Maarten Terpstra / Jieying Wang
 * 2011 / 2016 / 2019
 * Rijksuniversiteit Groningen
 
 * imConvert
 *
 */

#include <string>
#include "include/main.hpp"
#include "include/skeletonTracer.h"
#include "include/cornerDetector.h"
#include "include/utility.h"
#include "include/messages.h"
#include "include/evaluation.h"

int MSG_LEVEL = MSG_NORMAL;

void testDetectors() {
	std::string path = "E:\\Belangrijk\\School\\Master thesis\\datasets\\VGG\\";

	skeletons skeletonDetector(1.6, 0, 1, 20000, "Skel", false);
	evaluateDetector(path, skeletonDetector, dataset::VGG);

	std::array<int, 1> windowSizes{ 6 };
	for (const auto w : windowSizes) {
		harris harrisDetector(w, 3, 20000, "Harris", false);//apperture size fixed to 3
		evaluateDetector(path, harrisDetector, dataset::VGG);
	}

	std::array<int, 1> thresholds{ 6 };
	for (const auto threshold : thresholds) {
		fast fastDetector(threshold, cv::FastFeatureDetector::TYPE_9_16, 20000, "fast", false);
		evaluateDetector(path, fastDetector, dataset::VGG);
	}
}

int main(int argc, char **argv) {  
    //Example of how to find corners in an image using the skeleton corner detector
    std::string filename = "in\\boat.pgm";

    auto img1 = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    //Initialize a the detector with saliency 1.6 and stepsize 1.0 (the second parameter does not do anything
    //The island removal threshold is fixed in "computeEndpoints" (utility.cpp)
    skeletons skel(1.6, 0, 1.0f, 10000, "skel", true);

    std::vector<cv::KeyPoint> keyPoints;
    auto corners = skel.detectCorners(filename, { 5,5 },keyPoints);

    //Show the detected corners on the original image, last parameter sets the number of corners to display
    showImage(img1, corners, "", "detectedCorenrs", true, false, 200);
    
    //testDetectors will evaluate the Skel, Harris, and Fast corner detectors on the VGG dataset
    //testDetectors();

    return 0;
}




