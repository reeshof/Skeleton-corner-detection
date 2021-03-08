#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include "include/evaluation.h"
//#include "include/Triple.hpp"
#include "include/cornerDetector.h"
#include <fstream>
#include "include/utility.h"
#include "include/metrics.h"

//using experimental filesystem because the c++17 filesystem is not working
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
#include <iostream>
#include <fstream>
#include <map>

class performanceResult {
private:
	void writeScene(const std::string& key, ofstream& outputFile)  {
		for (const auto& image : detectorResults[key])	writeImage(image, outputFile);
	}

	void writeImage(const std::vector<float>& values, ofstream& outputFile) const {
		for (const auto v : values) outputFile << v << ",";
		outputFile << "\n";
	}

	void writeHeader(const std::string& name, ofstream& outputFile) const {
		outputFile << name << "\n";
	}

	std::vector<float> averageResult() const{
		std::vector<float> averagedMAP(size_);
		for (const auto& scene : detectorResults) {
			for (const auto& image : scene.second) {
				for (int i = 0; i < image.size(); ++i)
					averagedMAP[i] += image[i];
			}
		}
		std::for_each(averagedMAP.begin(), averagedMAP.end(), [this](float& a) {a /= (float)nImages_; });
		return averagedMAP;
	}

	std::map<std::string, std::vector<std::vector<float>>> detectorResults; //for each scene, hold for each image pair a vector with detector results
	std::map<std::string, float> MAPperScene;
	std::map<std::string, float> CoverageperScene;


	int nImages_ = 0;
	int nScenes_ = 0;
	int size_ = 0;

public:
	performanceResult(int size) :size_{size}{};

	void addDetectorResult(const std::string& sceneName, int deformationNr, std::vector<float>& MAPvalues, int imagesInScene) {
		if (!(detectorResults.count(sceneName) > 0))//if key doesnt exist yet create it
			detectorResults[sceneName] = std::vector<std::vector<float>>(imagesInScene-1);
		detectorResults[sceneName][deformationNr] = std::move(MAPvalues);
		++nImages_;
	}

	void addMAPScene(std::string& sceneName, float mapValue) {
		MAPperScene[sceneName] = mapValue;
		++nScenes_;
	}

	void addCoverageScene(std::string& sceneName, float coverage) {
		CoverageperScene[sceneName] = coverage;
	}

	void writeAvgResult(const std::string& path, const std::string& name) const {
		ofstream outputFile;
		outputFile.open(path + name + ".txt");
		writeHeader(name, outputFile);
		writeImage(averageResult(), outputFile);
	}

	void writeMAPCoverageScenes(const std::string& path, const std::string& name) const {
		ofstream outputFile;
		outputFile.open(path + name + ".txt");
		writeHeader(name, outputFile);
		float MMAP = 0;
		for (const auto& x : MAPperScene) {
			MMAP += x.second;
			outputFile << x.first << "," << x.second << "\n";
		}
		outputFile << "Total," << (MMAP / (float)nScenes_) << "\n";
		outputFile << "\n";
		outputFile << "coverge values\n";

		for (const auto& x : CoverageperScene) {
			outputFile << x.first << "," << x.second << "\n";
		}
	}

	void writeResultScenes(const std::string& path, const std::string& name) const {
		ofstream outputFile;
		outputFile.open(path + name + ".txt");
		writeHeader(name, outputFile);
		for (const auto& x : detectorResults) {
			std::vector<float> averagedMAP(size_);
			int totalImages = 0;
			for (const auto& y : x.second) {
				if (!y.empty()) {
					++totalImages;
					for (int i = 0; i < y.size(); ++i) {
						averagedMAP[i] += y[i];
					}
				}
			}
			std::for_each(averagedMAP.begin(), averagedMAP.end(), [totalImages](float& a) {a /= (float)totalImages; });
			outputFile << x.first << "\n";
			writeImage(averagedMAP, outputFile);
		}
	}

	int getNrScenes() const { return detectorResults.size(); }
	std::vector<float>& getVector(string sceneName, int nr) { return detectorResults[sceneName][nr]; }
};

void testDetectors() {
	std::string path = "E:\\Belangrijk\\School\\Master thesis\\datasets\\VGG\\";
	//std::string path = "E:\\Belangrijk\\School\\Master thesis\\datasets\\EF\\";
	//std::string path = "E:\\Belangrijk\\School\\Master thesis\\datasets\\WebCam\\";
	//std::string path = "E:\\Belangrijk\\School\\Master thesis\\datasets\\hpatches-sequences-release\\hpatches-sequences-release\\";
	//std::string path = "E:\\Belangrijk\\School\\Master thesis\\datasets\\hpatches-sequences-release.tar\\hpatches-sequences-release\\";


	/*
	std::array<float, 6> saliencyValues{ 1,1.2,1.4,1.6,1.8,2.0 };
	for (const auto s : saliencyValues) {
		char str[16];
		snprintf(str, sizeof(str), "%.2f", s);
		skeletons skeletonDetector(s, 0, 20000, "Saliency="+string(str), false);
		getImages(path, skeletonDetector, VGG);
	}*/

	//skeletons skeletonDetector(1.6, 0, 1, 20000, "Skel", false);
	//getImages(path, skeletonDetector, VGG);

	
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

	/*
	//harrisLaplace laplace(5, 20000, "laplaceDaisyHP", false);
	//getImages(path, laplace, VGG);

	sift siftDetector(20000, "siftWC", false);
	getImages(path, siftDetector, WEBCAM);*/
}

std::string getFileExtension(const std::string& FileName)
{
	if (FileName.find_last_of(".") != std::string::npos)
		return FileName.substr(FileName.find_last_of(".") + 1);
	return "";
}

std::string getFileWithoutExtension(const std::string& FileName)
{
	if (FileName.find_last_of(".") != std::string::npos)
		return FileName.substr(0, FileName.find_last_of("."));
	return "";
}

void evaluateDetector(const std::string& inputPath, cornerDetector& detector, const dataset data ) {
	namespace fs = std::experimental::filesystem;

	//VARIABLES FOR MAP CALCULATION
	cv::Size nmsSize(8, 8);
	int nCorners = 1000;
	float epsilon = 3.0f;
	int MAPcorners = 100;
	int coverageCorners = 200;
	int failedImages = 0;
	int totalImages = 0;
	float MAP = 0;

	performanceResult repeatabilityEvaluation(nCorners);
	performanceResult matchingEvaluation(nCorners);

	const std::string outputPath = "results\\evaluation\\RepeatabilityAverage\\";
	const std::string outputPath3 = "results\\evaluation\\RepeatabilityScene\\"; 

	const std::string outputPath2 = "results\\evaluation\\MAPCoverage\\";

	const std::string outputPath4 = "results\\evaluation\\MatchingAverage\\";
	const std::string outputPath6 = "results\\evaluation\\MatchingScene\\";

	for (const auto& entry : fs::directory_iterator(inputPath)) {
		int nImages = 6;
		if (data == dataset::EF || data == dataset::WEBCAM) {
			ifstream myfile;
			myfile.open(entry.path().string() + "\\n.txt");
			myfile >> nImages;
			myfile.close();
		}
		std::string mainImage;
		std::vector < std::string > images(nImages-1);//there are 6 images, 5 have homographys relating the transformation from the main image to a subsequent one
		std::vector < std::string > homographys(nImages-1);

		std::string directoryName = entry.path().filename().string();
		std::cout << entry.path().filename().string() << ", number of images: " << nImages << std::endl;
		
		for (const auto& scene : fs::directory_iterator(entry)) {
			std::string filePath = scene.path().string();
			std::string fileExtension =  getFileExtension(filePath);

			if (fileExtension == "txt")
				continue;
			
			if (fileExtension == "ppm" || fileExtension == "pgm") {//the file is one of the images
				std::string fileWithoutExtension = getFileWithoutExtension(filePath);
				size_t last_index = fileWithoutExtension.find_last_not_of("0123456789");
				int imageNr = std::stoi(fileWithoutExtension.substr(last_index + 1));

				if (imageNr == 1) mainImage = filePath;
				else images[imageNr - 2] = filePath;
			}
			
			else {
				char homographyNr;
				switch(data) {
				case dataset::HP: homographyNr = filePath.at(filePath.length() - 1);
					homographys[(int)(homographyNr - '0') - 2] = filePath; break;
				case dataset::WEBCAM: for(auto& h : homographys)h= filePath;break;
				default: homographyNr = filePath.at(filePath.length() - 2);
					homographys[(int)(homographyNr - '0') - 2] = filePath; break;
				}
			}
		}

		std::vector<cv::KeyPoint> firstImagekeypoints;

		auto firstImage = cv::imread(mainImage, cv::IMREAD_GRAYSCALE);
		auto cornersMain = detector.detectCorners(mainImage, nmsSize, firstImagekeypoints);

		float MAPScene = 0;
		float coverageScene = 0;

		for (int i = 0; i < images.size(); i++) {
			std::vector<cv::KeyPoint> secondImageKeypoints;
			auto secondImage = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
			auto cornersSecond = detector.detectCorners(images[i], nmsSize, secondImageKeypoints);
			float ratioRepeated = 0;

			performanceMetric perfMetrics(firstImagekeypoints, secondImageKeypoints, firstImage, secondImage, readHomography(homographys[i]));

			try {
				float AP = perfMetrics.computeAP(MAPcorners);
				float coverage = perfMetrics.computeCoverageScore(secondImageKeypoints, coverageCorners);
				auto results = perfMetrics.computeRepeatabilityMatchingScore(nCorners);

				coverageScene += coverage;
				MAPScene += AP;
				MAP += AP;
				
				repeatabilityEvaluation.addDetectorResult(directoryName, i, results.first,nImages);
				matchingEvaluation.addDetectorResult(directoryName, i, results.second, nImages);

				detector.nImagesCalculated++;
			}
			catch (...) {
				std::cout << "failed with this image: " << entry.path().filename() << " " << i+1 << "\n";
				++failedImages;
			}
			++totalImages;
		}

		MAPScene /= (float)images.size();
		coverageScene /= (float)images.size() + 1;
		std::cout << "MAP for scene " << directoryName << ": " << MAPScene << "| coverage: " << coverageScene << std::endl;
		
		repeatabilityEvaluation.addMAPScene(directoryName, MAPScene);
		repeatabilityEvaluation.addCoverageScene(directoryName, coverageScene);
	}

	MAP /= (float)totalImages;
	std::cout << "MAP = " << MAP << std::endl;

	repeatabilityEvaluation.writeAvgResult(outputPath,detector.getName());
	repeatabilityEvaluation.writeMAPCoverageScenes(outputPath2, detector.getName());
	repeatabilityEvaluation.writeResultScenes(outputPath3, detector.getName());

	matchingEvaluation.writeAvgResult(outputPath4, detector.getName());
	matchingEvaluation.writeResultScenes(outputPath6, detector.getName());

	std::cout << detector.getName() << "| total Images: " << totalImages << " | failed Images: " << failedImages << " | recordedImages: " << detector.nImagesCalculated <<  std::endl;
}



