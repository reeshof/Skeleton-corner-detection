#include <opencv2/opencv.hpp>
#include "include/utility.h"
#include <opencv2/features2d.hpp>

#include "shared/CUDASkel2D/include/field.h"
#include "include/image.hpp"
#include "include/metrics.h"

//#include "include/image.h"
#include "include/Node.hpp"
#include "include/connected.hpp"
#include "include/evaluation.h"
#include "include/cornerDetector.h"
#include <direct.h>

void writeHomography(ostream& output, cv::Mat& h) {
	for (int i = 0; i < h.cols; ++i) {
		for (int j = 0; j < h.rows; ++j) {
			output << h.at<float>(i, j) << " ";
		}
		output << "\n";
	}
}

cv::Mat readHomography(istream& input) {
	float a, b, c;
	float* values = new float[9];
	int i = 0;
	for (int j = 0; j < 3; ++j) {
		input >> a >> b >> c;
		values[i] = a;
		values[i + 1] = b;
		values[i + 2] = c;
		i += 3;
	}

	return cv::Mat(3, 3, CV_32FC1, values);;
}

cv::Mat readHomography(std::string& filename) {
	std::ifstream infile(filename);

	float a, b, c;
	float* values = new float[9];
	int i = 0;

	if (!infile.is_open()) {
		std::cout << "failed to open " << filename << '\n';
		exit(-1);
	}
	else {
		while (infile >> a >> b >> c) {
			values[i] = a;
			values[i + 1] = b;
			values[i + 2] = c;
			i += 3;
		}
	}

	cv::Mat H = cv::Mat(3, 3, CV_32FC1, values);

	return H;
}

void makeUINTarray(float* oldarray, char* newArray, int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		newArray[i] = (char)oldarray[i];
	}
}

void checkNodeForEndpoint(skel_tree_t* node, FIELD<float>& skeletonEndpoints) {
	int nChildren = node->numChildren();

	if (nChildren == 0) {//found a skeleton endpoint
		auto nodeValue = node->getValue();
		skeletonEndpoints.value(nodeValue.first, nodeValue.second)++;
		return;
	}

	for (int i = 0; i < nChildren; i++) {//travel through the tree
		checkNodeForEndpoint(node->getChild(i), skeletonEndpoints);
	}
}

void addSkeletonEndpoints(vector<std::pair<int, skel_tree_t*>>* forest, FIELD<float>& skeletonEndpoints) {
	std::cout << "Finding the skeleton endPoints from the forest..." << std::endl;

	for (int i = 0; i < forest->size(); i++) {//each element of forest contains the single skeleton of a threshold layer
		skel_tree_t* layer_skeleton = forest->at(i).second;//the layer skeleton represents the root node (-1,-1,-1)

		for (int j = 0; j < layer_skeleton->numChildren(); j++) {//Each child of the root note is the first skeleton point of an extremal region, thus numChildren = num extremal regions (connected components)
			skel_tree_t* child = layer_skeleton->getChild(j);
			if (child->numChildren() == 1) {//The first child of the root node is a skeleton endPoint if it has only 1 child
				auto nodeValue = child->getValue();//x,y,dt
				skeletonEndpoints.value(nodeValue.first, nodeValue.second)++;//count it as one skeleton endpoint
			}

			checkNodeForEndpoint(child, skeletonEndpoints);
		}
	}
}

void writeMatImage(cv::Mat& image, string directory, string name, string description) {
	cv::imwrite(directory + "/" + name + "" + description + ".png", image);
}

//shows image with points drawn on it and saves it
void showImage(FIELD<float>& image, std::vector<cv::Point2f>& cornerPoints, string directory, string name, bool show, bool writeImage, int nCorners) {
	FIELD<float>* dupeImage = image.dupe();

	cv::Mat test2 = cv::Mat(image.dimY(), image.dimX(), CV_32FC1, dupeImage->data());
	cv::Mat test; //= cv::Mat(image.dimY(), image.dimX(), CV_8U, charArray);
	double min, max;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(test2, &min, &max, &min_loc, &max_loc);
	test2 -= min;


	test2.convertTo(test, CV_8U, 255.0 / (max - min));
	//test /= 255;
	cv::Mat grayRBG;//image that is drawn on

	cv::cvtColor(test, grayRBG, cv::COLOR_GRAY2BGR);

	for (int i = 0; i < nCorners && i < cornerPoints.size(); i++) {
		//std::cout << "drawing circle at: " << cornerPoints[i].x << " | " << cornerPoints[i].y << std::endl;
		cv::circle(grayRBG, cv::Point(cornerPoints[i].x, cornerPoints[i].y), 5, cv::Scalar(255, 0, 0), 2);
	}
	if (writeImage)
		writeMatImage(grayRBG, directory,name, "");

	if (show) {
		cv::imshow(name, grayRBG);
		cv::waitKey();
	}
	
	//delete[] charArray;
	delete dupeImage;
}


void showImage(cv::Mat& image, std::vector<cv::Point2f>& cornerPoints, string directory, string name, bool show, bool writeImage, int nCorners) {
	cv::Mat test2 = image.clone();

	cv::Mat test; //= cv::Mat(image.dimY(), image.dimX(), CV_8U, charArray);
	double min, max;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(test2, &min, &max, &min_loc, &max_loc);
	test2 -= min;


	test2.convertTo(test, CV_8U, 255.0 / (max - min));
	//test /= 255;
	cv::Mat grayRBG;//image that is drawn on

	cv::cvtColor(test, grayRBG, cv::COLOR_GRAY2BGR);

	for (int i = 0; i < nCorners && i < cornerPoints.size(); i++) {
		//std::cout << "drawing circle at: " << cornerPoints[i].x << " | " << cornerPoints[i].y << std::endl;
		cv::circle(grayRBG, cv::Point(cornerPoints[i].x, cornerPoints[i].y), 5, cv::Scalar(255, 0, 0), 2);
	}
	if (writeImage)
		writeMatImage(grayRBG, directory, name, "");

	if (show) {
		cv::imshow(name, grayRBG);
		cv::waitKey();
	}
}

void showImage(cv::Mat& image, std::vector<cv::KeyPoint>& cornerPoints, string directory, string name, bool show, bool writeImage, int nCorners) {
	cv::Mat test2 = image.clone();

	cv::Mat test; //= cv::Mat(image.dimY(), image.dimX(), CV_8U, charArray);
	double min, max;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(test2, &min, &max, &min_loc, &max_loc);
	test2 -= min;


	test2.convertTo(test, CV_8U, 255.0 / (max - min));
	//test /= 255;
	cv::Mat grayRBG;//image that is drawn on

	cv::cvtColor(test, grayRBG, cv::COLOR_GRAY2BGR);

	for (int i = 0; i < nCorners && i < cornerPoints.size(); i++) {
		//std::cout << "drawing circle at: " << cornerPoints[i].x << " | " << cornerPoints[i].y << std::endl;
		cv::circle(grayRBG, cv::Point(cornerPoints[i].pt.x, cornerPoints[i].pt.y), cornerPoints[i].size, cv::Scalar(255, 0, 0), 2);
	}
	if (writeImage)
		writeMatImage(grayRBG, directory, name, "");

	if (show) {
		cv::imshow(name, grayRBG);
		cv::waitKey();
	}
}


void showImage(cv::Mat& image, string directory, string name, string description, bool show, bool writeImage, bool colorCode) {
	cv::Mat test = image.clone();

	double min, max;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(test, &min, &max, &min_loc, &max_loc);

	float avg = cv::mean(test, test > 0)[0];

	std::cout << "Showing " << description << " image, max value: " << max << " | avg = " << avg << std::endl;

	cv::normalize(test, test, 0, 255, cv::NORM_MINMAX);

	cv::Mat test2;
	if (colorCode)
		test2 = colorCodeImage(test, image.rows, image.cols);
	else
		test2 = test;

	if (writeImage)
		writeMatImage(test2, directory, name, description);

	if (show) {
		cv::imshow(name, test2);
		cv::waitKey();
	}
}

cv::Mat colorCodeImage(cv::Mat& image, int rows, int cols) {
	cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);

	int arraySize = rows*cols;
	char* charArray = new char[arraySize];
	makeUINTarray((float*)image.data, charArray, arraySize);
	cv::Mat test = cv::Mat(rows,cols, CV_8U, charArray);

	cv::Mat colorImage;
	cv::applyColorMap(test, colorImage, cv::COLORMAP_INFERNO);

	delete[] charArray;
	return colorImage;
}

void showImage(FIELD<float>& image, string directory, string name, string description, bool show, bool writeImage, bool colorCode) {
	FIELD<float>* dupeImage = image.dupe();//Clone image so that the data does not get removed by cv::Mat (not entirely sure if it does or not

	cv::Mat test = cv::Mat(image.dimY(), image.dimX(), CV_32F, dupeImage->data());

	double min, max;
	cv::Point min_loc, max_loc;
	cv::minMaxLoc(test, &min, &max, &min_loc, &max_loc);
	float avg = cv::mean(test, test > 0)[0];
	std::cout << "Showing image " << name << ", max value: " << max << " | avg = " << avg << std::endl;
	cv::normalize(test, test, 0, 255,cv::NORM_MINMAX);

	cv::Mat test2;
	if (colorCode)
		test2 = colorCodeImage(test, test.rows, test.cols);
	else
		test2 = test;

	if (writeImage)
		writeMatImage(test2, directory, name, description);

	if (show) {
		cv::imshow(name, test2);
		cv::waitKey();
	}

	//delete[] charArray;
	delete dupeImage;
}

cv::Mat readImage(std::string fileName) {
	cv::Mat src = cv::imread(fileName);//read any file extension using opencv
	cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);//conver to greyscale

	return src;
}

void showCorners(cv::Mat& image, std::vector<cv::Point2f>& cornerPoints, std::vector<short>& isRepeated, std::string name, int nCorners) {
	cv::Mat grayRBG;//image that is drawn on

	std::vector<cv::Mat> channels;
	channels.push_back(image); channels.push_back(image); channels.push_back(image);

	cv::merge(channels, grayRBG);

	for (int i = 0; i < nCorners; i++) {
		//std::cout << "drawing circle at: " << cornerPoints[i].x << " | " << cornerPoints[i].y << std::endl;
		short color = isRepeated[i];
		if (color == 0)//not repeated: red circle
			cv::circle(grayRBG, cv::Point(cornerPoints[i].x, cornerPoints[i].y), 5, cv::Scalar(0, 0, 255), 2);
		else if (color == 1)//out of image: blue circle
			cv::circle(grayRBG, cv::Point(cornerPoints[i].x, cornerPoints[i].y), 5, cv::Scalar(255, 0, 0), 2);
		else if (color == 2)//repeated: green circle
			cv::circle(grayRBG, cv::Point(cornerPoints[i].x, cornerPoints[i].y), 5, cv::Scalar(0, 255, 0), 2);
	}

	cv::imshow(name, grayRBG);
	cv::waitKey();
}

//tau = threshold for pixels after convolution with gaussian kernel
void detectSkeletonCorners(FIELD<float>& image, float tau, bool writeImage, std::string directoryName, int filterWidth) {
	//std::cout << "obtaining thresholded density map for image with skeleton points..." << std::endl;
	cv::Mat test = cv::Mat(image.dimY(), image.dimX(), CV_32FC1, image.data());

	auto gausfilter = cv::getGaussianKernel(filterWidth, -1, CV_32F);
	float middlevalue = gausfilter.at<float>((int)(gausfilter.rows / 2));
	float factor = (float)1 / (middlevalue * middlevalue);
	cv::GaussianBlur(test, test, cv::Size(filterWidth, filterWidth), 0, 0, cv::BORDER_ISOLATED);
	//std::cout << middlevalue << " " << factor << std::endl;
	if (factor > 1)
		test *= factor;

	if (writeImage) {
		auto imageDuped = image.dupe();
		auto imageDuped2 = image.dupe();//for visualising endpoints

		cv::Mat test2 = cv::Mat(imageDuped->dimY(), imageDuped->dimX(), CV_32FC1, imageDuped->data());
		cv::Mat test3 = cv::Mat(imageDuped2->dimY(), imageDuped2->dimX(), CV_32FC1, imageDuped2->data());
		cv::threshold(test2, test2, 0, 1, cv::THRESH_BINARY);


		//show density map applied with log function
		cv::GaussianBlur(test3, test3, cv::Size(filterWidth, filterWidth), 0, 0, cv::BORDER_ISOLATED);
		std::for_each(test3.begin<float>(), test3.end<float>()
			, [](float& pixel) {
				if (pixel >= 1)
					pixel = log(pixel);
				else
					pixel = 0;
			});

		if (writeImage)
			showImage(image, directoryName, "", "endPoints", false, writeImage, true);

		if (writeImage)
			showImage(test2, directoryName, "", "endPointsBinary", false, writeImage, false);

		if (writeImage)
			showImage(*imageDuped2, directoryName, "", "densityMapLog", false, writeImage, true);


		if (writeImage)
			showImage(image, directoryName, "", "densityMap", false, writeImage, true);
	}
}

//function that makes a float array required for using the skeleton methods. Most arrays used in opencv are 8 or 16 bits so no information loss will occur probably.
template<class T>
void makeFloatArray(T* oldArray, float* newArray, int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		newArray[i] = (float)oldArray[i];
	}
}

//Read a pgm file, convert it to greyscale and create a FIELD<float> using opencv functions.
FIELD<float>* readPGM(string name) {
	cv::Mat image = cv::imread(name);
	cv::Mat grey;

	cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);

	int size = grey.rows * grey.cols;//number of pixels in the image
	float* floatData = new float[size];//

	makeFloatArray(grey.data, floatData, size);

	FIELD<float>* newImage = new FIELD<float>();
	newImage->setAll(grey.cols, grey.rows, floatData);

	//showImage(*newImage,"test","test",false);

	return newImage;
}

void computeEndpoints(FIELD<float>* im, FIELD<std::vector<cornerData>>& skeletonEndpoints, FIELD<float>& skeletonEndpoints2, float saliency, float stepSize) {
	//image, islandthreshold, layerThreshold, DistinguishableInterval, last two are not relevant for corner detection
	Image* il = new Image(im, 0.05, 100, 5);

	il->removeIslands();
	int firstLayer = 0;
	il->collectEndpoints(&firstLayer, skeletonEndpoints, skeletonEndpoints2, saliency, stepSize);
}

void showSkeletonEndPoints(FIELD<float>& image, string name, string description, bool writeImage ) {
	FIELD<float>* dupeImage = image.dupe();//Clone image so that the data does not get removed by cv::Mat (not entirely sure if it does or not

	cv::Mat test = cv::Mat(image.dimY(), image.dimX(), CV_32F, dupeImage->data());

	cv::dilate(test, test, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));

	showImage(test, name,name, description, true, writeImage, false);
}

int roundNearestFilterSize(const float a) {
	int filtersize = 9;
	if (a < 9)
		filtersize = 7;
	if (a < 7)
		filtersize = 5;
	if (a < 5)
		filtersize = 3;
	if (a < 3)
		filtersize = 1;
	return filtersize;
}
