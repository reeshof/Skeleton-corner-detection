#pragma once
#include<string>
#include<opencv2/opencv.hpp>
#include "Triple.hpp"
#include "cornerDetector.h"

enum class dataset { VGG = 1, HP = 2, EF = 3, WEBCAM };

void testDetectors();
void evaluateDetector(const std::string& inputPath, cornerDetector& detector, const dataset data);

