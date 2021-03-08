# Skeleton-corner-detection

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


<!-- GETTING STARTED -->
## Getting Started

A visual studio project is included to build the code. Most of the dependencies are included in the "shared" folder, but Opencv and Cuda have to be installed seperately. Opencv is used 
for comparing the skeleton corner detector with the Harris and Fast detectors, and several other image processing tasks (reading files, converting to grey-scale, etc..). Cuda is used
for the gpu accelerated Augmented Fast Marching Method (AFMM) which computes the skeletons.

### Opencv
Opencv 4.5 is required with the extra modules from opencv_contrib (required for the matching score performance metric). 
1. Download Opencv 4.5 [https://opencv.org/releases/](https://opencv.org/releases/)
2. Download Opencv_contrib [https://github.com/opencv/opencv_contrib](https://github.com/opencv/opencv_contrib)
3. Build Opencv 4.5 and include the modules from the Opencv_contrib repository (follow the instructions from the Opencv_contrib repository). Make sure to not build Opencv_world.lib,
but instead have all the seperate lib files.

### Cuda
Download the correct version of Cuda for your system [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

### Visual studio project
If using the visual studio project provided, make sure to add the paths of the include and library directories of Opencv and Cuda to the 'additional include directories' and 'additional library directories', 
in the property page. The required lib files will already be present as additional dependencies.

<!-- USAGE EXAMPLES -->
## Usage

The following code snippet shows how to detect corners from an image using the skeleton corner detector.

```Cpp
    std::string filename = "in\\boat.pgm";

    auto img1 = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    //Initialize a the detector with saliency 1.6 and stepsize 1.0 (the second parameter does not do anything)
    //The island removal threshold is fixed in "computeEndpoints" (utility.cpp)
    skeletons skel(1.6, 0, 1.0f, 10000, "skel", true);

    std::vector<cv::KeyPoint> keyPoints;
    auto corners = skel.detectCorners(filename, { 5,5 },keyPoints);

    //Show the detected corners on the original image, last parameter sets the number of corners to display
    showImage(img1, corners, "", "detectedCorenrs", true, false, 200);
   ```

