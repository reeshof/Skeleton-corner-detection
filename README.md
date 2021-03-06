# Skeleton-corner-detection

<!-- GETTING STARTED -->
## Getting Started

A visual studio project is included to build the code. Most of the dependencies are included in the "shared" folder, but Opencv and Cuda have to be installed seperately. Opencv is used 
for comparing the skeleton corner detector with the Harris and Fast detectors, and several other image processing tasks (reading files, converting to grey-scale, etc..). Cuda is used
for the gpu accelerated Augmented Fast Marching Method (AFMM) which computes the skeletons.

### Opencv
Opencv 4.5 is required with the extra modules from opencv$\_$contrib (required for the matching score performance metric). This process requires three main steps:
1. Download Opencv 4.5 [https://opencv.org/releases/](https://opencv.org/releases/)
2. Download Opencv_contrib [https://github.com/opencv/opencv_contrib](https://github.com/opencv/opencv_contrib)
3. Build Opencv 4.5 and include the modules from the Opencv_contrib repository (follow the instructions from the Opencv_contrib repository). Make sure to not build Opencv_world.lib,
but instead have all the seperate lib files.

### Cuda
Cuda is a parallel computing platform and programming model developed by NVIDIA. The correct version of Cuda for the system in use has to be downloaded and installed  [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

### Visual studio project
If using the visual studio project provided, make sure to add the paths of the include and library directories of Opencv and Cuda to the 'additional include directories' and 'additional library directories', 
in the property page. The required lib files will already be present as additional dependencies.

<!-- USAGE EXAMPLES -->
## Usage

The following code snippet shows how to detect corners from an image using the skeleton corner detector:

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
   
A detector can be evaluated on the [VGG dataset](https://www.robots.ox.ac.uk/~vgg/research/affine/), using the code below:
```Cpp
  std::string path = "datasets\\VGG\\";//change this to the path of the VGG dataset
  
  skeletons skeletonDetector(1.6, 0, 1, 20000, "Skel", false);
  evaluateDetector(path, skeletonDetector, dataset::VGG);
 ```
Two python scripts to generate the repeatability and matching graphs is supplied in the results folder. To run these scripts the packages 'numpy' and 'matplotlib' are required and the desired outputpath has to be changed inside the script.
