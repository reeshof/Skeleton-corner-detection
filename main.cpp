/* main.cpp */

/**
 * Author   :   Yuri Meiburg / Maarten Terpstra / Jieying Wang
 * 2011 / 2016 / 2019
 * Rijksuniversiteit Groningen
 
 * imConvert
 *
 * This program can read PGM files, and convert them to the SIR method,
 * which stands for "Skeletonal Image Representation".
 *
 * To view SIR-files, use imShow.
 *
 */

#include <math.h>
#include "shared/fileio/fileio.hpp"
#include <string>
//#include <sys/resource.h>
#include "include/main.hpp"
#include <omp.h>
#include <vector>
#include "include/Image.hpp"
#include "include/io.hpp"
#include "include/messages.h"
#include "include/image.h"
#include "include/ImageWriter.hpp"
#include "shared/configParser/include/Config.hpp"
#include <chrono>
#include <boost/algorithm/string.hpp>
#include <fstream>

#include "include/evaluation.h"
#include "include/skeletonTracer.h"
#include "include/cornerDetector.h"
#include "include/utility.h"
using namespace std;


/* All properties that can be set in the config files: */
/* For a more detailed explanation, see our example.conf */
int MSG_LEVEL = MSG_NORMAL;
string filename_stdstr;
string compress_method;
float islandThreshold = 0;
float layerThreshold = 0;
int DistinguishableInterval = 0;
COLORSPACE c_space;
extern string OUTPUT_FILE;
extern string COMP_METHOD_NAME;
extern string ENCODING;
extern bool MSIL_SELECTION;
// Overlap pruning
extern bool OVERLAP_PRUNE;
extern float OVERLAP_PRUNE_EPSILON;
// Bundling variables
extern bool BUNDLE;
extern int EPSILON;
extern float ALPHA;
extern int COMP_LEVEL;
/* Tree variables */
extern int MIN_OBJECT_SIZE;
extern int MIN_SUM_RADIUS;
extern int MIN_PATH_LENGTH;

// Set special colorspace in case of color image
// Defaults to RGB when an color image is encountered, otherwise to gray
COLORSPACE set_color_space(string c_space) {
    if (boost::iequals(c_space, "hsv")) {
        return COLORSPACE::HSV;
    } else if (boost::iequals(c_space, "ycbcr") || boost::iequals(c_space, "yuv")) {
        return COLORSPACE::YCC;
    } else if (boost::iequals(c_space, "rgb")) {
        return COLORSPACE::RGB;
    } else {
        return COLORSPACE::NONE;
    }
}
/* Set all parameters according to the configuration file parsed
 * with the Config object */


vector<std::pair<int, skel_tree_t*>>* execute_skeleton_pipeline(FIELD<float>* im, int *firstLayer) {//firstlayer = 0
   
    Image* il = new Image(im, islandThreshold, layerThreshold, DistinguishableInterval);
    
    il->removeIslands();
    il->calculateImportance();
    //il->removeLayers();

    auto begin = std::chrono::steady_clock::now();
    vector<std::pair<int, skel_tree_t*>>* forest;// = il->computeSkeletons(firstLayer, *im, 3);//CHANGES THIS TO *im DOESNT WORK NOW
    auto end = std::chrono::steady_clock::now();
    auto diff_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    auto diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Computing skeletons took = " << diff_seconds << " s (" << diff_ms << " ms)" << std::endl;
    
    delete il;
    return forest;
}

string colorspace_to_string() {
    switch (c_space) {
    case GRAY:
        return string("Gray");
    case RGB:
        return string("RGB");
    case HSV:
        return string("HSV");
    case YCC:
        return string("YCbCr");
    default:
        return string("NONE! (FIX THIS)");
    }
}

float get_min_elem(FIELD<float>* im) {
    float min_elem = 1e5;
    int nPix = im->dimX() * im->dimY();
    float *c = im->data();
    float *end = im->data() + nPix;
    while (c != end)
        min_elem = min(min_elem, *c++);
    return min_elem;
}

void execute_gray_pipeline(FIELD<float>* im) {
    int clear_color = get_min_elem(im);
    cout<<"clear_color "<<clear_color<<endl;
    int firstLayer = 0;
    vector<std::pair<int, skel_tree_t*>>* forest = execute_skeleton_pipeline(im, &firstLayer);
    PRINT(MSG_NORMAL, "Creating ImageWriter object...\n");
    ImageWriter iw(OUTPUT_FILE.c_str());
    if (c_space == COLORSPACE::NONE)
        c_space = COLORSPACE::GRAY;
    PRINT(MSG_NORMAL, "Using colorspace %s\n", colorspace_to_string().c_str());
    iw.writeHeader(im->dimX(), im->dimY(), c_space, clear_color, firstLayer, 0 , 0);
    iw.write_image(forest);
    delete im;
}

void execute_color_pipeline(IMAGE<float>* im) {
    for (float* r = im->r.data(), *g = im->g.data(), *b = im->b.data(), *rend = im->r.data() + im->dimX() * im->dimY(); r < rend; ++r, ++g, ++b) {
        unsigned char r_prime = static_cast<unsigned char>((*r) * 255.0);
        unsigned char g_prime = static_cast<unsigned char>((*g) * 255.0);
        unsigned char b_prime = static_cast<unsigned char>((*b) * 255.0);
        unsigned char Y, Cb, Cr;
        float H, S, V;
        double max_val, min_val, delta;
        switch (c_space) {
        case COLORSPACE::YCC:
        //cout<<"color ycc"<<endl;
            Y  = min(max(0.0, round( 0.299  * r_prime + 0.587  * g_prime + 0.114  * b_prime      )), 255.0);
            Cb = min(max(0.0, round(-0.1687 * r_prime - 0.3313 * g_prime + 0.5    * b_prime + 128)), 255.0);
            Cr = min(max(0.0, round( 0.5    * r_prime - 0.4187 * g_prime - 0.0813 * b_prime + 128)), 255.0);
            r_prime = Y;
            g_prime = Cb;
            b_prime = Cr;
            break;
        case COLORSPACE::HSV:
            min_val = fmin(*r, fmin(*g, *b));
            max_val = fmax(*r, fmax(*g, *b));
            delta = max_val - min_val;
            H = 0.0;
            V = max_val;
            S = max_val > 1e-6 ? delta / max_val : 0.0f;

            if (S > 0.0f) {
                if (*r == max_val)       H = (0.0f + (*g - *b) / delta) * 60.0;
                else if (*g == max_val)  H = (2.0f + (*b - *r) / delta) * 60.0;
                else                H = (4.0f + (*r - *g) / delta) * 60.0;


                if (H < 0.0f)   H += 360.0f;
            }
            H = fmod(H, 360.0f) / 360.0f;
            r_prime = round(H * 255.0);
            g_prime = round(S * 255.0);
            b_prime = round(V * 255.0);
            break;
        default:
            // Do nothing and encode RGB
            break;
        }
        *r = r_prime, *g = g_prime, *b = b_prime;
    }

    FIELD<float> red_channel   = im->r;
    FIELD<float> green_channel = im->g;
    FIELD<float> blue_channel  = im->b;
    red_channel.writePGM("R.pgm");
    green_channel.writePGM("G.pgm");
    blue_channel.writePGM("B.pgm");
    int num_layers_old;
    if (c_space == COLORSPACE::NONE)
        c_space = COLORSPACE::RGB;
    float r_min = get_min_elem(&(im->r));
    float g_min = get_min_elem(&(im->g));
    float b_min = get_min_elem(&(im->b));
    int Rfirst = 0;
    int Gfirst = 0;
    int Bfirst = 0;
    auto red_forest = execute_skeleton_pipeline(&red_channel, &Rfirst);
    auto green_forest = execute_skeleton_pipeline(&green_channel, &Gfirst);
    auto blue_forest = execute_skeleton_pipeline(&blue_channel, &Bfirst);
    PRINT(MSG_NORMAL, "Creating ImageWriter object...\n");
    ImageWriter iw(OUTPUT_FILE.c_str());
    PRINT(MSG_NORMAL, "Using colorspace %s\n", colorspace_to_string().c_str());
    iw.writeHeader(im->dimX(), im->dimY(), c_space, min(min(r_min, g_min), b_min), Rfirst, Gfirst, Bfirst);
    iw.write_color_image(red_forest, green_forest, blue_forest);
}

int main(int argc, char **argv) {
    /* Initialize GLUT */
    //glutInit(&argc, argv);

    

    //std::vector<std::string> realImages{ "bark.ppm","bikes.ppm","boat.pgm","graf.ppm","leuven.ppm","trees.ppm","ubc.ppm","wall.ppm" };
    //std::string fileName = "greyTest.pgm";
    //std::string fileNameWithoutExtension = fileName.substr(0, fileName.find_last_of("."));
    
      
    std::string filename = "E:\\Belangrijk\\School\\Master thesis\\datasets\\VGG\\boat.tar\\img1.pgm";

    auto img1 = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    skeletons skel(1.6, 0, 1.0f, 10000, "skel", true);
    std::vector<cv::KeyPoint> keyPoints;
    auto corners = skel.detectCorners(filename, { 5,5 },keyPoints);

    showImage(img1, corners, "", "detectedCorenrs", true, false, 200);
    

    /*
    for (const auto img : realImages) {
        auto imgWithout = img.substr(0, img.find_last_of("."));
        testFunction(readPGM("E:\\Belangrijk\\School\\Master thesis\\code\\imageConvertNew\\in\\" + img), false, imgWithout, 100, false);
    }*/
    //showCornersMethods();

    //testFunction(readPGM("E:\\Belangrijk\\School\\Master thesis\\code\\imageConvertNew\\in\\" + fileName),true, fileNameWithoutExtension,100, false);
    //testDescriptor();
        //readPGM("E:\\Belangrijk\\School\\Master thesis\\datasets\\VGG\\graf.tar\\img1.ppm"));
    //harrisCorner("in/greyTest2.pgm");
    /*testRepeatability("E:\\Belangrijk\\School\\Master thesis\\datasets\\VGG\\boat.tar\\img1.pgm",
        "E:\\Belangrijk\\School\\Master thesis\\datasets\\VGG\\boat.tar\\img2.pgm",
        "E:\\Belangrijk\\School\\Master thesis\\datasets\\VGG\\boat.tar\\H1to2p", true );*/
    //testDetectors();
    //testScale();
    
    /*
    std::string filename;
    std::getline(std::cin,filename);
    std::cout << filename << std::endl;
    testFunction(readPGM(filename), false, "boat2");//E:\Belangrijk\School\Master thesis\datasets\VGG\boat.tar\img1.pgm
                                                    //E:\Belangrijk\School\Master thesis\datasets\VGG\boat.tar\img2.pgm*/
    // TODO(maarten): Rely on smt better than file type detection or expand the list
    return 0;
}




