#include "include/Image.hpp"
#include <sys/stat.h>
#include <omp.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <set>

#include "include/skeleton_cuda.hpp"
#include "include/connected.hpp"
#include "include/messages.h"
//#include <parallel/algorithm> seems to be from a previous c++ version
#include <unordered_set>
#include <map>
#include <queue>
//#include <boost/functional/hash.hpp>

#include <cfenv>
#include <cerrno>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "include/utility.h"
#include "include/skeletonTracer.h"

#include "shared/CUDASkel2D/include/skelft.h"
#include "shared/CUDASkel2D/include/field.h"
#include "shared/CUDASkel2D/include/vis.h"
#include "shared/CUDASkel2D/include/skelcomp.h"

string OUTPUT_FILE;
bool BUNDLE, OVERLAP_PRUNE, MSIL_SELECTION;
float ALPHA;
float OVERLAP_PRUNE_EPSILON;
int EPSILON;
int MIN_PATH_LENGTH;
int MIN_SUM_RADIUS;
int MIN_OBJECT_SIZE;
int peaks;
int distinguishable_interval;
#define EPSILON 0.00001

/*************** CONSTRUCTORS ***************/
Image::Image(FIELD<float> *in, float islandThresh, float importanceThresh, int GrayInterval) {
    //PRINT(MSG_NORMAL, "Creating Image Object...\n");
    this->layerThreshold = importanceThresh;
    this->DistinguishableInterval = GrayInterval;
    this->islandThreshold = islandThresh;
    this->importance = NULL;
    this->im = in;
    this->nPix = in->dimX() * in->dimY();
    std::set<int> levels(in->data(), in->data() + this->nPix);
    this->graylevels = reinterpret_cast<int*>(malloc(levels.size() * sizeof(int)));
    std::set<int>::iterator it = levels.begin();
    for (unsigned int i = 0; i < levels.size(); ++i) {
        this->graylevels[i] = *it;
        std::advance(it, 1);
    }
    this->numLayers = levels.size();
    //PRINT(MSG_NORMAL, "Done!\n");
}

Image::Image(FIELD<float> *in) {
    Image(in, 0, 0, 0);
}

Image::~Image() {
    free(importance);
    free(graylevels);
}



/*************** FUNCTIONS **************/
void detect_peak(
    const double*   data, /* the data */
    int             data_count, /* row count of data */
    vector<int>&    emi_peaks, /* emission peaks will be put here */
    double          delta, /* delta used for distinguishing peaks */
    int             emi_first /* should we search emission peak first of
                                     absorption peak first? */
) {
    int     i;
    double  mx;
    double  mn;
    int     mx_pos = 0;
    int     mn_pos = 0;
    int     is_detecting_emi = emi_first;


    mx = data[0];
    mn = data[0];

    for (i = 1; i < data_count; ++i) {
        if (data[i] > mx) {
            mx_pos = i;
            mx = data[i];
        }
        if (data[i] < mn) {
            mn_pos = i;
            mn = data[i];
        }

        if (is_detecting_emi &&
                data[i] < mx - delta) {

            emi_peaks.push_back(mx_pos);
            is_detecting_emi = 0;

            i = mx_pos - 1;

            mn = data[mx_pos];
            mn_pos = mx_pos;
        } else if ((!is_detecting_emi) &&
                   data[i] > mn + delta) {

            is_detecting_emi = 1;

            i = mn_pos - 1;

            mx = data[mn_pos];
            mx_pos = mn_pos;
        }
    }
}

void find_peaks(double* importance, double width) {
    double impfac = 0.1;
    vector<int> v;
    int numiters = 0;
    while (numiters < 1000) {
        v.clear();
        detect_peak(importance, 256, v, impfac, 0);
        if (v.size() < width)
            impfac *= .9;
        else if (v.size() > width)
            impfac /= .9;
        else
            break;
        numiters++;
    }
    memset(importance, 0, 256 * sizeof(double));
    for (auto elem : v)
        importance[elem] = 1;
}

void detect_layers(double* upper_level, double threshold, bool needAssign)
{
    peaks = 0;
    int i = 0;
    int StartPoint = distinguishable_interval; //There is no need to check i and i+1, which is not distinguished by eyes, so check between i and i+StartPoint.
    double difference;

    double copy_upper_level[256];
    for (int j = 0; j < 256; ++j){
        copy_upper_level [j] = upper_level [j];
    }

    while ((i + StartPoint) < 256)
    {
        difference = copy_upper_level[i] - copy_upper_level[i + StartPoint];//attention: here shouldn't be upper_level
        if (difference > threshold)//choose this layer
        {
            if (needAssign) {
                upper_level[i + StartPoint] = 1;
            }
            
            i = i + StartPoint;
           // PRINT(MSG_NORMAL, "Choose_layers: %d\n",i);
            StartPoint = distinguishable_interval;
            peaks++;
        }
        else
        {
            StartPoint += 1;
        }
        
    }

}

//binary search
void find_layers(double* importance_upper, double width)
{
    double impfac = 0.5;
    double head = 0;
    double tail = 0.5;
    int numiters = 0;
    while (numiters < 50) {
       
        detect_layers(importance_upper, impfac, 0);
        //PRINT(MSG_NORMAL, "the number of peaks: %d\n",peaks);
        if (peaks < width){//impfac need a smaller one
            tail = impfac;
            impfac = (head + impfac)/2;
        }
            
        else if (peaks > width) //impfac need a bigger one
            {
                head = impfac;
                impfac = (tail + impfac)/2;
            }
            else
             break;
        numiters++;
    }
   // PRINT(MSG_NORMAL, "numiter:%d...\n", numiters);
  // PRINT(MSG_NORMAL, "numiter:%f...\n", impfac);
   
    detect_layers(importance_upper, impfac, 1);
    
     for (int i = 0; i < 256; ++i) {
        if (importance_upper[i] != 1) {
            importance_upper[i] = 0;
        }
    }
}

/*
* Calculate the histogram of the image, which is equal to the importance for each level.
* Avoid the use of in->value(), because it is less efficient (performs multiplications).
* The order is irrelevant anyway.
*/
void Image::calculateImportance() {
    PRINT(MSG_NORMAL, "Calculating the importance for each layer...\n");
    int normFac = 0;
    float *c = im->data();
    float *end = im->data() + nPix;
    /* If importance was already calculated before, cleanup! */
    if (importance) { free(importance); }
    importance = static_cast<double*>(calloc(256, sizeof(double)));
    double* UpperLevelSet = static_cast<double*>(calloc(256, sizeof(double)));

    if (!importance) {
        PRINT(MSG_ERROR, "Error: could not allocate importance histogram\n");
        exit(-1);
    }
    // Create a histogram
    while (c < end) {
        importance[static_cast<int>(*c++)] += 1;
    }

    ////////upper_level set histogram/////////
    UpperLevelSet[255] = importance[255];
    for (int i = 255; i > 0; i--)
    {
        UpperLevelSet[i-1] = importance[i-1] + UpperLevelSet[i];
    }

    // Normalize. Local-maxima method.
    normFac = static_cast<int>(*std::max_element(importance, importance + 256));//find the max one.
    for (int i = 0; i < 256; ++i) {
        importance[i] /= static_cast<double>(normFac);
    }
    // Normalize. Cumulative method.
    double max = UpperLevelSet[0];
    for (int i = 0; i < 256; ++i) {
        UpperLevelSet[i] = UpperLevelSet[i] / static_cast<double>(max) - EPSILON;//To avoid to be 1.
    }

    // cumulative histogram method
    if (MSIL_SELECTION) {
        distinguishable_interval = DistinguishableInterval;
        find_layers(UpperLevelSet, layerThreshold);
        //memcpy(importance, UpperLevelSet, sizeof(UpperLevelSet));
        importance = UpperLevelSet;
    } 
    else {
        // else local-maxima method
        find_peaks(importance, layerThreshold);
    }
    
    
    std::vector<int> v;
    for (int i = 0; i < 256; ++i) {//CHANGES THIS change1
        if (importance[i] == 1) {
            v.push_back(i);
        }
    }
    if (v.size() == 0) {
        PRINT(MSG_ERROR, "ERROR: No layers selected. Exiting...\n");
        exit(-1);
    }
    PRINT(MSG_NORMAL, "Selected %lu layers: ", v.size());
    std::ostringstream ss;

    std::copy(v.begin(), v.end() - 1, std::ostream_iterator<int>(ss, ", "));
    ss << v.back();
    PRINT(MSG_NORMAL, "%s\n", ss.str().c_str());
    v.clear();
}

/**
* fullDilate and fullErode are placeholders. Although they do work, they
* should be replaced by better erode and dilation functions. This is used primarily
* to test how much an opening on the skeleton will reduce the image size.
*/

/* fullDilate -- Perform dilation with a S.E. of 3x3, filled with ones. */
FIELD<float> * fullDilate(FIELD<float> *layer) {
    FIELD<float> *ret = new FIELD<float>(layer->dimX(), layer->dimY());
    memset(ret->data(), 0, layer->dimX() * layer->dimY() * sizeof(float));
    for (int y = 0; y < layer->dimY(); ++y) {
        for (int x = 0; x < layer->dimX(); ++x) {
            if (layer->value(x, y)) {
                ret->set(x - 1, y - 1, 255);
                ret->set(x - 1, y  , 255);
                ret->set(x - 1, y + 1, 255);
                ret->set(x  , y - 1, 255);
                ret->set(x  , y  , 255);
                ret->set(x  , y + 1, 255);
                ret->set(x + 1, y - 1, 255);
                ret->set(x + 1, y  , 255);
                ret->set(x + 1, y + 1, 255);
            }
        }
    }
    delete layer;
    return ret;
}

/* fullErode -- Perform erosion with a S.E. of 3x3, filled with ones. */
FIELD<float> * fullErode(FIELD<float> *layer) {
    FIELD<float> *ret = new FIELD<float>(layer->dimX(), layer->dimY());
    for (int y = 0; y < layer->dimY(); ++y) {
        for (int x = 0; x < layer->dimX(); ++x) {
            if (
                layer->value(x - 1, y - 1) &&
                layer->value(x - 1, y) &&
                layer->value(x - 1, y + 1) &&
                layer->value(x, y - 1) &&
                layer->value(x, y) &&
                layer->value(x, y + 1) &&
                layer->value(x + 1, y - 1) &&
                layer->value(x + 1, y) &&
                layer->value(x + 1, y + 1)
            ) {
                ret->set(x, y, 255);
            } else { ret->set(x, y, 0); }
        }
    }
    delete layer;
    return ret;
}

/* rmObject -Remove current object in a 3x3 kernel, used for removeDoubleSkel: */
void rmObject(int *k, int x, int y) {
    if (x < 0 || x > 2 || y < 0 || y > 2 || k[y * 3 + x] == 0) { return; }
    k[y * 3 + x] = 0;
    rmObject(k, x + 1, y + 1);
    rmObject(k, x + 1, y);
    rmObject(k, x + 1, y - 1);
    rmObject(k, x, y + 1);
    rmObject(k, x, y - 1);
    rmObject(k, x - 1, y + 1);
    rmObject(k, x - 1, y);
    rmObject(k, x - 1, y - 1);
}
/* numObjects - Count the number of objects in a 3x3 kernel, used for removeDoubleSkel: */
int numObjects(int *k) {
    int c = 0;
    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; ++y) {
            if (k[y * 3 + x]) { rmObject(k, x, y); c++; }
        }
    }
    return c;
}
/* End count code */

/**
* removeDoubleSkel
* @param FIELD<float> * layer -- the layer of which the skeleton should be reduced
* @return new FIELD<float> *. Copy of 'layer', where all redundant skeleton-points are removed (i.e. rows of width 2.)
*/
FIELD<float> * removeDoubleSkel(FIELD<float> *layer) {
    int *k = (int *)calloc(9, sizeof(int));
    for (int y = 0; y < layer->dimY(); ++y) {
        for (int x = 0; x < layer->dimX(); ++x) {
            if (layer->value(x, y)) {
                k[0] = layer->value(x - 1, y - 1);
                k[1] = layer->value(x - 1, y);
                k[2] = layer->value(x - 1, y + 1);
                k[3] = layer->value(x  , y - 1);
                k[4] = 0;
                k[5] = layer->value(x  , y + 1);
                k[6] = layer->value(x + 1, y - 1);
                k[7] = layer->value(x + 1, y);
                k[8] = layer->value(x + 1, y + 1);
                if (k[0] + k[1] + k[2] + k[3] + k[4] + k[5] + k[6] + k[7] + k[8] > 256) {
                    int b = numObjects(k);
                    if (b < 2 ) {layer->set(x, y, 0); }
                }
            }
        }
    }
    free(k);
    return layer;
}


/**
* Given a binary layer, remove all islands smaller than iThresh.
* , where k is the current intensity.
* @param layer
* @param iThresh
*/
void Image::removeIslands(FIELD<float>*layer, unsigned int iThresh) {
    int                     nPix    = layer->dimX() * layer->dimY();
    ConnectedComponents     *CC     = new ConnectedComponents(255);
    int                     *labeling = new int[nPix];
    float                   *fdata  = layer->data();
    int                     highestLabel;
    unsigned int            *num_pix_per_label;

    /* CCA -- store highest label in 'max' -- Calculate histogram */
    highestLabel = CC->connected(fdata, labeling, layer->dimX(), layer->dimY(), std::equal_to<float>(), true);
    // Count the number of pixels for each label
    num_pix_per_label = (unsigned int*) calloc(highestLabel + 1, sizeof(unsigned int));
    for (int j = 0; j < nPix; j++) { num_pix_per_label[labeling[j]]++; }

    /* Remove small islands based on a percentage of the image size*/
    for (int j = 0; j < nPix; j++) {
        fdata[j] = (num_pix_per_label[labeling[j]] >= (iThresh / 100.0) * nPix) ? fdata[j] : 255 - fdata[j];
    }

    /* Cleanup */
    free(num_pix_per_label);
    delete [] labeling;
    delete CC;
}


/*
* Remove small islands according the the islandThreshold variable. Notice that both "on" and "off"
* regions will be filtered.
*/
void Image::removeIslands() {
    int i, j, k;                    /* Iterators */
    FIELD<float> *inDuplicate = 0;  /* Duplicate, because of inplace modifications */
    FIELD<float> *newImg = new FIELD<float>(im->dimX(), im->dimY());
    int highestLabel;               /* for the CCA */
    int *ccaOut;                    /* labeled output */
    ConnectedComponents *CC;        /* CCA-object */
    float *fdata;
    unsigned int *hist;

    //PRINT(MSG_NORMAL, "Removing small islands...\n");
    /* Some basic initialization */
    memset(newImg->data(), 0, nPix * sizeof(float));

    /* Connected Component Analysis */
    #pragma omp parallel for private(i, j, k, ccaOut, CC, fdata, highestLabel, hist, inDuplicate)

    
    for (i = 0; i < 0xff; ++i) {
        PRINT(MSG_VERBOSE, "Layer: %d\n", i);
        // The below value refers to the expected number of components in an image.
        CC = new ConnectedComponents(255);
        ccaOut = new int[nPix];

        inDuplicate = (*im).dupe();
        inDuplicate->threshold(i);//threshold-set..
        
        //if (i==53)
         //   inDuplicate->writePGM("gray53.pgm"); //debug
        
        fdata = inDuplicate->data();

        /* CCA -- store highest label in 'max' -- Calculate histogram */
        highestLabel = CC->connected(fdata, ccaOut, im->dimX(), im->dimY(), std::equal_to<float>(), true);
        hist = static_cast<unsigned int*>(calloc(highestLabel + 1, sizeof(unsigned int)));
        if (!hist) {
            PRINT(MSG_ERROR, "Error: Could not allocate histogram for connected components\n");
            exit(-1);
        }
        for (j = 0; j < nPix; j++) { hist[ccaOut[j]]++; }

        /* Remove small islands */
       
        for (j = 0; j < nPix; j++) {
           fdata[j] = (hist[ccaOut[j]] >= (islandThreshold/100*im->dimX()*im->dimY())) ? fdata[j] : 255 - fdata[j]; //change the absolute num. to %
             //fdata[j] = (hist[ccaOut[j]] >= islandThreshold) ? fdata[j] : 255 - fdata[j]; //original one.
         
        }
        /* for debug
        if (i==53)
        {
            int xM = im->dimX();
            int yM = im->dimY();
            FIELD<float>* f = new FIELD<float>(xM, yM);
            
            for (int i = 0; i < xM; ++i) {
                for (int j = 0; j < yM; ++j) {
                    
                    f->set(i, j, fdata[j * im->dimX() + i]);
                }
            }
            f->writePGM("afterIsland_53.pgm");
        }
        */
        #pragma omp critical
        {
            for (j = 0; j < im->dimY(); j++)
                for (k = 0; k < im->dimX(); k++)
                    if (0 == fdata[j * im->dimX() + k] && newImg->value(k, j) < i) { newImg->set(k, j, i); }
        }

        
        /* Cleanup */
        free(hist);
        delete [] ccaOut;
        delete CC;
        delete inDuplicate;
    }
    for (j = 0; j < im->dimY(); j++)
        for (k = 0; k < im->dimX(); k++)
            im->set(k, j, newImg->value(k, j));

    delete newImg;
    //PRINT(MSG_NORMAL, "Done!\n");
}

/**
* Remove unimportant layers -- Filter all layers for which their importance is lower than layerThreshold
*/
void Image::removeLayers() {
    float val_up, val_dn;

    PRINT(MSG_NORMAL, "Filtering image layers...\n");
    PRINT(MSG_VERBOSE, "The following grayscale intensities are removed:\n");
    if (MSG_LEVEL == MSG_VERBOSE)
        for (int i = 0; i < 256; ++i) {
            if (importance[i] < 1) {
                PRINT(MSG_VERBOSE, "(%d, %6.5f)\n", i, importance[i]);
            }
        }

    for (int y = 0; y < im->dimY(); y++) {
        for (int x = 0; x < im->dimX(); x++) {
            val_up = im->value(x, y);
            val_dn = im->value(x, y);
            if (importance[(int)im->value(x, y)] == 1)
                continue;
            while (val_dn >= 0 || val_up <= 256) {
                if (val_dn >= 0) {
                    if (importance[(int)val_dn] == 1) {
                        im->set(x, y, val_dn);
                        break;
                    }
                }
                if (val_up <= 256) {
                    if (importance[(int)val_up] == 1) {
                        im->set(x, y, val_up);
                        break;
                    }
                }
                val_dn--;
                val_up++;
                
            }
        }
    }
    
}

coord2D_list_t* neighbours(int x, int y, FIELD<float>* skel) {
    coord2D_list_t* neigh = new coord2D_list_t();
    int n[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

    /* Check if we are hitting a boundary on the image */
    if (x <= 0) { n[0] = 0;        n[3] = 0;        n[5] = 0; }
    if (x >= skel->dimX() - 1) { n[2] = 0;        n[4] = 0;        n[7] = 0; }
    if (y <= 0) { n[0] = 0;        n[1] = 0;        n[2] = 0; }
    if (y >= skel->dimY() - 1) { n[5] = 0;        n[6] = 0;        n[7] = 0; }

    /* For all valid coordinates in the 3x3 region: check for neighbours*/
    if ((n[0] != 0) && (skel->value(x - 1, y - 1) > 0)) { neigh->push_back(coord2D_t(x - 1, y - 1)); }
    if ((n[1] != 0) && (skel->value(x, y - 1) > 0)) { neigh->push_back(coord2D_t(x, y - 1)); }
    if ((n[2] != 0) && (skel->value(x + 1, y - 1) > 0)) { neigh->push_back(coord2D_t(x + 1, y - 1)); }
    if ((n[3] != 0) && (skel->value(x - 1, y) > 0)) { neigh->push_back(coord2D_t(x - 1, y)); }
    if ((n[4] != 0) && (skel->value(x + 1, y) > 0)) { neigh->push_back(coord2D_t(x + 1, y)); }
    if ((n[5] != 0) && (skel->value(x - 1, y + 1) > 0)) { neigh->push_back(coord2D_t(x - 1, y + 1)); }
    if ((n[6] != 0) && (skel->value(x, y + 1) > 0)) { neigh->push_back(coord2D_t(x, y + 1)); }
    if ((n[7] != 0) && (skel->value(x + 1, y + 1) > 0)) { neigh->push_back(coord2D_t(x + 1, y + 1)); }

    return neigh;
}

void neighboursSet(int x, int y, FIELD<float>* skel, std::set<coord2D_t>& neighbours) {
    int n[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

    /* Check if we are hitting a boundary on the image */
    if (x <= 0) { n[0] = 0;        n[3] = 0;        n[5] = 0; }
    if (x >= skel->dimX() - 1) { n[2] = 0;        n[4] = 0;        n[7] = 0; }
    if (y <= 0) { n[0] = 0;        n[1] = 0;        n[2] = 0; }
    if (y >= skel->dimY() - 1) { n[5] = 0;        n[6] = 0;        n[7] = 0; }

    /* For all valid coordinates in the 3x3 region: check for neighbours*/
    if ((n[0] != 0) && (skel->value(x - 1, y - 1) > 0)) {   neighbours.insert(coord2D_t(x - 1, y - 1));}// skel->value(x - 1, y - 1) = 0; }
    if ((n[1] != 0) && (skel->value(x, y - 1) > 0)) {       neighbours.insert(coord2D_t(x, y - 1)); }//skel->value(x, y - 1) = 0;}
    if ((n[2] != 0) && (skel->value(x + 1, y - 1) > 0)) {   neighbours.insert(coord2D_t(x + 1, y - 1)); }// skel->value(x + 1, y - 1) = 0; }
    if ((n[3] != 0) && (skel->value(x - 1, y) > 0)) {       neighbours.insert(coord2D_t(x - 1, y)); }//skel->value(x - 1, y) = 0;  }
    if ((n[4] != 0) && (skel->value(x + 1, y) > 0)) {       neighbours.insert(coord2D_t(x + 1, y)); }//skel->value(x + 1, y) = 0;  }
    if ((n[5] != 0) && (skel->value(x - 1, y + 1) > 0)) {   neighbours.insert(coord2D_t(x - 1, y + 1));}// skel->value(x - 1, y + 1) = 0;  }
    if ((n[6] != 0) && (skel->value(x, y + 1) > 0)) {       neighbours.insert(coord2D_t(x, y + 1));}// skel->value(x, y + 1) = 0;   }
    if ((n[7] != 0) && (skel->value(x + 1, y + 1) > 0)) {   neighbours.insert(coord2D_t(x + 1, y + 1));}// skel->value(x + 1, y + 1) = 0;  }
}

//gives the coords of all neighbors that are within the boundary, also non skeleton points
coord2D_list_t nonSkeletonNeighbours(int x, int y, FIELD<float> *skel) {
    coord2D_list_t neigh;
    int n[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    /* Check if we are hitting a boundary on the image */
    if (x <= 0 )             {        n[0] = 0;        n[3] = 0;        n[5] = 0;    }
    if (x >= skel->dimX() - 1) {        n[2] = 0;        n[4] = 0;        n[7] = 0;    }
    if (y <= 0)              {        n[0] = 0;        n[1] = 0;        n[2] = 0;    }
    if (y >= skel->dimY() - 1) {        n[5] = 0;        n[6] = 0;        n[7] = 0;    }

    /* For all valid coordinates in the 3x3 region: check for neighbours*/
    if ((n[0] != 0)) { neigh.push_back(coord2D_t(x - 1, y - 1)); }
    if ((n[1] != 0)) { neigh.push_back(coord2D_t(x    , y - 1)); }
    if ((n[2] != 0)) { neigh.push_back(coord2D_t(x + 1, y - 1)); }
    if ((n[3] != 0)) { neigh.push_back(coord2D_t(x - 1, y    )); }
    if ((n[4] != 0)) { neigh.push_back(coord2D_t(x + 1, y    )); }
    if ((n[5] != 0)) { neigh.push_back(coord2D_t(x - 1, y + 1)); }
    if ((n[6] != 0)) { neigh.push_back(coord2D_t(x    , y + 1)); }
    if ((n[7] != 0)) { neigh.push_back(coord2D_t(x + 1, y + 1)); }

    return neigh;
}

//gives the coords of all neighbors that are within the boundary, also non skeleton points
coord2D_list_t boundaryNeighbours(int x, int y, FIELD<float>* skel, int index, SkelEngine* skelE) {
    coord2D_list_t neigh;
    int n[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

    /* Check if we are hitting a boundary on the image */
    if (x <= 0) { n[0] = 0;        n[3] = 0;        n[5] = 0; }
    if (x >= skel->dimX() - 1) { n[2] = 0;        n[4] = 0;        n[7] = 0; }
    if (y <= 0) { n[0] = 0;        n[1] = 0;        n[2] = 0; }
    if (y >= skel->dimY() - 1) { n[5] = 0;        n[6] = 0;        n[7] = 0; }

    /* For all valid coordinates in the 3x3 region: check for neighbours*/
    if ((n[0] != 0) && skelE->getSiteLabelReplace({ x - 1, y - 1 }, index) != index) { neigh.push_back(coord2D_t(x - 1, y - 1)); }
    if ((n[1] != 0) && skelE->getSiteLabelReplace({ x    , y - 1 }, index) != index) { neigh.push_back(coord2D_t(x, y - 1)); }
    if ((n[2] != 0) && skelE->getSiteLabelReplace({ x + 1, y - 1 }, index) != index) { neigh.push_back(coord2D_t(x + 1, y - 1)); }
    if ((n[3] != 0) && skelE->getSiteLabelReplace({ x - 1, y }, index) != index) { neigh.push_back(coord2D_t(x - 1, y)); }
    if ((n[4] != 0) && skelE->getSiteLabelReplace({ x + 1, y }, index) != index) { neigh.push_back(coord2D_t(x + 1, y)); }
    if ((n[5] != 0) && skelE->getSiteLabelReplace({ x - 1, y + 1 }, index) != index) { neigh.push_back(coord2D_t(x - 1, y + 1)); }
    if ((n[6] != 0) && skelE->getSiteLabelReplace({ x    , y + 1 }, index) != index) { neigh.push_back(coord2D_t(x, y + 1)); }
    if ((n[7] != 0) && skelE->getSiteLabelReplace({ x + 1, y + 1 }, index) != index) { neigh.push_back(coord2D_t(x + 1, y + 1)); }

    return neigh;
}

double SafeAcos(double x)
{
    if (x <= -1.0) x = -1.0;
    else if (x >= 1.0) x = 1.0;
    return acos(x);
}

//calculates the angle between three points
float getAngleABC(coord2D_t &a, coord2D_t &b, coord2D_t &c)
{   /*
    coord2D_t ab = { b.first - a.first, b.second - a.second };
    coord2D_t cb = { b.first - c.first, b.second - c.second };

    float dot = (ab.first * cb.first + ab.second * cb.second); // dot product
    float cross = (ab.first * cb.second - ab.second * cb.first); // cross product

    float alpha = atan2(cross, dot);
    //std::cout << abs(alpha) << std::endl;
    acos()
    return alpha * 180. / CV_PI + 0.5;//abs(alpha);//*/
    coord2D_t ab = { b.first - a.first, b.second - a.second };
    coord2D_t cb = { b.first - c.first, b.second - c.second };

    float dot = (ab.first * cb.first + ab.second * cb.second); // dot product
    float lengthab  = sqrt(ab.first * ab.first + ab.second * ab.second); // cross product
    float lengthbc = sqrt(cb.first * cb.first + cb.second * cb.second);

    float alpha = dot / (lengthab * lengthbc + EPSILON);
    errno = 0;
    std::feclearexcept(FE_ALL_EXCEPT);
    float angle = SafeAcos(alpha);
    //std::cout << angle << "|  " << lengthab << " | " << lengthbc << " |  "<< dot << " | " << alpha << std::endl;
    
    if (errno == EDOM)
        std::cout << "    errno == EDOM: " << std::strerror(errno) << '\n';
    if (std::fetestexcept(FE_INVALID))
        std::cout << "    FE_INVALID raised" << '\n';
    return angle;//abs(alpha);
}

struct nodeInfo {
    coord2D_t thisNode;
    coord2D_t lastNode;
    int distance;//distance to the first node in this chain
    int indexLastNode;//index of last node
    int thisIndex;
};

std::pair<coord2D_t,float> addSkeletonPoint(int x, int y, FIELD<float>* skel, FIELD<float>* dt, FIELD<float>& skeletonEndpoints, int pointIndex, SkelEngine* skelE) {
    coord2D_t f1 = skelE->getFT(x, y);//gets the coordinate of f1
    bool show = 0;
    auto newNeighbours = nonSkeletonNeighbours(x, y, skel);//gets the coords of its 8 neighbors

    float maxAngle = 0;
    coord2D_t f2(-1, -1);

    for (int i = 0; i < newNeighbours.size(); i++) {
        coord2D_t potentialf2 = skelE->getFT(newNeighbours[i].first, newNeighbours[i].second);//the potential second feature transform point
        float angle = getAngleABC(f1, coord2D_t(x, y), potentialf2);//calculate teh angle between f1,original point,f2
        if(show)
            std::cout << potentialf2.first << " " << potentialf2.second << " " << angle << std::endl;
        if (angle >= maxAngle) {
            f2 = potentialf2;
            maxAngle = angle;
        }
    }
    if (f2 == coord2D_t(-1, -1)) {
        std::cout << "error was not able to find the estimate feature point f2" << std::endl;
        exit(-1);
    }

    if (show)std::cout << x << " " << y << " | " << f1.first << " " << f1.second << " | " << f2.first << " " << f2.second << " | " << std::endl;

    //so now we have f1 and hopefully an approximation of f2, now need to travel on the boundary to the point between f1 and f2
    float f1Label = skelE->getSiteLabel(f1);
    float f2Label = skelE->getSiteLabel(f2);
    if (show)std::cout << "labels: " << f1Label << " " << f2Label << std::endl;
    //do a bfs to find f2 from f1.

    std::vector<nodeInfo> visitedPairsImproved;
    //std::set<coord2D_t> visitedPairs;
    std::queue<std::pair<coord2D_t,int>> unvisitedNodes;
    int index = 0;

    //get the first neighbours
    unvisitedNodes.push({ f1,index });
    //visitedPairs.insert(f1);

    visitedPairsImproved.push_back({ coord2D_t{f1.first,f1.second},coord2D_t{-1,-1},0,0,index});
    ++index;

    bool found;
    (f1 == f2) ? found = true : found = false;
    
    
    while (!unvisitedNodes.empty() && !found) {
        auto currentNode = unvisitedNodes.front();
        unvisitedNodes.pop();
        auto currentNeighbours = boundaryNeighbours(currentNode.first.first, currentNode.first.second, skel, pointIndex,skelE);
        
        for (int i = 0; i < currentNeighbours.size(); i++) {//add new nodes to queue
             if (currentNeighbours[i] == f2) {//found it
                    found = true;
                    int distance = visitedPairsImproved[currentNode.second].distance;
                    auto newNode = nodeInfo{ currentNeighbours[i], visitedPairsImproved[currentNode.second].thisNode,distance + 1,visitedPairsImproved[currentNode.second].thisIndex,index};
                    if (show)std::cout << currentNeighbours[i].first << " " << currentNeighbours[i].second << " " <<
                         currentNode.first.first << " " << currentNode.first.second << " " << distance << std::endl;
                    //std::cout << newNode.thisIndex << " " << newNode.indexLastNode << " " << index << "\n";
                    visitedPairsImproved.push_back(newNode);//mark it as visited, and save currentNode as its parent
                }
                else{//not there yet, keep going
                    unvisitedNodes.push({ currentNeighbours[i],index });//add this neighbour to the queue
                    int distance = visitedPairsImproved[currentNode.second].distance;
                    auto newNode = nodeInfo{ currentNeighbours[i], visitedPairsImproved[currentNode.second].thisNode,distance + 1,visitedPairsImproved[currentNode.second].thisIndex,index};
                    visitedPairsImproved.push_back(newNode);//mark it as visited, and save currentNode as its parent
                    
                    //if (show)std::cout << currentNeighbours[i].first << " " << currentNeighbours[i].second << " " <<
                        //currentNode.first.first << " " << currentNode.first.second << " " << distance << std::endl;
                    ++index;
                }

        }
    }
    //cv::waitKey();
    auto lastNode = visitedPairsImproved[visitedPairsImproved.size()-1];//the last node added is f2
    float adjustedImportance = lastNode.distance;
    coord2D_t finalCoords;
    if (1) {
        int distance = lastNode.distance;
        int middleDistance = distance / 2;//the node we need to find

        while (distance != middleDistance) {
            int lastIndex = lastNode.indexLastNode;
            if (show)std::cout << distance << "| " << middleDistance << " " << lastIndex << std::endl;
            lastNode = visitedPairsImproved[lastIndex];
            distance = lastNode.distance;
        }

        finalCoords = lastNode.thisNode;
    }
    else {
        std::cout << "WARNING PROBLEMS WITH FINDING CORNER POINT" << std::endl;
        finalCoords = coord2D_t(x,y);
        return { finalCoords,adjustedImportance };
    }

    if (show) std::cout << "finalcoords: " << finalCoords.first << " " << finalCoords.second << "\n\n";
    //skeletonEndpoints.value(finalCoords.first, finalCoords.second)++;
    //skeletonEndpoints.value(x, y)++;
    return { finalCoords,adjustedImportance };
}

void bundle(FIELD<float>* skelCurr, FIELD<float>* currDT, FIELD<float>* prevDT, short* prev_skel_ft, int fboSize) {
    ALPHA = max(0.0f, min(1.0f, ALPHA));
    for (int i = 0; i < skelCurr->dimX(); ++i) {
        for (int j = 0; j < skelCurr->dimY(); ++j) {
            if (skelCurr->value(i, j) > 0) {
                // We might bundle this point
                int id = j * fboSize + i;

                // Find closest point
                int closest_x = prev_skel_ft[2 * id];
                int closest_y = prev_skel_ft[2 * id + 1];
                if (closest_x == -1 && closest_y == 0)
                    continue;
                // If it is already on a previous skeleton point, were done
                double distance = sqrt((i - closest_x) * (i - closest_x) + (j - closest_y) * (j - closest_y));
                if (distance == 0.0)
                    continue;

                // Create vector from the current point to the closest prev skel point
                double xvec = (closest_x - i);
                double yvec = (closest_y - j);

                // What would be the location with no bound on the shift?
                int trans_vec_x = ALPHA * xvec;
                int trans_vec_y = ALPHA * yvec;
                double trans_vec_len = sqrt(trans_vec_x * trans_vec_x + trans_vec_y * trans_vec_y);

                // If we don't move it at all were done
                if (trans_vec_len == 0.0)
                    continue;

                // Find the limited length of this vector
                double desired_length = min(trans_vec_len, EPSILON);
                // Rescale the vector such that it is bounded by epsilon
                int new_x_coor = round(i + (trans_vec_x / trans_vec_len) * desired_length);
                int new_y_coor = round(j + (trans_vec_y / trans_vec_len) * desired_length);
                // Verify that the new distance is at most near EPSILON (i.e. FLOOR(new_distance) <= EPSILON)
                double new_distance = sqrt((i - new_x_coor) * (i - new_x_coor) + (j - new_y_coor) * (j - new_y_coor));
                PRINT(MSG_VERBOSE, "%d: (%d, %d) -> (%d, %d) (%lf) -> (%d, %d) (%lf)\n", EPSILON, i, j, closest_x, closest_y, distance, new_x_coor, new_y_coor, new_distance);

                // compute a new radius
                double new_r = min(currDT->value(i, j), prevDT->value(new_x_coor, new_y_coor));
                // double new_r = currDT->value(i, j);
                skelCurr->set(i, j, 0);
                skelCurr->set(new_x_coor, new_y_coor, 255);
                currDT->set(new_x_coor, new_y_coor, new_r);
            }
        }
    }
}

double overlap_prune(FIELD<float>* skelCurr, FIELD<float>* skelPrev, FIELD<float>* currDT, FIELD<float>* prevDT) {
    // Bigger at same location
    int count = 0, total_skel_points = 0;
    for (int i = 0; i < skelPrev->dimX(); ++i) {
        for (int j = 0; j < skelPrev->dimY(); ++j) {
            // If there is a skeleton point...
            if (skelCurr->value(i, j) > 0 && skelPrev->value(i, j) > 0) {
                // cout << skelCurr->value(i, j) << ", " << skelPrev->value(i, j) << endl;
                total_skel_points++;
                // ...and the radius difference w.r.t. the next layer is small enough...
                if ((currDT->value(i, j)) >= (prevDT->value(i, j))) {
                    // ...we can delete the skeleton point
                    skelPrev->set(i, j, 0);
                    count++;
                }
            }
        }
    }
// Return overlap prune ratio for statistics
    if (total_skel_points == 0)
        return 0;
    double overlap_prune_ratio = static_cast<double>(count) / total_skel_points * 100.0;
    (MSG_VERBOSE, "Overlap pruned %d points out %d points (%.2f%%)\n", count, total_skel_points, overlap_prune_ratio);

    return overlap_prune_ratio;
}

SkelEngine* perform_skeletonization(int level, FIELD<float>* upper_level_set, bool doReverse, float saliency, bool scan_bot_to_top) {
    int nx = upper_level_set->dimX();
    int ny = upper_level_set->dimY();
    short fboSize = skelft2DSize(nx, ny);

    unsigned char* image = new unsigned char[fboSize * fboSize];
    memset(image, 0, fboSize * fboSize * sizeof(unsigned char));

    short xm = nx, ym = nx, xM = 0, yM = 0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            if (!(*upper_level_set)(i, j))
            {
                image[j * fboSize + i] = 255;
                xm = min(xm, i); ym = min(ym, j);
                xM = max(xM, i); yM = max(yM, j);
            }

    xM = nx;
    yM = ny;
    xm = 0;
    ym = 0;

    SkelEngine* skel = new SkelEngine();

    skel->initialize(image, fboSize, xm, ym, xM, yM, scan_bot_to_top);
    float threshold = saliency;

    skel->compute(threshold);

    delete[] image;
   
    return skel;
}

void Image::collectEndpoints(int *firstLayer, FIELD<std::vector<cornerData>>& skeletonEndpoints, FIELD<float>& skeletonEndpoints2, float saliency, float stepSize) {
    for (int i = 1; i < 256; i = i + stepSize) {
        auto imDupeCurr = im->dupe();
        imDupeCurr->threshold(i);

        auto t1 = std::chrono::high_resolution_clock::now();

        //Compute skeleton with labeling from bottom to top
        auto skelE2 = perform_skeletonization(i, imDupeCurr, false, saliency, true);
        auto skel2 = skelE2->getSkelField();
        auto imp2 = skelE2->getImpField();
        delete skelE2;

        //Compute skeleton with labeling from top to bottom
        auto skelE = perform_skeletonization(i, imDupeCurr, false, saliency, false);
        auto skel1 = skelE->getSkelField();
        auto imp1 = skelE->getImpField();
        delete skelE;

        //Combine the two skeletons. Using cv::Mat does not copy the data.
        cv::Mat test1 = cv::Mat(skel1->dimY(), skel1->dimX(), CV_32FC1, skel1->data());
        cv::Mat test2 = cv::Mat(skel2->dimY(), skel2->dimX(), CV_32FC1, skel2->data());
        cv::bitwise_and(test1, test2, test1);

        //The importance is also combined, because in both images there are pixels with a wrong estimated importance.
        test1 = cv::Mat(imp1->dimY(), imp1->dimX(), CV_32FC1, imp1->data());
        test2 = cv::Mat(imp2->dimY(), imp2->dimX(), CV_32FC1, imp2->data());
        test1 = cv::min(test1, test2);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "calculated skeleton in time: " << duration << "| ";

        auto imThreshold = im->dupe();
        imThreshold->thresholdInv(i);
        
        t1 = std::chrono::high_resolution_clock::now();
        
        //Returns for each connected component find a skeleton point belonging to the main skeleton
        std::vector<coord2D_t> startPoints = findLargestComponent(imThreshold, imp1, skel1);

        auto tracerobject = tracer(skelE, skel1, imp1, imDupeCurr, skeletonEndpoints, skeletonEndpoints2, startPoints, true);
        auto endPoints = tracerobject.traceSkeleton();

        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "trace skeleton in time : " << duration << std::endl;

        delete imThreshold;
        delete skel1;
        delete skel2;
        delete imDupeCurr;
        delete imp1;
        delete imp2;
    }
}

