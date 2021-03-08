#ifndef IMAGE_GUARD
#define IMAGE_GUARD

#include "../shared/CUDASkel2D/include/field.h"
#include "shared/CUDASkel2D/include/skelcomp.h"
#include "ImageWriter.hpp"
#include <set>
#include <vector>

struct featureVector {
    std::pair<float, string> avgX{ 0,"avgX" }; //standard deviation of the coordinates, stddev x + stddev y / 2
    std::pair<float, string> avgY{ 0,"avgY" };

    float nEndpoints;
    float response;

    std::pair<int, string> label{ 0,"label" };//0 = not repeated. 1= repeated. 2 = not in the other image, thus irrelevant
    std::pair<int, string> closestIndexDistance{ 0,"closestIndexDistance" };//distance in terms of index to its repeated corner in the second image.

    std::vector<string> featureNames = { "stddX", "stddY" ,"avgSaliency","stddSaliency" ,"avgNormalizedDt" ,"stddNnormalizedDt" ,"avgObjectSizeNorm","stddObjectSizeNorm"
        ,"avgEndpointsObject" ,"stddEndpointsObject" ,"nEndpoints" };
    std::vector<float> features = std::vector<float>(11);

    void printHeader(ostream& output);
};

struct cornerData {
    int x;
    int y;
    float saliency;
    float normalizedDt;//dt divided by branch length
    float objectSizeNorm;//branch length divided by objectSize 
    float nEndpointsObject;
    float distance;
    //cornerData(nx,ny,nsaliency)
};

struct totalCornerData {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> saliency;
    std::vector<float> normalizedDt;//dt divided by branch length
    std::vector<float> objectSizeNorm;//branch length divided by objectSize 
    std::vector<float> nEndpointsObject;
    void addCornerData(const cornerData&& newCorner) {
        x.push_back(newCorner.x); 
        y.push_back(newCorner.y); 
        saliency.push_back(newCorner.saliency);
        normalizedDt.push_back(newCorner.normalizedDt);
        objectSizeNorm.push_back(newCorner.objectSizeNorm);
        nEndpointsObject.push_back(newCorner.nEndpointsObject);
    }
};

class Image {
  private:
    /** VARIABLES **/
    float      islandThreshold;
    float             layerThreshold;
    int               DistinguishableInterval;
    double            *importance;
    //double            *UpperLevelSet;
    int               numLayers;
    int               nPix; /* Short for (dimX * dimY) */
    string            compress_method;
    int*  graylevels = nullptr;
    skel_tree_t *traceLayer(FIELD<float> *skel, FIELD<float> *dt);
    skel_tree_t *tracePath(int x, int y, FIELD<float> *skel, FIELD<float> *dt);
    //coord2D_list_t *neighbours(int x, int y, FIELD<float> *skel);//wang

  public:
    /** VARIABLES **/
    FIELD<float>   *im;
    /** CONSTRUCTORS **/
    Image(FIELD<float> *in);
    Image(FIELD<float> *in, float islandThresh, float importanceThresh, int GrayInterval);

    /** DESTRUCTOR **/
    ~Image();

    /** FUNCTIONS **/
    static void removeIslands(FIELD<float>*layer, unsigned int iThresh);
    void removeIslands();
    void removeLayers();
    void calculateImportance();
    void collectEndpoints(int *firstLayer, FIELD<std::vector<cornerData>>& skeletonEndpoints, FIELD<float>& skeletonEndpoints2, float saliency, float stepSize);
    void computeCUDASkeletons();
    void removeDuplicatePoints(FIELD<float> *imPrev, FIELD<float> *skP, FIELD<float> *imCur, FIELD<float> *skC);
    pair<int, int> find_closest_point(int i, int j, FIELD<float>* skelPrev);
    void bundle(FIELD<float>* skelCurr, FIELD<float>* skelPrev, FIELD<float>* currDT, FIELD<float>* prevDT, short* prev_skel_ft, int fboSize);
    double overlap_prune(FIELD<float>* skelPrev, FIELD<float>* currDT, FIELD<float>* prevDT);
};

void neighboursSet(int x, int y, FIELD<float>* skel, std::set<coord2D_t>& neighbours);
coord2D_list_t nonSkeletonNeighbours(int x, int y, FIELD<float>* skel);
std::pair<coord2D_t, float> addSkeletonPoint(int x, int y, FIELD<float>* skel, FIELD<float>* dt, FIELD<float>& skeletonEndpoints, int index, SkelEngine* skelE);
#endif
