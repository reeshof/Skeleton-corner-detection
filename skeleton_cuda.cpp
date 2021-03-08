/* skeleton.cpp */

#include <math.h>
#include "shared/fileio/fileio.hpp"
#include <cuda_runtime_api.h>
#include "shared/CUDASkel2D/include/genrl.h"
#include "shared/CUDASkel2D/include/field.h"
#include "shared/CUDASkel2D/include/skelft.h"
#include "include/ImageWriter.hpp"
#include "include/skeleton_cuda.hpp"
#include "include/connected.hpp"
#include "include/messages.h"


float SKELETON_SALIENCY_THRESHOLD;
float SKELETON_ISLAND_THRESHOLD;
float        SKELETON_DT_THRESHOLD;

#define INDEX(i,j) (i)+fboSize*(j)

float* siteParam;
float* importanceHost;//CHANGE1: added importance buffer

// The following are CUDA buffers for several images.
unsigned char* outputSkeleton;
short* outputFT;
short* skelFT;
bool* foreground_mask;
int xm = 0, ym = 0, xM, yM, fboSize;
double length;

void allocateCudaMem(int size) {
    skelft2DInitialization(size);
    cudaMallocHost((void**)&outputFT, size * size * 2 * sizeof(short));
    cudaMallocHost((void**)&foreground_mask, size * size * sizeof(bool));
    cudaMallocHost((void**)&outputSkeleton, size * size * sizeof(unsigned char));
    cudaMallocHost((void**)&siteParam, size * size * sizeof(float));
    cudaMallocHost((void**)&skelFT, size * size * 2 * sizeof(short));
}

void deallocateCudaMem() {
    skelft2DDeinitialization();
    cudaFreeHost(outputFT);
    cudaFreeHost(foreground_mask);
    cudaFreeHost(outputSkeleton);
    cudaFreeHost(siteParam);

    cudaFreeHost(skelFT);
}

void clearBuffers() {
    delete[] importanceHost;
}


int initialize_skeletonization(FIELD<float>* im) {
    xM = im->dimX();
    yM = im->dimY();
    fboSize = skelft2DSize(xM, yM);//   Get size of the image that CUDA will actually use to process our nx x ny image
   
    importanceHost = (float*)malloc(fboSize * fboSize*sizeof(float));

    allocateCudaMem(fboSize);
    return fboSize;
}

FIELD<float>* skel_to_field() {
    FIELD<float>* f = new FIELD<float>(xM, yM);
    for (int i = 0; i < xM; ++i) {
        for (int j = 0; j < yM; ++j) {
            bool is_skel_point = outputSkeleton[INDEX(i, j)];
            f->set(i, j, is_skel_point ? 255 : 0);
        }
    }
    return f;
}

// TODO(maarten): dit moet beter kunnen, i.e. in een keer de DT uit cuda halen
void dt_to_field(FIELD<float>* f) {
    for (int i = 0; i < xM; ++i) {
        for (int j = 0; j < yM; ++j) {
            int id = INDEX(i, j);
            int ox = outputFT[2 * id];
            int oy = outputFT[2 * id + 1];
            float val = sqrt((i - ox) * (i - ox) + (j - oy) * (j - oy));
            f->set(i, j, val);
        }
    }
}

coord2D_t getFT(int x, int y) {
    int id = INDEX(x, y);
    return coord2D_t(outputFT[2 * id], outputFT[2 * id + 1]);
}

float getSiteLabel(coord2D_t p) {
    int id = INDEX(p.first, p.second);
    return siteParam[id];
}

//gets the site label and also sets it to index after
float getSiteLabelReplace(coord2D_t p, int index) {
    int id = INDEX(p.first, p.second);
    auto param = siteParam[id];
    if (param != 0) {
        siteParam[id] = index;
        return param;
    }
    else
        return index;
}


FIELD<float>* skelft_to_field() {
 
    FIELD<float>* f = new FIELD<float>(xM, yM);
    
    for (int i = 0; i < xM; ++i) {
        for (int j = 0; j < yM; ++j) {
            int id = INDEX(i, j);
            /*CURRENTLY ITS SHOWING THE SITEPARAM OF EACH PIXEL (IS 0 FOR MOST, BECAUSE MOST PIXELS ARE NOT A BOUNDARY PIXEL)
            int ox = outputFT[2 * id];
           // PRINT(MSG_NORMAL, "f2: %d\n", ox);
            int oy = outputFT[2 * id + 1];
            double val = (siteParam[INDEX(ox,oy)]);
            */
            double val = siteParam[id];
            f->set(i, j, val);
        }
    }
    return f;
}

FIELD<float>* skelImportance_to_field() {

    FIELD<float>* f = new FIELD<float>(xM, yM);
    //std::cout << fboSize << std::endl;
    //std::cout << xM << " " << yM << std::endl;
    //f->setData(importanceHost);
    for (int i = 0; i < xM; ++i) {
        for (int j = 0; j < yM; ++j) {
            int id = INDEX(i, j);
            
            double val = (importanceHost[id]);
            f->set(i, j, val);
        }
    }
    return f;
}

void analyze_cca(int level, FIELD<float>* skel) {

    float* ssd = skel->data();
    
    ConnectedComponents *CC = new ConnectedComponents(255);
    int                 *ccaOut = new int[skel->dimX() * skel->dimY()];
    int                 hLabel;
    unsigned int        *hist;
    int                 i,
                        nPix = skel->dimX() * skel->dimY();

    /* Perform connected component analysis */
    hLabel = CC->connected(ssd, ccaOut, skel->dimX(), skel->dimY(), std::equal_to<float>(), true);
    hist = static_cast<unsigned int*>(calloc(hLabel + 1, sizeof(unsigned int)));    
    if (!hist) {
        PRINT(MSG_ERROR, "Error: Could not allocate histogram for skeleton connected components\n");
        exit(-1);
    }
    //for (i = 0; i < hLabel; i++) { hist[i] = 0; /*if (hist[i] != 0) {/*cout<<hist[i]<<endl; counNonZero =+ 1;}*/ }
 
    for (i = 0; i < nPix; i++) { hist[ccaOut[i]]++; }
    
    /* Remove small islands */
    for (i = 0; i < nPix; i++) {
        ssd[i] = (hist[ccaOut[i]] >= SKELETON_ISLAND_THRESHOLD/100*sqrt(skel->dimX()*skel->dimY())) ? ssd[i] : 0;
       //ssd[i] = (hist[ccaOut[i]] >= SKELETON_ISLAND_THRESHOLD && hist[ccaOut[i]] <= SKELETON_ISLAND_THRESHOLD * 2) ? ssd[i] : 0;
        
       //  ssd[i] = (hist[ccaOut[i]] >= SKELETON_ISLAND_THRESHOLD) ? ssd[i] : 0;
    }

    free(hist);
    delete CC;
    delete [] ccaOut;

    skel->setData(ssd);
    //skel->writePGM("skel_removeIsland.pgm");
   /* 
   if (level == 44) {
        skel -> writePGM("layers_44.pgm");
   }
   */ 
}

FIELD<float>* computeSkeleton(int level, FIELD<float> *input, bool doReverse, float saliency) {
   
    memset(siteParam, 0, fboSize * fboSize * sizeof(float));

 
    memset(foreground_mask, false, fboSize * fboSize * sizeof(bool));
    
    //unsigned char* image = new unsigned char[fboSize*fboSize];

    int nx = input->dimX();
    int ny = input->dimY();
    xm = ym = nx; xM = yM = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (!(*input)(i, j)) {//0
                foreground_mask[INDEX(i, j)] = true;
                siteParam[INDEX(i, j)] = 1;
                //image[j*fboSize+i] = 255;
                xm = min(xm, i); ym = min(ym, j);
                xM = max(xM, i); yM = max(yM, j);
            }
        }
    }
    //xM = nx - 1; yM = ny - 1;//orginal method, which cause artifacts.
    xM = (xm==nx)? (nx - 1): nx;//must run before xm=...
    yM = (xm==nx)? (ny - 1): ny;
    xm = (xm==0)? 0 : (xm-1); // In this way, we further expand the boundary, which avoid producing artifacts.
    ym = (ym==0)? 0 : (ym-1);
   
    skelft2DFT(0, siteParam, 0, 0, fboSize, fboSize, fboSize);
    skelft2DDT(outputFT, 0, xm, ym, xM, yM);
    
    //length = skelft2DMakeBoundary((unsigned char*)outputFT, xm, ym, xM, yM, siteParam, fboSize, doReverse, 0, false);
	
    if (!length) return NULL;
    //skelft2DFillHoles((unsigned char*)outputFT, xm + 1, ym + 1, 1);
    skelft2DFT(outputFT, siteParam, xm, ym, xM, yM, fboSize);
    //skelft2DSkeleton(outputSkeleton, foreground_mask, length, saliency, xm, ym, xM, yM, importanceHost, fboSize);
    //skel2DSkeletonFT(skelFT, xm, ym, xM, yM);
    dt_to_field(input);
    auto skel = skel_to_field();
   //if (level==53) skel->writePGM("skel53.pgm");
    analyze_cca(level, skel);
    return skel;
}

short* get_current_skel_ft() {
    size_t sz = fboSize * fboSize * 2;
    short* skelft = (short*)(malloc(sz * sizeof(short)));
    if (!skelft) {
        PRINT(MSG_ERROR, "Error: could not allocate array for skeleton FT\n");
        exit(-1);
    }
    memcpy(skelft, skelFT, sz);
    return skelft;
}
