#pragma once

#include "cudawrapper.h"
#include "field.h"


class SkelEngine                                //Engine that computes FTs, DTs, and (thresholded) skeletons. Holds all computed info internally.
{
public:
              SkelEngine();
             ~SkelEngine();

void          initialize(unsigned char* image,short fboSize,short xm,short ym,short xM,short yM, bool scan_bot_to_top);
                                                  //Initialize this from given area (xm,ym,xM,yM) of binary image image[] of size fboSoze*fboSize. Compute FT.
void          compute(float threshold);           //Compute simplified skeleton for given threshold
void          computeEndpoints();

bool          isBoundary(short x, short y) const; //Whether (x,y) is on the shape boundary
float         boundaryParam(short x, short y) const;  //Norrmalized boundary-param value [0,1] for (x,y) on boundary.
bool          isSkeleton(short x,short y) const;  //Whether (x,y) is on the simplified skeleton
short         imageSize() const;                  //Size (x==y) of all images used by this engine
int           numEndpoints() const;               //# detected skel-branch endpoints
short2        getEndpoint(int) const;             //Return i-th endpoint

//Added functions
FIELD<float>* getSkelField() const;
FIELD<float>* getFtField() const;
FIELD<float>* getImpField() const;

std::pair<int,int> getFT(int x, int y);
float getSiteLabel(std::pair<int,int> p);
//gets the site label and also sets it to index after
float getSiteLabelReplace(std::pair<int,int> p, int index);


private:

float*         siteParam;													//Boundary parameterization image (fboSize*fboSize). value(i,j) = boundary-param of (i,j) if on boundary, else unused
short*         outputFT;													//1-point FT image (fboSize*fboSize*2). value(i,j) = the x,y coords of closest boundary-site to (i,j)
unsigned char* outputSkeleton;										//Skeleton: fboSize x fboSize. value(i,j)!=0 => (i,j) is in the simplified skeleton
float*         importanceHost;
short*         outputTopo;                        //Endpoints of skel branches (nendpoints*2). value(i) = x,y coords of i-th skel-branch endpoint
int            nendpoints;                        //Number of skel-branch endpoints in outputTopo[]
int            fboSize;                           //All images in this engine have this size (squared)
short          xm,ym,xM,yM;                       //Bounding-box in images where all relevant info resides (the rest is background)
float          length;                            //
float          maxImp;                            //Maximum value of the skeleton importance (AFMM)
short2         maxPos;                            //Position on skeleton where max importance occurs; so to speak 'center' of skeleton
};

inline bool SkelEngine::isBoundary(short x,short y) const
{
  return siteParam[y*fboSize+x] >= 0;
}

inline float SkelEngine::boundaryParam(short x,short y) const
{
  unsigned int id = 2*(y*fboSize+x);
  short ox = outputFT[id], oy = outputFT[id+1];
  unsigned int vid = oy * fboSize + ox;
  return siteParam[vid]/length;
}

inline bool SkelEngine::isSkeleton(short x,short y) const
{
  return outputSkeleton[y*fboSize+x] != 0;
}

inline short SkelEngine::imageSize() const
{
  return fboSize;
}

inline int SkelEngine::numEndpoints() const
{
  return nendpoints;
}

inline short2 SkelEngine::getEndpoint(int i) const
{
  unsigned int idx =  2*i;
  return make_short2(outputTopo[idx],outputTopo[idx+1]);
}
