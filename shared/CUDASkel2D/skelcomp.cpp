#include "include/skelcomp.h"
#include "include/skelft.h"


// CUDA utilities and system includes
#include "include/cudawrapper.h"
#include <cuda_gl_interop.h>
#include <stack>



using namespace std;

#define INDEX(i,j) (i)+fboSize*(j)

const int MAX_IMAGE_SIZE = 1024;                  //Max image size used by CUDA. Input image must be smaller
const int MAX_NENDPOINTS = 10000;                 //Max #endpoints we want to detect
const int MAX_SKELPOINTS = 50000;                 //Max # points in skeleton, for the current simplification level
short2    skelpoints[MAX_SKELPOINTS];             //Skel points detected in the current skeleton
int       num_skelpoints = 0;                     //#entries in skelpoints[]

void computeCoreComponent(unsigned char* outputSkeleton, int SZ, short2 seed, int& num_skelpoints, short2* skelpoints)
{
	//Do flood fill through 'seed' on 'outputSkeleton', eliminate rest; Do this on the CPU since far easier than on the GPU

  stack<short2> q;
  q.push(seed);                                         //Start process from pixel 'seed'
  outputSkeleton[seed.x+SZ*seed.y] = 128;								//Mark seed as 'visited'
  num_skelpoints = 0;																		//We start gathering a new set of connected 2D skelpoints

  while(!q.empty())																			//Do the flood-fill
  {
    short2 p = q.top(); q.pop();												//Get a seed; seeds have a value of 128
    skelpoints[num_skelpoints] = p;											//One more skelpoint connected to 'seed'

    for(int i=p.x-1;i<=p.x+1;++i)												//Flood fill seed's on-the-skeleton neighbors
      for(int j=p.y-1;j<=p.y+1;++j)
      {
        if (i==p.x && j==p.y) continue;									//Skip seed itself
        if (outputSkeleton[i+SZ*j]!=255) continue;			//Neighbor is either background or visited, skip it
        outputSkeleton[i+SZ*j] = 128;										//Neighbor is skeleton and unvisited: mark it as 'visited'
        q.push(make_short2(i,j));												//...and add it to the flood fill queue
      }

    ++num_skelpoints;
  }

  //skelft2DComputeCoreComponent(outputSkeleton,0,0);      //Replicate the same data (skeleton core component) on the GPU
}




SkelEngine::SkelEngine()
{
    skelft2DInitialization(MAX_IMAGE_SIZE);
  	cudaMallocHost((void**)&outputFT,MAX_IMAGE_SIZE*MAX_IMAGE_SIZE*2*sizeof(short));
  	cudaMallocHost((void**)&outputSkeleton,MAX_IMAGE_SIZE*MAX_IMAGE_SIZE*sizeof(unsigned char));
  	cudaMallocHost((void**)&siteParam,MAX_IMAGE_SIZE*MAX_IMAGE_SIZE*sizeof(float));
  	cudaMallocHost((void**)&outputTopo,MAX_IMAGE_SIZE*MAX_IMAGE_SIZE*2*sizeof(short));

}

SkelEngine::~SkelEngine()
{
  skelft2DDeinitialization();
	cudaFreeHost(outputFT);
	cudaFreeHost(outputSkeleton);
	cudaFreeHost(siteParam);
	cudaFreeHost(outputTopo);
    delete[] importanceHost;
}


void SkelEngine::initialize(unsigned char* image,short fboSize_,short xm_,short ym_,short xM_,short yM_, bool scan_bot_to_top)
{
  xm = xm_; ym = ym_; xM = xM_; yM = yM_; fboSize = fboSize_;

  memset(siteParam,0,fboSize*fboSize*sizeof(float));					    //2. Create siteParam simply by thresholding input image (we'll only use it for DT)

  length = skelft2DMakeBoundary((unsigned char*)image,xm,ym,xM,yM,siteParam, fboSize, scan_bot_to_top, 1,true);
                                    //5. Parameterize boundary of inflated shape into 'siteParam' for skeleton computation

  skelft2DFT(outputFT,siteParam,xm,ym,xM,yM,fboSize);			//6. Compute FT of 'siteParam'

  importanceHost = (float*)malloc(fboSize * fboSize * sizeof(float));
}


void SkelEngine::compute(float threshold)
{

  //cout<<"Computing simplified skeleton for thr="<<threshold<<endl;
  skelft2DSkeleton(outputSkeleton,length,threshold,xm,ym,xM,yM,&maxImp,&maxPos, importanceHost, fboSize);	//7. Skeletonize the FT into 'outputSkeleton'
  //computeCoreComponent(outputSkeleton,fboSize,maxPos,num_skelpoints,skelpoints);
  //nendpoints = MAX_NENDPOINTS;                                                   //MAx #endpoints we want to detect
  //skelft2DTopology(0,outputSkeleton,&nendpoints,outputTopo,xm,ym,xM,yM);				//8. Detect endpoints of the skeleton, put them in outputTopo[]

  //cout<<"Found skeleton endpoints: "<<nendpoints<<endl;
}

void SkelEngine::computeEndpoints() {
    skelft2DTopology(0, outputSkeleton, &nendpoints, outputTopo, xm, ym, xM, yM);
}

FIELD<float>* SkelEngine::getSkelField() const
{
    FIELD<float>* f = new FIELD<float>(xM, yM);
    for (int i = 0; i < xM; ++i) {
        for (int j = 0; j < yM; ++j) {
            bool is_skel_point = outputSkeleton[INDEX(i, j)];
            f->set(i, j, is_skel_point ? 255 : 0);
        }
    }
    return f;
}

FIELD<float>* SkelEngine::getFtField() const
{
    FIELD<float>* f = new FIELD<float>(xM, yM);

    for (int i = 0; i < xM; ++i) {
        for (int j = 0; j < yM; ++j) {
            int id = INDEX(i, j);
            double val = siteParam[id];
            f->set(i, j, val);
        }
    }
    return f;
}

FIELD<float>* SkelEngine::getImpField() const
{
    FIELD<float>* f = new FIELD<float>(xM, yM);
    
    for (int i = 0; i < xM; ++i) {
        for (int j = 0; j < yM; ++j) {
            int id = INDEX(i, j);

            double val = (importanceHost[id]);
            f->set(i, j, val);
        }
    }
    return f;
}

std::pair<int, int> SkelEngine::getFT(int x, int y)
{
    int id = INDEX(x, y);
    return std::pair<int,int>(outputFT[2 * id], outputFT[2 * id + 1]);
}

float SkelEngine::getSiteLabel(std::pair<int, int> p)
{
    int id = INDEX(p.first, p.second);
    return siteParam[id];
}

float SkelEngine::getSiteLabelReplace(std::pair<int, int> p, int index)
{
    int id = INDEX(p.first, p.second);
    auto param = siteParam[id];
    if (param != 0) {
        siteParam[id] = index;
        return param;
    }
    else
        return index;
}

