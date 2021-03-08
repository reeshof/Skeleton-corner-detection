#include <cuda_runtime.h>
#include "include/skelft.h"
#include <stdio.h>



// Parameters for CUDA kernel executions; more or less optimized for a 1024x1024 image.
#define BLOCKX		16
#define BLOCKY		16
#define BLOCKSIZE	64
#define TILE_DIM	32
#define BLOCK_ROWS	16



/****** Global Variables *******/
const int NB = 7;						// Nr buffers we use and store in the entire framework
short2 **pbaTextures;					// Work buffers used to compute and store resident images
//	0: work buffer
//	1: FT
//	2: thresholded DT
//	3: thresholded skeleton
//	4: topology analysis
//  5: work buffer for topology
//  6: skeleton FT
//

float*			pbaTexSiteParam;		// Stores boundary parameterization (>0: boundary-code; =0: non-boundary pixel)
int					pbaTexSize;					// Texture size (squared) actually used in all computations
int					floodBand  = 4,			// Various FT computation parameters; defaults are good for an 1024x1024 image.
						maurerBand = 4,
						colorBand  = 4;

texture<short2> pbaTexColor;			// 2D textures (bound to various buffers defined above as needed)
texture<short2> pbaTexColor2;			//
texture<short2> pbaTexLinks;
texture<float>  pbaTexParam;			// 1D site parameterization texture (bound to pbaTexSiteParam)
texture<unsigned char> pbaTexGray;	// 2D texture of unsigned char values, e.g. the binary skeleton

#if __CUDA_ARCH__ < 110					// We cannot use atomic intrinsics on SM10 or below. Thus, we define these as nop.
//#define atomicInc(a,b) 0				// The default will be that some code e.g. endpoint detection will thus not do anything.
#endif


__device__ unsigned int topo_gc			= 0;		//Used for pixel-array-creation (for skeleton endpoints and skeletons)
__device__ unsigned int topo_gc_last	= 0;

#define X 255

__constant__ const																					//REMARK: put following constants (for kernelTopology) in CUDA constant-memory, as this gives a huge speed difference
unsigned char topo_patterns[][9] =		{ {0,0,0,							//These are the 3x3 templates that we use to detect skeleton endpoints
										   0,X,0,																//(with four 90-degree rotations for each)
										   0,X,0},
										  {0,0,0,
										   0,X,0,
										   0,0,X},
										  {0,0,0,
										   0,X,0,
										   0,X,X},
										  {0,0,0,
										   0,X,0,
										   X,X,0}
										};

#define topo_NPATTERNS  4																	//Number of patterns we try to match (for kernelTopology)
																													//REMARK: #define faster than __constant__

__constant__ const unsigned char topo_rot[][9] = { {0,1,2,3,4,5,6,7,8}, {2,5,8,1,4,7,0,3,6}, {8,7,6,5,4,3,2,1,0}, {6,3,0,7,4,1,8,5,2} };
																				//These encode the four 90-degree rotations of the patterns (for kernelTopology);


#include "include/skelftKernel.h"



// Initialize necessary memory (CPU/GPU sides)
// - textureSize: The max size of any image we will process until re-initialization
void skelft2DInitialization(int maxTexSize)
{
	  cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp,0);																			// Query device properties, list something about them

    int pbaMemSize = maxTexSize * maxTexSize * sizeof(short2);								// A buffer has 2 shorts / pixel

    pbaTextures  = (short2 **) malloc(NB * sizeof(short2*));									// We will use NB buffers

	  for(int i=0;i<NB;++i)
       cudaMalloc((void **) &pbaTextures[i], pbaMemSize);											// Allocate work buffer 'i'

    cudaMalloc((void **) &pbaTexSiteParam, maxTexSize * maxTexSize * sizeof(float));		// Sites texture
}

// Deallocate all allocated memory
void skelft2DDeinitialization()
{
    for(int i=0;i<NB;++i) cudaFree(pbaTextures[i]);
	cudaFree(pbaTexSiteParam);
    free(pbaTextures);
}

__global__ void kernelSiteParamInit(short2* inputVoro, int size)							//Initialize the Voronoi textures from the sites' encoding texture (parameterization)
{																																							//REMARK: we interpret 'inputVoro' as a 2D texture, as it's much easier/faster like this
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx<size && ty<size)																											//Careful not to go outside the image..
	{
	  int i = TOID(tx,ty,size);
	  float param = tex1Dfetch(pbaTexParam,i);																	//The sites-param has non-zero (parameter) values precisely on non-boundary points

	  short2& v = inputVoro[i];
	  v.x = v.y = MARKER;																												//Non-boundary points are marked as 0 in the parameterization. Here we will compute the FT.
	  if (param>0)																															//These are points which define the 'sites' to compute the FT/skeleton (thus, have FT==identity)
	  {																																					//We could use an if-then-else here, but it's faster with an if-then
	     v.x = tx; v.y = ty;
	  }
	}
}



void skelft2DInitializeInput(float* siteParam, int size)									// Copy input sites from CPU to GPU; Also set up site param initialization in pbaTextures[0]
{
    pbaTexSize = size;																										// Size of the actual texture being used in this run; can be smaller than the max-tex-size
																																					// which was used in skelft2DInitialization()

	cudaMemcpy(pbaTexSiteParam, siteParam, pbaTexSize * pbaTexSize * sizeof(float), cudaMemcpyHostToDevice);
																																					// Pass sites parameterization to CUDA.  Must be done before calling the initialization
																																					// kernel, since we use the sites-param as a texture in that kernel
	cudaBindTexture(0, pbaTexParam, pbaTexSiteParam);												// Bind the sites-param as a 1D texture so we can quickly index it next
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

	kernelSiteParamInit<<<grid,block>>>(pbaTextures[0],pbaTexSize);					// Do the site param initialization. This sets up pbaTextures[0]
	cudaUnbindTexture(pbaTexParam);																					// Done with the sites-param 1D texture
}





// In-place transpose a squared texture.
// Block orders are modified to optimize memory access.
// Point coordinates are also swapped.
void pba2DTranspose(short2 *texture)
{
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(pbaTexSize / TILE_DIM, pbaTexSize / TILE_DIM);

    cudaBindTexture(0, pbaTexColor, texture);
    kernelTranspose<<< grid, block >>>(texture, pbaTexSize);
    cudaUnbindTexture(pbaTexColor);
}

// Phase 1 of PBA. m1 must divides texture size
void pba2DPhase1(int m1, short xm, short ym, short xM, short yM)
{
    dim3 block = dim3(BLOCKSIZE);
    dim3 grid = dim3(pbaTexSize / block.x, m1);

    // Flood vertically in their own bands
    cudaBindTexture(0, pbaTexColor, pbaTextures[0]);
    kernelFloodDown<<< grid, block >>>(pbaTextures[1], pbaTexSize, pbaTexSize / m1);
    cudaUnbindTexture(pbaTexColor);

    cudaBindTexture(0, pbaTexColor, pbaTextures[1]);
    kernelFloodUp<<< grid, block >>>(pbaTextures[1], pbaTexSize, pbaTexSize / m1);

    // Passing information between bands
    grid = dim3(pbaTexSize / block.x, m1);
    kernelPropagateInterband<<< grid, block >>>(pbaTextures[0], pbaTexSize, pbaTexSize / m1);

    cudaBindTexture(0, pbaTexLinks, pbaTextures[0]);
    kernelUpdateVertical<<< grid, block >>>(pbaTextures[1], pbaTexSize, m1, pbaTexSize / m1);
    cudaUnbindTexture(pbaTexLinks);
    cudaUnbindTexture(pbaTexColor);
}

// Phase 2 of PBA. m2 must divides texture size
void pba2DPhase2(int m2)
{
    // Compute proximate points locally in each band
    dim3 block = dim3(BLOCKSIZE);
    dim3 grid = dim3(pbaTexSize / block.x, m2);
    cudaBindTexture(0, pbaTexColor, pbaTextures[1]);
    kernelProximatePoints<<< grid, block >>>(pbaTextures[0], pbaTexSize, pbaTexSize / m2);

    cudaBindTexture(0, pbaTexLinks, pbaTextures[0]);
    kernelCreateForwardPointers<<< grid, block >>>(pbaTextures[0], pbaTexSize, pbaTexSize / m2);

    // Repeatly merging two bands into one
    for (int noBand = m2; noBand > 1; noBand /= 2) {
        grid = dim3(pbaTexSize / block.x, noBand / 2);
        kernelMergeBands<<< grid, block >>>(pbaTextures[0], pbaTexSize, pbaTexSize / noBand);
    }

    // Replace the forward link with the X coordinate of the seed to remove
    // the need of looking at the other texture. We need it for coloring.
    grid = dim3(pbaTexSize / block.x, pbaTexSize);
    kernelDoubleToSingleList<<< grid, block >>>(pbaTextures[0], pbaTexSize);
    cudaUnbindTexture(pbaTexLinks);
    cudaUnbindTexture(pbaTexColor);
}

// Phase 3 of PBA. m3 must divides texture size
void pba2DPhase3(int m3)
{
    dim3 block = dim3(BLOCKSIZE / m3, m3);
    dim3 grid = dim3(pbaTexSize / block.x);
    cudaBindTexture(0, pbaTexColor, pbaTextures[0]);
    kernelColor<<< grid, block >>>(pbaTextures[1], pbaTexSize);
    cudaUnbindTexture(pbaTexColor);
}



void skel2DFTCompute(short xm, short ym, short xM, short yM, int floodBand, int maurerBand, int colorBand)
{
    pba2DPhase1(floodBand,xm,ym,xM,yM);										//Vertical sweep

    pba2DTranspose(pbaTextures[1]);											//

    pba2DPhase2(maurerBand);												//Horizontal coloring

    pba2DPhase3(colorBand);													//Row coloring

    pba2DTranspose(pbaTextures[1]);
}





__global__ void kernelThresholdDT(unsigned char* output, int size, float threshold2, short xm, short ym, short xM, short yM)
//Input:    pbaTexColor: closest-site-ids per pixel, i.e. FT
//Output:   output: thresholded DT
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx>xm && ty>ym && tx<xM && ty<yM)									//careful not to index outside the image..
	{
  	  int    id     = TOID(tx, ty, size);
	  short2 voroid = tex1Dfetch(pbaTexColor,id);							//get the closest-site to tx,ty into voroid.x,.y
	  float  d2     = (tx-voroid.x)*(tx-voroid.x)+(ty-voroid.y)*(ty-voroid.y);
	  output[id]    = (d2<=threshold2);										//threshold DT into binary image
    }
}



__global__ void kernelDT(short* output, int size, float threshold2, short xm, short ym, short xM, short yM)
//Input:    pbaTexColor: closest-site-ids per pixel, i.e. FT
//Output:   output: DT
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx>xm && ty>ym && tx<xM && ty<yM)									//careful not to index outside the image..
	{
  	  int    id     = TOID(tx, ty, size);
	    short2 voroid = tex1Dfetch(pbaTexColor,id);							//get the closest-site to tx,ty into voroid.x,.y
	    float  d2     = (tx-voroid.x)*(tx-voroid.x)+(ty-voroid.y)*(ty-voroid.y);
	    output[id]    = sqrtf(d2);											//save the Euclidean DT
  }
}





__global__ void kernelSkel(unsigned char* output, short xm, short ym,
						   short xM, short yM, short size, float threshold, float length,
						   float*  output_skelmaxImp, short2* output_skelmaxPos, const float SKELFT_FAR_AWAY, float* importanceDevice)
																			//Input:    pbaTexColor: closest-site-ids per pixel
																			//			    pbaTexParam: labels for sites (only valid at site locations)
{																			//Output:	  output: binary thresholded skeleton; serialized skeleton pixels
																			//WARNING: this kernel may sometimes create 2-pixel-thick branches.. Study the AFMM original code to see if this is correct.

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int    id = TOID(tx, ty, size);

	//compare the current location with its own feature transform. For boundary points these are the same, thus onyl the boundary points are ignored.
	short2 curLocation { tx,ty };
	short2 ftLocation = tex1Dfetch(pbaTexColor, id);

	if (tx>xm && ty>ym && tx<xM-1 && ty<yM-1 && !(ftLocation.x == curLocation.x && ftLocation.y == curLocation.y))
	{
  	
	  //if (tex1Dfetch(pbaTexParam,id)==SKELFT_FAR_AWAY)						//compute only the foreground-skeleton (the background one is usually not interesting)
	  //{
	    int    Id     = id;
	    short2 voroid = tex1Dfetch(pbaTexColor,id);							  //get the closest-site to tx,ty into voroid.x,.y
		  float  d2     = sqrtf((tx-voroid.x)*(tx-voroid.x)+(ty-voroid.y)*(ty-voroid.y));	//get DT^2 of tx,ty

	    int    id2    = TOID(voroid.x,voroid.y,size);						//convert the site's coord to an index into pbaTexParam[], the site-label-texture
	    float  imp    = tex1Dfetch(pbaTexParam,id2);						//get the site's label

	           ++id;																						//TOID(tx+1,ty,size)
	           voroid = tex1Dfetch(pbaTexColor,id);							//
	           id2    = TOID(voroid.x,voroid.y,size);						//
	    float  imp_r  = tex1Dfetch(pbaTexParam,id2);						//

			id -= 2;																						//TOID(tx+1,ty,size)
			voroid = tex1Dfetch(pbaTexColor, id);							//
			id2 = TOID(voroid.x, voroid.y, size);						//
		float  imp_r2 = tex1Dfetch(pbaTexParam, id2);						//

	           id     += size+1;																//TOID(tx,ty+1,size)
	           voroid = tex1Dfetch(pbaTexColor,id);							//
	           id2    = TOID(voroid.x,voroid.y,size);						//
	    float  imp_u  = tex1Dfetch(pbaTexParam,id2);						//

			id -= size*2;																//TOID(tx,ty+1,size)
			voroid = tex1Dfetch(pbaTexColor, id);							//
			id2 = TOID(voroid.x, voroid.y, size);						//
		float  imp_u2 = tex1Dfetch(pbaTexParam, id2);						//

		float imp_dx  = fabsf(imp_r-imp);
		float imp_dx2 = fabsf(imp_r2 - imp);
	    float imp_dy  = fabsf(imp_u-imp);
		float imp_dy2 = fabsf(imp_u2 - imp);
	    float Imp     = max(imp_dy2,max(imp_dx2,max(imp_dx,imp_dy)));
	    Imp = min(Imp,fabsf(length-Imp));
		  float Imp_sal = Imp / d2;										//Compute salience metric

		  if (Imp_sal >= threshold && Imp > 1 && !(Imp < 5 && d2 < 1.5f))									//REMARK: exclude Imp==1 points since these are the boundary
			  {
				  output[Id] = 255;																//By filling only in-skeleton-values, we reduce memory access somehow (writing to output[] is expensive)

				  int id = blockIdx.x + blockIdx.y * gridDim.x;
				  if (Imp > output_skelmaxImp[id])
				  {
					  output_skelmaxImp[id] = Imp;
					  output_skelmaxPos[id] = make_short2(tx, ty);
				  }
			  }
		importanceDevice[Id] = Imp;//CHANGE2: 
	  //}
	}
}


__global__ void kernelTopology(unsigned char* output, short2* output_set, short xm, short ym, short xM, short yM, short size, int maxpts)
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char t[9];

	if (tx>xm && ty>ym && tx<xM-1 && ty<yM-1)									//careful not to index outside the image; take into account the template size too
	{
	   int    id = TOID(tx, ty, size);
	   unsigned char  p  = tex1Dfetch(pbaTexGray,id);							//get the skeleton pixel at tx,ty
	   if (p)																	//if the pixel isn't skeleton, nothing to do
	   {
	     unsigned char idx=0;
		   for(int j=ty-1;j<=ty+1;++j)											//read the template into t[] for easier use
		   {
		     int id = TOID(tx-1, j, size);
	       for(int i=0;i<=2;++i,++id,++idx)
		         t[idx] = tex1Dfetch(pbaTexGray,id);								//get the 3x3 template centered at the skel point tx,ty
		   }

		   for(unsigned char r=0;r<4;++r)											//try to match all rotations of a pattern:
		   {
		     const unsigned char* rr = topo_rot[r];
	       for(unsigned char p=0;p<topo_NPATTERNS;++p)							//try to match all patterns:
	       {
	         const unsigned char* pat = topo_patterns[p];
			     unsigned char j = (p==0)? 0 : 7;									//Speedup: for all patterns except 1st, check only last 3 entries, the first 6 are identical for all patterns
			     for(;j<9;++j)														//try to match rotated pattern vs actual pattern
			       if (pat[j]!=t[rr[j]]) break;										//this rotation failed
			     if (j<6) break;													//Speedup: if we have a mismatch on the 1st 6 pattern entries, then none of the patterns can match
																				//		   since all templates have the same first 6 entries.

			     if (j==9)															//this rotation succeeded: mark the pixel as a topology event and we're done
			     {
				     int crt_gc = atomicInc(&topo_gc,maxpts);						//REMARK: this serializes (compacts) all detected endpoints in one array.
						 output_set[crt_gc] = make_short2(tx,ty);						//To do this, we use an atomic read-increment-return on a global counter,
																				//which is guaranteed to give all threads unique consecutive indexes in the array.
			       output[id] = 1;													//Also create the topology image
				     return;
			     }
		     }
		   }
	   }
	}
	else																		//Last thread: add zero-marker to the output point-set, so the reader knows how many points are really in there
	if (tx==xM-1 && ty==yM-1)													//Also reset the global vector counter topo_gc, for the next parallel-run of this function
	{
		//!!!topo_gc_last = topo_gc; topo_gc = 0;
	}									//We do this in the last thread so that no one modifies topo_gc from now on.
																				//REMARK: this seems to be the only way I can read a __device__ variable back to the CPU
}




void skelft2DParams(int floodBand_, int maurerBand_, int colorBand_)		//Set up some params of the FT algorithm
{
  floodBand   = floodBand_;
  maurerBand  = maurerBand_;
  colorBand   = colorBand_;
}





// Compute 2D FT / Voronoi diagram of a set of sites
// siteParam:   Site parameterization. 0 = non-site points; >0 = site parameter value.
// output:		FT. The (x,y) at (i,j) are the coords of the closest site to (i,j)
// size:        Texture size (pow 2)
void skelft2DFT(short* output, float* siteParam, short xm, short ym, short xM, short yM, int size)
{
    skelft2DInitializeInput(siteParam,size);								    // Initialization of already-allocated data structures

    skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);			// Compute FT

    // Copy FT to CPU, if required
    if (output) cudaMemcpy(output, pbaTextures[1], size*size*sizeof(short2), cudaMemcpyDeviceToHost);
}








void skelft2DDT(short* outputDT, float threshold,								//Compute (thresholded) DT (into pbaTextures[2]) from resident FT (in pbaTextures[1])
					  short xm, short ym, short xM, short yM)
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

    cudaBindTexture(0, pbaTexColor, pbaTextures[1]);							//Used to read the FT from

	if (threshold>=0)
	{
	  xm -= threshold; if (xm<0) xm=0;
	  ym -= threshold; if (ym<0) ym=0;
	  xM += threshold; if (xM>pbaTexSize-1) xM=pbaTexSize-1;
	  yM += threshold; if (yM>pbaTexSize-1) yM=pbaTexSize-1;

      kernelThresholdDT<<< grid, block >>>((unsigned char*)pbaTextures[2], pbaTexSize, threshold*threshold, xm-1, ym-1, xM+1, yM+1);
      cudaUnbindTexture(pbaTexColor);

	  //Copy thresholded image to CPU
	  if (outputDT) cudaMemcpy(outputDT, (unsigned char*)pbaTextures[2], pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	}
	else
	{
	  xm = ym = 0; xM = yM = pbaTexSize-1;
	  kernelDT <<< grid, block >>>((short*)pbaTextures[2], pbaTexSize, threshold*threshold, xm-1, ym-1, xM+1, yM+1);
      cudaUnbindTexture(pbaTexColor);
	  //Copy thresholded image to CPU
	  if (outputDT) cudaMemcpy(outputDT, pbaTextures[2], pbaTexSize * pbaTexSize * sizeof(short), cudaMemcpyDeviceToHost);
	}
}



void skelft2DSkeleton(unsigned char* outputSkel, float length, float threshold,	//Compute thresholded skeleton (into pbaTextures[3]) from resident FT (in pbaTextures[1])
					  short xm,short ym,short xM,short yM,
					  float* globalMaxImp, short2* globalMaxPos, float* importanceHost, int fboSize)
{																				//length:     boundary length
	dim3 block = dim3(BLOCKX,BLOCKY);											//threshold:  skeleton importance min-value (below this, we ignore branches)
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);
	int  SZ	   = grid.x*grid.y;

	cudaBindTexture(0, pbaTexColor, pbaTextures[1]);							//Used to read the resident FT
	cudaBindTexture(0, pbaTexParam, pbaTexSiteParam);							//Used to read the resident boundary parameterization
	cudaMemset(outputSkel,0,sizeof(unsigned char)*pbaTexSize*pbaTexSize);		//Faster to zero result and then fill only 1-values (see kernel)
	cudaMemset(pbaTextures[0],0,sizeof(float)*SZ);								//Use pbaTextures[0] and [2] to collect the max-importance and argmax-importance per grid-block, respectively

	float* importanceDevice;
	cudaMalloc((void**)&importanceDevice, fboSize * fboSize * sizeof(float));//CHANGE1: added importance buffer

	kernelSkel<<< grid, block >>>(outputSkel, xm, ym, xM-1, yM-1, pbaTexSize, threshold, length, (float*)pbaTextures[0],pbaTextures[2], SKELFT_FAR_AWAY, importanceDevice);
																				//Compute the skeleton; it's returned both as a texture (outputSkel)
	cudaUnbindTexture(pbaTexColor);
	cudaUnbindTexture(pbaTexParam);

	/*float* maxImp = new float[SZ];															//Now compute the global (per-image) importance max and arg-max.
	short2* maxPos = new short2[SZ];
	cudaMemcpy(maxImp,pbaTextures[0],SZ*sizeof(float),cudaMemcpyDeviceToHost);	//Get the per-grid-bloc max and arg-max from CUDA. We need this on the CPU to compute the per-image max and arg-max
	cudaMemcpy(maxPos,pbaTextures[2],SZ*sizeof(short2),cudaMemcpyDeviceToHost);	//*/

	cudaMemcpy(importanceHost, importanceDevice, fboSize * fboSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(importanceDevice);

	//*globalMaxImp = -1;															//This will store the per-image max and arg-max of the skel importance.
	//for(int i=0;i<SZ;++i)														//Aggregate per-grid max values into the global max
	//   if (maxImp[i] > *globalMaxImp)
	 //  {
	//	   *globalMaxImp = maxImp[i];
	//	   *globalMaxPos = maxPos[i];
	 //  }
	//delete[] maxImp;
	//delete[] maxPos;
}




void skelft2DTopology(unsigned char* outputTopo, unsigned char* inputSkel, int* npts, short* outputPoints, //Compute topology-points of the resident skeleton (in pbaTextures[3])
					  short xm,short ym,short xM,short yM)
{
  int maxpts = (npts)? *npts : pbaTexSize*pbaTexSize;							//This is the max # topo-points we are going to return in outputPoints[]

	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

  cudaBindTexture(0, pbaTexGray, inputSkel);								//Used to read the resident skeleton
	cudaMemset(pbaTextures[4],0,sizeof(unsigned char)*pbaTexSize*pbaTexSize);	//Faster to zero result and then fill only 1-values (see kernel)

  unsigned int zero = 0;
	cudaMemcpyToSymbol(topo_gc,&zero,sizeof(unsigned int),0,cudaMemcpyHostToDevice);		//Set topo_gc to 0

  kernelTopology<<< grid, block >>>((unsigned char*)pbaTextures[4], pbaTextures[5], xm, ym, xM, yM, pbaTexSize, maxpts+1);

  cudaUnbindTexture(pbaTexGray);

	if (outputPoints && maxpts)													//If output-point vector desired, copy the end-points, put in pbaTexture[5] as a vector of short2's,
	{																			//into caller space. We copy only 'maxpts' elements, as the user instructed us.
	  unsigned int num_pts;
		cudaMemcpyFromSymbol(&num_pts,topo_gc,sizeof(unsigned int),0,cudaMemcpyDeviceToHost);		//Get #topo-points we have detected from the device-var from CUDA

		if (npts && num_pts)																			//Copy the topo-points to caller
		   cudaMemcpy(outputPoints,pbaTextures[5],num_pts*sizeof(short2),cudaMemcpyDeviceToHost);
		if (npts) *npts = num_pts;												//Return #detected topo-points to caller
	}

	if (outputTopo)																//If topology image desired, copy it into user space
		cudaMemcpy(outputTopo,pbaTextures[4],pbaTexSize*pbaTexSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
}




__global__ void kernelSiteFromSkeleton(short2* outputSites, int size)						//Initialize the Voronoi textures from the sites' encoding texture (parameterization)
{																							//REMARK: we interpret 'inputVoro' as a 2D texture, as it's much easier/faster like this
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx<size && ty<size)																	//Careful not to go outside the image..
	{
	  int i = TOID(tx,ty,size);
	  unsigned char param = tex1Dfetch(pbaTexGray,i);										//The sites-param has non-zero (parameter) values precisely on non-boundary points

	  short2& v = outputSites[i];
	  v.x = v.y = MARKER;																	//Non-boundary points are marked as 0 in the parameterization. Here we will compute the FT.
	  if (param)																			//These are points which define the 'sites' to compute the FT/skeleton (thus, have FT==identity)
	  {																						//We could use an if-then-else here, but it's faster with an if-then
	     v.x = tx; v.y = ty;
	  }
	}
}




__global__ void kernelSkelInterpolate(float* output, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx<size && ty<size)																	//Careful not to go outside the image..
	{
  	  int    id     = TOID(tx, ty, size);
	  short2 vid    = tex1Dfetch(pbaTexColor,id);
	  float  T      = sqrtf((tx-vid.x)*(tx-vid.x)+(ty-vid.y)*(ty-vid.y));
	  short2 vid2   = tex1Dfetch(pbaTexColor2,id);
	  float  D      = sqrtf((tx-vid2.x)*(tx-vid2.x)+(ty-vid2.y)*(ty-vid2.y));
	  float  B      = ((D)? min(T/2/D,0.5f):0.5) + 0.5*((T)? max(1-D/T,0.0f):0);
	  output[id]    = B;
	}
}




void skel2DSkeletonDT(float* outputSkelDT,short xm,short ym,short xM,short yM)
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

    cudaBindTexture(0,pbaTexGray,pbaTextures[3]);							//Used to read the resident binary skeleton
    kernelSiteFromSkeleton<<<grid,block>>>(pbaTextures[0],pbaTexSize);		//1. Init pbaTextures[0] with sites on skeleton i.e. from pbaTexGray
	cudaUnbindTexture(pbaTexGray);

	//!!Must first save pbaTextures[1] since we may need it later..
	cudaMemcpy(pbaTextures[5],pbaTextures[1],pbaTexSize*pbaTexSize*sizeof(short2),cudaMemcpyDeviceToDevice);
    skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);		//2. Compute FT of the skeleton into pbaTextures[6]
    cudaMemcpy(pbaTextures[6],pbaTextures[1],pbaTexSize*pbaTexSize*sizeof(short2),cudaMemcpyDeviceToDevice);
    cudaMemcpy(pbaTextures[1],pbaTextures[5],pbaTexSize*pbaTexSize*sizeof(short2),cudaMemcpyDeviceToDevice);

	//Compute interpolation
    cudaBindTexture(0,pbaTexColor,pbaTextures[1]);							// FT of boundary
    cudaBindTexture(0,pbaTexColor2,pbaTextures[6]);							// FT of skeleton
	kernelSkelInterpolate<<<grid,block>>>((float*)pbaTextures[0],pbaTexSize);
	cudaUnbindTexture(pbaTexColor);
	cudaUnbindTexture(pbaTexColor2);
	if (outputSkelDT) cudaMemcpy(outputSkelDT, pbaTextures[0], pbaTexSize * pbaTexSize * sizeof(float), cudaMemcpyDeviceToHost);
}




__device__  bool fill_gc;														//Indicates if a fill-sweep did fill anything or not


__global__ void kernelFill(unsigned char* output, int size, unsigned char bg, unsigned char fg, short xm, short ym, short xM, short yM, bool ne)
{																				//Fill image in pbaTexGray[] from a preset seed; result goes into output[] image.
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx>xm && ty>ym && tx<xM && ty<yM)										//careful not to index outside the image..
	{
	  int    id0 = TOID(tx, ty, size);
	  unsigned char val = tex1Dfetch(pbaTexGray,id0);							//
	  if (val==fg)																//do we have a filled pixel? Then fill all to left/top/up/bottom of it which is background
	  {
	    bool fill = false;
		int id = id0;
		if (ne)																	//fill in north+east direction:
		{
			for(short x=tx+1;x<xM;++x)											//REMARK: here and below, the interesting thing is that it's faster, by about 10-15%, to fill a whole
			{																	//        scanline rather than only until the current block's borders (+1). The reason is that filling a whole
																				//		  scanline decreases the total #sweeps, which seems to be the limiting speed factor
			  if (tex1Dfetch(pbaTexGray,++id)!=bg) break;
			  output[id] = fg; fill = true;
			}

			id = id0;
			for(short y=ty-1;y>ym;--y)
			{
			  if (tex1Dfetch(pbaTexGray,id-=size)!=bg) break;
			  output[id] = fg; fill = true;
			}
		}
		else																	//fill in south+west direction:
		{
			for(short x=tx-1;x>xm;--x)
			{
			  if (tex1Dfetch(pbaTexGray,--id)!=bg) break;
			  output[id] = fg; fill = true;
			}

			id = id0;
			for(short y=ty+1;y<yM;++y)
			{
			  if (tex1Dfetch(pbaTexGray,id+=size)!=bg) break;
			  output[id] = fg; fill = true;
			}
		}

	    if (fill) fill_gc = true;												//if we filled anything, inform caller; we 'gather' this info from a local var into the
																				//global var here, since it's faster than writing the global var in the for loops
	  }
    }
}




__global__ void kernelComputeCoreComponent(unsigned char* output, int size, short2* output_set, int maxpts)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx>=0 && ty>=0 && tx<size && ty<size)									//careful not to index outside the image..
	{
  	  int            id = TOID(tx, ty, size);
	  unsigned char val = tex1Dfetch(pbaTexGray,id);							//
	  if (val==255)																//Skeleton component _not_ in the core-skeleton: Erase it
	     output[id] = 0;
	  else if (val==128)														//Skeleton core component: keep this, and mark it 255 so we're clear what it is
	  {
	     output[id] = 255;
		 int crt_gc = atomicInc(&topo_gc,maxpts);								//REMARK: this serializes (compacts) all detected endpoints in one array.
		 output_set[crt_gc] = make_short2(tx,ty);								//To do this, we use an atomic read-increment-return on a global counter
	  }
	}
																				//Last thread: add zero-marker to the output point-set, so the reader knows how many points are really in there
	if (tx==size-1 && ty==size-1)												//Also reset the global vector counter topo_gc, for the next parallel-run of this function
	{
		topo_gc_last = topo_gc; topo_gc = 0;									//We do this in the last thread so that no one modifies topo_gc from now on.

	}																			//REMARK: this seems to be the only way I can read a __device__ variable back to the CPU
}



int skelft2DFill(unsigned char* outputFill, short sx, short sy, short xm, short ym, short xM, short yM, unsigned char fill_value)
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

    unsigned char background;
	int id = sy * pbaTexSize + sx;
	cudaMemcpy(&background,(unsigned char*)pbaTextures[2]+id,sizeof(unsigned char),cudaMemcpyDeviceToHost); //See which is the value we have to fill from (sx,sy)

	cudaMemset(((unsigned char*)pbaTextures[2])+id,fill_value,sizeof(unsigned char));					//Fill the seed (x,y) on the GPU

	cudaBindTexture(0, pbaTexGray, pbaTextures[2]);														//Used to read the thresholded DT

	int iter=0;
	bool xy = true;																						//Direction of filling for current sweep: either north-east or south-west
																										//This kind of balances the memory-accesses nicely over kernel calls
	for(;;++iter,xy=!xy)																				//Keep filling a sweep at a time until we have no background pixels anymore
	{
	   bool filled = false;																				//Initialize flag: we didn't fill anything in this sweep
	   cudaMemcpyToSymbol(fill_gc,&filled,sizeof(bool),0,cudaMemcpyHostToDevice);						//Pass flag to CUDA
       kernelFill<<<grid, block>>>((unsigned char*)pbaTextures[2],pbaTexSize,background,fill_value,xm,ym,xM,yM,xy);
																										//One fill sweep
	   cudaMemcpyFromSymbol(&filled,fill_gc,sizeof(bool),0,cudaMemcpyDeviceToHost);						//See if we filled anything in this sweep
	   if (!filled) break;																				//Nothing filled? Then we're done, the image didn't change
	}
	cudaUnbindTexture(pbaTexGray);

	if (outputFill) cudaMemcpy(outputFill, (unsigned char*)pbaTextures[2], pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	return iter;																						//Return #iterations done for the fill - useful as a performance measure for caller
}



void skelft2DComputeCoreComponent(unsigned char* outputFill, int* npts, short2* outputPoints)
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

    int maxpts = (npts)? *npts : pbaTexSize*pbaTexSize;												//This is the max # skel-points we are going to return in outputPoints[]

    cudaBindTexture(0, pbaTexGray, outputFill);														//Used to read the input data

    unsigned int zero = 0;
  	cudaMemcpyToSymbol(topo_gc,&zero,sizeof(unsigned int),0,cudaMemcpyHostToDevice);				//Set topo_gc to 0

    kernelComputeCoreComponent<<<grid, block>>>(outputFill, pbaTexSize, pbaTextures[5], maxpts+1);

    cudaUnbindTexture(pbaTexGray);

	if (outputPoints && maxpts)													//If output-point vector desired, copy the end-points, put in pbaTexture[5] as a vector of short2's,
	{																			//into caller space. We copy only 'maxpts' elements, as the user instructed us.
	  unsigned int num_pts;
		cudaMemcpyFromSymbol(&num_pts,topo_gc_last,sizeof(unsigned int),0,cudaMemcpyDeviceToHost);		//Get #points we have detected from the device-var from CUDA
		if (npts && num_pts)																			//Copy the topo-points to caller
		   cudaMemcpy(outputPoints,pbaTextures[5],num_pts*sizeof(short2),cudaMemcpyDeviceToHost);
		if (npts) *npts = num_pts;												//Return #detected topo-points to caller
	}
}
