#ifndef SKEL_CUDA_HPP
#define SKEL_CUDA_HPP

FIELD<float>* computeSkeleton(int level, FIELD<float> *im, bool doReverse, float saliency);
int initialize_skeletonization(FIELD<float>* im);

void clearBuffers();
void deallocateCudaMem();
short* get_current_skel_ft();

FIELD<float>* skelft_to_field();
FIELD<float>* skelImportance_to_field();

coord2D_t getFT(int, int);
float getSiteLabel(coord2D_t);
float getSiteLabelReplace(coord2D_t, int);

#endif