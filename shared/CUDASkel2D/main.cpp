#include "include/skelft.h"
#include "include/field.h"
#include "include/vis.h"
#include "include/skelcomp.h"
#include <math.h>
#include <iostream>

using namespace std;




int main(int argc,char **argv)
{
	cout<<"CUDA DT/FT/Salience Skeletonization"<<endl<<endl;
	cout<<"  +,-:     skeleton simplification level"<<endl;
	cout<<"  <,>:     zoom in/out"<<endl;
	cout<<"    t:     texture interpolation on/off"<<endl;
	cout<<"space:     display skeleton / FT"<<endl;
  cout<<" ESC:      quit"<<endl;
  cout<<endl<<endl;
	if (argc<2) return 1;

	FIELD<unsigned char>* input = FIELD<unsigned char>::read(argv[1]);						//1. Read PGM binary image
	int nx = input->dimX();
	int ny = input->dimY();
	short fboSize = skelft2DSize(nx,ny);									                        //   Get size of the image that CUDA will actually use to process our input image

  unsigned char* image = new unsigned char[fboSize*fboSize];                    //   Make image of standard CUDA size
  memset(image,0,fboSize*fboSize*sizeof(unsigned char));					               //2. Create a CUDA-size copy of the input image
                                                                                 //   Also find the extents withing which actual foreground info is in the input image
  short xm=nx,ym=nx,xM=0,yM=0;
	for(int i=0;i<nx;++i)
  	 for(int j=0;j<ny;++j)
		    if (!(*input)(i,j))
		    {
          image[j*fboSize+i] = 255;
		      xm = min(xm,i); ym = min(ym,j);
		      xM = max(xM,i); yM = max(yM,j);
		    }
	xM = nx-1; yM = ny-1;

	delete input;                                                                   //Done with the input, all input data is now in 'image'

  SkelEngine* skel = new SkelEngine();                                            //3. Create skeletonization engine. This will do all the skeleton related work

  skel->initialize(image,fboSize,xm,ym,xM,yM);                                    //4. Initialize skeletonization engine from the input image's area (*m,*M)

  float threshold = 2;												                                    //Initial skeleton threshold

  Display* dpy = new Display(800,skel,threshold,argc,argv);                       //Initialize visualization engine; takes over from now
  dpy->update();                                                                  //Update skel engine and display engine to show the initial results

  glutMainLoop();                                                                 //Control goes to GLUT for keyboard and mouse

	delete dpy;                                                                     //Shutdown
  delete skel;
  return 0;
}
