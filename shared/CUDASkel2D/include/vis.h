#pragma once


//	Display:		Visualizes the results of a skeletonization engine SkelEngine
//
//

#include <stdlib.h>
#include <GL/glut.h>



class SkelEngine;


class	Display
{
public:

		  		  Display(int winSize, SkelEngine* skel, float threshold, int argc, char** argv);
					 ~Display();
void 				update();			 

private:

static void mouse(int button, int state, int x, int y);
static void motion(int x, int y);
static void display();
static void keyboard(unsigned char k,int,int);
void			  generateTexture();


static Display* instance;

SkelEngine* skel;														//Skeletonization engine to get results from
float  			threshold;											//Simplification level  of the skeleton

GLuint 			texture;												//Visualization state
int    			winSize;
float  			scale, transX, transY;
bool   			isLeftMouseActive, isRightMouseActive;
int    			oldMouseX, oldMouseY;
int    			show_what;
bool   			tex_interp;
unsigned char* texImage;
};
