#include "include/vis.h"
#include "include/skelft.h"
#include "include/skelcomp.h"
#include <iostream>
#include <math.h>


using namespace std;


Display*	Display::instance = 0;						//Singleton object




Display::Display(int winSize_,SkelEngine* skel_,float thr,int argc,char** argv):
        winSize(winSize_),scale(1),skel(skel_),
				transX(0),transY(0),isLeftMouseActive(false),isRightMouseActive(false),oldMouseX(0),oldMouseY(0),
				show_what(0),threshold(thr),tex_interp(false)
{
	  instance = this;

    glutInitWindowSize(winSize, winSize);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA);
    glutInit(&argc, argv);
    glutCreateWindow("2D Salience Skeletons");
    glutDisplayFunc(display);
	  glutMouseFunc(mouse);
    glutMotionFunc(motion);
	  glutKeyboardFunc(keyboard);

		short imgSize = skel->imageSize();
		texImage = new unsigned char[imgSize * imgSize * 3];							// Local buffer to store the data to create GL textures
    glGenTextures(1, &texture);
	  generateTexture();
}

Display::~Display()
{
	delete[] texImage;
}

void Display::display()
{
    // Initialization
    glViewport(0, 0, (GLsizei) instance->winSize, (GLsizei) instance->winSize);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    // Setup projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);

    // Setup modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(instance->scale, instance->scale, 1.0);
    glTranslatef(instance->transX, instance->transY, 0.0);

    // Setting up lights
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
	  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (instance->tex_interp)?GL_LINEAR:GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (instance->tex_interp)?GL_LINEAR:GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // Draw the image
    glBindTexture(GL_TEXTURE_2D, instance->texture);

    glColor3f(1,1,1);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 0.0);
    glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 0.0);
    glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 0.0); glVertex2f(0.0, 1.0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
	  glFinish();

    glutSwapBuffers();
}


void Display::generateTexture()																		  //Encode the skeletonization output
{
	short imgSize = skel->imageSize();

  unsigned int id=0;
  for (int j = 0; j < imgSize; ++j)													// Generate visualization texture
    for (int i = 0; i < imgSize; ++i)
		{
      unsigned char r=0,g=0,b=0;
			if (skel->isBoundary(i,j)>0)													// Boundary (site): mark as red
			{
				 r = 255; g = b = 0;
			}
			else																									// Non-boundary: show FT or skeleton
			{
			  if (show_what==0)																		// Show the FT as grayscale values
			  {
				  r = g = b = (unsigned char)(255*skel->boundaryParam(i,j));
			  }
			  else if (show_what==1)
			  {																					            // Show the skeleton as white
				  if (skel->isSkeleton(i,j)) r=g=b=255;
			  }
			}
			texImage[id++] = r;
      texImage[id++] = g;
      texImage[id++] = b;
		}

    if (show_what==1)
      for(int i=0,ne=skel->numEndpoints();i<ne;i++)							//Show skel-branch endpoints as green
	    {
 		    short2 p = skel->getEndpoint(i);
 	      unsigned int  id = p.y * imgSize + p.x;
 	      texImage[id * 3 + 0] = 0;
 	      texImage[id * 3 + 1] = 255;
 	      texImage[id * 3 + 2] = 0;
	    }

    glBindTexture(GL_TEXTURE_2D, texture);										// Create the texture; the texture-id is already allocated
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imgSize, imgSize, 0, GL_RGB, GL_UNSIGNED_BYTE, texImage);
}







void Display::mouse(int button, int state, int x, int y)
{
    if (state == GLUT_UP)
        switch (button)
    {
        case GLUT_LEFT_BUTTON:
            instance->isLeftMouseActive = false;
            break;
        case GLUT_RIGHT_BUTTON:
            instance->isRightMouseActive = false;
            break;
    }

    if (state == GLUT_DOWN)
    {
        instance->oldMouseX = x;
        instance->oldMouseY = y;

        switch (button)
        {
        case GLUT_LEFT_BUTTON:
            instance->isLeftMouseActive = true;
            break;
        case GLUT_RIGHT_BUTTON:
            instance->isRightMouseActive = true;
            break;
        }
    }
}

void Display::update()
{
	skel->compute(threshold);
	generateTexture();
}


void Display::motion(int x, int y)
{
    if (instance->isLeftMouseActive)
		{
        instance->transX += 2.0 * double(x - instance->oldMouseX) / instance->scale / instance->skel->imageSize();
        instance->transY -= 2.0 * double(y - instance->oldMouseY) / instance->scale / instance->skel->imageSize();
        glutPostRedisplay();
    }
    else if (instance->isRightMouseActive)
		{
        instance->scale -= (y - instance->oldMouseY) * instance->scale / instance->winSize / 2;
        glutPostRedisplay();
    }

    instance->oldMouseX = x; instance->oldMouseY = y;
}



void Display::keyboard(unsigned char k,int,int)
{
  switch (k)
  {
    case ',':  instance->scale *= 0.9; break;
	  case '.':  instance->scale *= 1.1; break;
	  case '-':  instance->threshold-=0.1;
							 if (instance->threshold<1) instance->threshold=1;
				       instance->update(); break;
	  case '=':  instance->threshold+=0.1;
       	       instance->update(); break;
	  case ' ':  instance->show_what++;
		           if (instance->show_what>1) instance->show_what=0;
							 instance->generateTexture(); break;
	  case 't':  instance->tex_interp = !instance->tex_interp; break;
	  case 27:   exit(0);
  }

  glutPostRedisplay();
}
