#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <string>
#include <iostream>
#include <time.h>
#include "field.h"
#include "flags.h"
#include <GLUT/glut.h>
#include "genrl.h"
#include "io.h"
#include "mfmm.h"

extern DARRAY<Coord> from; //!!hack

using namespace std;

//----------------------------------------------------------------

FIELD<float>* do_one_pass(FIELD<float>*,FLAGS*,FIELD<std::multimap<float,int> >*,int,ModifiedFastMarchingMethod::METHOD,ModifiedFastMarchingMethod::LABELING,float&);
void comp_grad(FLAGS*,FIELD<float>*,FIELD<float>*);
void comp_grad2(FLAGS*,FIELD<float>*,FIELD<float>*,FIELD<float>*);
void postprocess(FIELD<float>*,FIELD<float>*,float,int*);
FIELD<float>* make_num_origs(FIELD<std::multimap<float,int> >*);
FIELD<float>* make_vdt_dist(FIELD<std::multimap<float,int> >* origs);
FIELD<float>* make_dt_diff(FIELD<std::multimap<float,int> >*, FIELD<float>*);
FIELD<float>* make_dt_diff(FIELD<float>*, FIELD<float>*);
FIELD<float>* exact_dt(FIELD<std::multimap<float,int> >* o);
void compute_skel_dt(const FIELD<float>* skel,const FIELD<float>* dt,FIELD<float>* skel_dt,float max_dist);
void compute_skel_dt_interpolation(const FIELD<float>* dt,const FIELD<float>* skel_dt,FIELD<float>* interp);
void compute_centers(FLAGS* flags,FIELD<float>* grad,FIELD<float>* dt);
void key_cb(unsigned char,int,int);
void reshape_cb(int w,int h);
void display_cb();


float         length;
int           iter;
float         sk_lev;
FIELD<float> *skel,*grad,*skdt,*interp,*f;
FIELD<float>* fields[20];
int           f_display;
int           n_fields;
bool          draw_color = true;
int			  maximp_pt[2];
float		  bcenter[2],scenter[2],dtcenter[2];

char* method_name[] = { "Vector DT", "AFMM Star", "AFMM", "AFMM obsolete" };
char* field_name[20];			


inline float distance(int cd)
//Computes distance along boundary in a wrap-around fashion...
{ return min(fabs(float(cd)),fabs(length-fabs(float(cd)))); }


int main(int argc,char* argv[])
{
   glutInit(&argc, argv);
   char inpf[200],*input; string output;
   argc--;argv++;						//Skip program name arg

   float k = -1;	 			    	//Threshold for clusters (default)
   sk_lev  = 2.0;						//Skeleton threshold; must be >1 if we use ARC_LENGTH
   int   twopass = 0;					//Using 2-pass or 1-pass method for boundary treatment 
   ModifiedFastMarchingMethod::METHOD	//Skeletonization/DT method used
         meth = ModifiedFastMarchingMethod::AFMM_STAR;
   ModifiedFastMarchingMethod::LABELING	//Boundary labeling method used
   label_meth = ModifiedFastMarchingMethod::ARC_LENGTH; 

   for(;argc;--argc,++argv)				//Process cmdline args
   {
	if (argv[0][0]!='-') 
		input = argv[0];	
	else 
	{	
		char* opt = argv[0]+1;
		char* val = argv[1];		
	if (opt[0]=='t') 				// -t <threshold>
	{  k = atof(val); argc--;argv++;  }
	else if (opt[0]=='s')				// -s <skeleton_level>
	{  sk_lev = atof(val); argc--;argv++;  }
    else if (opt[0]=='p')				// -p
    {  twopass = 0;  }
	else if (opt[0]=='m')				// -m <method>; most used method is AFMM_STAR
    {
	   meth = (atoi(val)==0)? ModifiedFastMarchingMethod::AFMM_STAR :
	          (atoi(val)==1)? ModifiedFastMarchingMethod::AVERAGING :
	          (atoi(val)==2)? ModifiedFastMarchingMethod::AFMM_VDT :
			                  ModifiedFastMarchingMethod::ONE_POINT;
	   argc--;argv++;
    }
	else if (opt[0]=='o')
	{  output = val; argc--; argv++;  }
	else if (opt[0]=='v')				// -v compute Voronoi diagram of a set of disjoint sites
	{
	   label_meth = ModifiedFastMarchingMethod::PER_COMPONENT;
	   sk_lev     = 0.9;	
	}	   
	}
   }

   char  inp[100]; strcpy(inp,input);

   f = FIELD<float>::read(inp);			//Read scalar field input
   if (!f) { cout<<"Can not open file: "<<inp<<endl; return 1; }

   cout<<"Method: "<<method_name[meth]<<endl;

   FLAGS*		 flags = new FLAGS(*f,k); 	
   FIELD<std::multimap<float,int> >* 
                 origs = (meth==ModifiedFastMarchingMethod::AFMM_VDT)? new FIELD<std::multimap<float,int> >(f->dimX(),f->dimY()) : 0;   
                 grad  = new FIELD<float>(f->dimX(),f->dimY());
                 skel  = new FIELD<float>(f->dimX(),f->dimY());
				 skdt  = new FIELD<float>(f->dimX(),f->dimY());
				interp = new FIELD<float>(f->dimX(),f->dimY());
   clock_t       clk   = clock();
   FIELD<float>* cnt0  = do_one_pass(f,flags,origs,0,meth,label_meth,length);
   FIELD<float>* cnt1  = (twopass)? do_one_pass(f,flags,origs,1,meth,label_meth,length) : 0;
   float cost = (clock()-clk)/float(CLOCKS_PER_SEC);
   if (twopass)  comp_grad2(flags,cnt0,cnt1,grad);
   else          comp_grad(flags,cnt0,grad);  
   postprocess(grad,skel,sk_lev,maximp_pt);
   float fm,fM,fa;
   //f->minmax(fm,fM,fa);
   compute_skel_dt(skel,f,skdt,fM);
   compute_skel_dt_interpolation(f,skdt,interp);
   compute_centers(flags,grad,f);


   cout<<"Iterations: "<<iter<<" in "<<cost<<" seconds"<<endl;
   cout<<"Commands (first select graphics window)"<<endl;
   cout<<"SPACE toggle through computed datasets"<<endl;
   cout<<"+     increase simplification"<<endl;
   cout<<"-     decrease simplification"<<endl;
   cout<<"      (for +,- first select skeleton image)"<<endl;
   cout<<"ESC   quit"<<endl;

   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
   glutInitWindowSize(f->dimX(),f->dimY());
   
   cnt0->write("count.vtk");
   
   f_display=0;					//Gather all fields we're going to visualize
   fields[f_display]=skel; field_name[f_display++] = "Skeleton";
   fields[f_display]=cnt0; field_name[f_display++] = "Boundary Param. U";
   if (twopass)
   {
     fields[f_display]=cnt1; field_name[f_display++] = "Boundary Param. U (2)";
   }
   fields[f_display]=grad; field_name[f_display++] = "Boundary Param. U Derivative";
   fields[f_display]=f;    field_name[f_display++] = "Distance Transform (FMM)";
   fields[f_display]=skdt; field_name[f_display++] = "Skeleton DT";
   fields[f_display]=interp; field_name[f_display++] = "Interpolation";
   if (origs)   
   { 
     FIELD<float> *vdt, *edt;
     fields[f_display]=vdt=make_vdt_dist(origs);  field_name[f_display++] = "Distance Transform (VDT)";
     fields[f_display]=edt=exact_dt(origs);		  field_name[f_display++] = "Exact DT";
     cout<<"Exact-VDT: ";
     fields[f_display]=make_dt_diff(vdt,edt);     field_name[f_display++] = "Exact-VDT Difference"; 
     cout<<"Exact-FMM: ";
     fields[f_display]=make_dt_diff(f,edt);       field_name[f_display++] = "Exact-FMM Difference"; 
     fields[f_display]=make_num_origs(origs);     field_name[f_display++] = "Origin Count";
   }

   n_fields = f_display;
   fields[0]->display(field_name[0]);
   glutKeyboardFunc(key_cb);
   glutMainLoop();
   
   delete f;
   delete flags;
   delete cnt0;
   delete cnt1;
   delete grad;
   return 0;
}


FIELD<float>* do_one_pass(FIELD<float>* fi,FLAGS* flagsi,FIELD<std::multimap<float,int> >* origs,int dir,ModifiedFastMarchingMethod::METHOD meth,
						  ModifiedFastMarchingMethod::LABELING label_meth,float& length)
{
	FIELD<float>*      f = (dir)? new FIELD<float>(*fi) : fi;	//Copy input field (if we're doing the 2nd pass) 
	FLAGS*   	  flags = new FLAGS(*flagsi);					//Copy flags field
	FIELD<float>*  count = new FIELD<float>;
	
	ModifiedFastMarchingMethod fmm(f,flags,count,origs);			//Create fast marching method engine
	fmm.setScanDir(dir);											//Set its various parameters
	fmm.setMethod(meth);
	fmm.setBoundaryLabeling(label_meth);	
	
	int nfail,nextr;			      
	iter   = fmm.execute(nfail,nextr);							//...and execute the skeletonization
	length = fmm.getLength();
	
	delete flags;
	if (f!=fi) delete f;
	return count;
}


void comp_grad2(FLAGS* forig,FIELD<float>* cnt1,FIELD<float>* cnt2,FIELD<float>* grad)
//Gradient computation using 2-pass method (i.e. grad computed out of 2 fields)
{
   int i,j;
   float MYINFINITY_2 = MYINFINITY/2;

   for(j=0;j<grad->dimY();j++)				//Compute grad in a special way, i.e. on a 2-pixel 
     for(i=0;i<grad->dimX();i++) 			//neighbourhood - this ensures pixel-size skeletons!	
     {
		 float ux1 = cnt1->value(i+1,j) - cnt1->value(i,j);
		 float uy1 = cnt1->value(i,j+1) - cnt1->value(i,j);
		 float g1  = max(fabs(ux1),fabs(uy1));
		 float ux2 = cnt2->value(i+1,j) - cnt2->value(i,j);
		 float uy2 = cnt2->value(i,j+1) - cnt2->value(i,j);
		 float g2  = max(fabs(ux2),fabs(uy2));
		 grad->value(i,j) = (g1>MYINFINITY_2 || g2>MYINFINITY_2)? 0:min(g1,g2);
     }
}


void comp_grad(FLAGS* forig,FIELD<float>* cnt,FIELD<float>* grad)
//Gradient computation using 1-pass method (i.e. grad computed out of 1 field)
{
   int i,j;
   for(j=0;j<grad->dimY();j++)				//Compute grad in a special way, i.e. on a 2-pixel 
     for(i=0;i<grad->dimX();i++) 			//neighbourhood - this ensures pixel-size skeletons!	
     {
		 float ux = cnt->value(i+1,j) - cnt->value(i,j);
		 float uy = cnt->value(i,j+1) - cnt->value(i,j);
		 grad->value(i,j) = forig->faraway(i,j)? max(distance(ux),distance(uy)) : 0;
     }
}


void postprocess(FIELD<float>* grad,FIELD<float>* skel,float level,int* maximp_pt)
//Simple reduction of gradient to binary skeleton via thresholding
{
   float maximp = 0;

   for(int j=0;j<grad->dimY();j++)			//Threshold 'grad' to get real skeleton
     for(int i=0;i<grad->dimX();i++)
     {
	   float g = grad->value(i,j);
	   int  sk;
	   
	   if (g==INFINITY) sk = 1; else if (g>level) sk = 1; else sk = 0;
	   skel->value(i,j) = sk;
	   
	   if (sk==1)
	   {
	      if (g>maximp) 
		  { 
		    maximp_pt[0] = i; maximp_pt[1] = j;
		    maximp = g;
		  }
	   }
     } 
}



//---- Less important code below -----------------------------------------------------------

FIELD<float>* exact_dt(FIELD<std::multimap<float,int> >* o)
{
    FIELD<float>* dt = new FIELD<float>(o->dimX(),o->dimY());
    for(int i=0;i<o->dimX();i++)
		for(int j=0;j<o->dimY();j++)
			if (o->value(i,j).size())
			{
				float dmin = 1.0e8;
				for(int b=0;b<from.Count();b++)
				{
					Coord& p = from[b];
					float d = (p.i-i)*(p.i-i) + (p.j-j)*(p.j-j);
					if (d<dmin) dmin=d;
				}
				dmin = 1+sqrt(dmin);
				dt->value(i,j) = dmin;
			}
			else dt->value(i,j) = 0;
	
    return dt;
}

FIELD<float>* make_num_origs(FIELD<std::multimap<float,int> >* origs)
//Create a scalar-field == #origins-per-point, for display purposes
{							
  FIELD<float>* f = new FIELD<float>(origs->dimX(),origs->dimY());

  for(int i=0;i<f->dimX();i++)
    for(int j=0;j<f->dimY();j++)
    {
      f->value(i,j) = origs->value(i,j).size(); if (f->value(i,j)==0) f->value(i,j) = INFINITY;
    }

  return f;
}



FIELD<float>* make_dt_diff(FIELD<std::multimap<float,int> >* origs, FIELD<float>* dt)
//Create a scalar-field == difference between two fields
{							
  FIELD<float>* f = new FIELD<float>(origs->dimX(),origs->dimY());

  float diff_max = -1, diff_sum = 0;
  float diff_min = 1.0e+8;

  for(int i=0;i<f->dimX();i++)
    for(int j=0;j<f->dimY();j++)
    {
      if (origs->value(i,j).size())
      {
	float px=0,py=0,dst=0;
        std::multimap<float,int>& o=origs->value(i,j);
	for(std::multimap<float,int>::iterator it=o.begin();it!=o.end();it++)
	{
	  Coord& p = from[(*it).second];
          px += p.i; py += p.j; dst += (*it).first;
	}
        px /= o.size(); py /= o.size(); dst /= o.size();
	px -= i; py -= j; 

	//f->value(i,j) = fabs(dst+1 - dt->value(i,j));
	//f->value(i,j) = fabs(sqrt(px*px+py*py)+1 - dt->value(i,j));
        f->value(i,j) = fabs((*origs->value(i,j).begin()).first+1 - dt->value(i,j));
	 
	if (diff_max < f->value(i,j)) diff_max = f->value(i,j);
	if (diff_min > f->value(i,j)) diff_min = f->value(i,j);
	diff_sum += f->value(i,j);
      }
      else
	 f->value(i,j) = 0;
    }

  cout<<"diff max: "<<diff_max<<" min: "<<diff_min<<" avg/pix: "<<diff_sum/(f->dimX()*f->dimY())<<endl;

  return f;
}


FIELD<float>* make_dt_diff(FIELD<float>* dt1, FIELD<float>* dt2)
//Create a scalar-field == difference between two fields
{							
  FIELD<float>* f = new FIELD<float>(dt1->dimX(),dt1->dimY());

  float diff_max = -1, diff_sum = 0;
  float diff_min = 1.0e+8; int num=0;

  for(int i=0;i<f->dimX();i++)
    for(int j=0;j<f->dimY();j++)
    {
         float v = fabs(dt1->value(i,j) - dt2->value(i,j));
	 if (diff_max < v) diff_max = v;
	 if (diff_min > v) diff_min = v;
	 diff_sum += v; num++;
	 f->value(i,j) = v;
    }

  cout<<"diff max: "<<diff_max<<" min: "<<diff_min<<" avg/pix: "<<diff_sum/num<<endl;

  return f;
}




FIELD<float>* make_vdt_dist(FIELD<std::multimap<float,int> >* origs)  //Extract DT component from VDT. Note we add 1 to
{								      //the extracted distance. This is so since we want
  FIELD<float>* f = new FIELD<float>(origs->dimX(),origs->dimY());    //to compare this with the FMM, and the FMM starts with 1
								      //on the initial boundary, while the VDT not.
  for(int i=0;i<f->dimX();i++)
    for(int j=0;j<f->dimY();j++)
      f->value(i,j) = (origs->value(i,j).size())? (*origs->value(i,j).begin()).first+1 : 0;

  return f;

}


void compute_centers(FLAGS* flags,FIELD<float>* grad,FIELD<float>* f)
{
   const float* bcenter_ = flags->boundaryCenter();
   const float* scenter_ = flags->surfaceCenter();
   
   bcenter[0] = bcenter_[0]; bcenter[1] = bcenter_[1];
   scenter[0] = scenter_[0]; scenter[1] = scenter_[1];
   
   dtcenter[0] = dtcenter[1] = 0;
   float w = 0;
   
   for(int i=0;i<f->dimX();i++)
    for(int j=0;j<f->dimY();j++)
    {
	  float dt = f->value(i,j);
	  dtcenter[0] += i*dt;
	  dtcenter[1] += j*dt; 
	  w += dt;
	}   
	
	dtcenter[0] /= w; dtcenter[1] /= w;
}


void glutDrawString(const char* s)
{
  while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10,*s++);
}


void key_cb(unsigned char ch,int,int)		      //Handles key presses, displays various fields
{
    bool sk_changed = false;
    switch(ch)
    {
      case '=': sk_lev++; sk_changed=true; break;
      case '-': sk_lev--; sk_changed=true; break;
      case '_': sk_lev-=10; sk_changed=true; break;
      case '+': sk_lev+=10; sk_changed=true; break;
      case ' ': { 
				  int f = (++f_display)%(n_fields+1); 
				  if (f<n_fields)
				     fields[f]->display(0,field_name[f]); 
				  else 
				  {
					 glutReshapeFunc(reshape_cb);
					 glutDisplayFunc(display_cb);  
				  }
				  break; 
				}
      case 27 : exit(0);
      case 'c': draw_color=!draw_color; FIELD<float>::drawColor(draw_color);
    }

    if (sk_changed)
    {
      if (sk_lev<1) sk_lev = 1;
      postprocess(grad,skel,sk_lev,maximp_pt);
	  float fm,fM,fa;
	  f->minmax(fm,fM,fa);
	  compute_skel_dt(skel,f,skdt,fM);
	  compute_skel_dt_interpolation(f,skdt,interp);    
	}
    glutPostRedisplay();
}

void reshape_cb(int w, int h) 
{
 	glViewport(0.0f, 0.0f, (GLfloat)w, (GLfloat)h);
	glMatrixMode(GL_PROJECTION);  
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
}

void display_cb()
{
   int ht = skel->dimY();
   
   glPointSize(3);

   glColor3f(1,1,1);
   glBegin(GL_POINTS);
     glVertex2f(bcenter[0],ht-bcenter[1]);
   glEnd();

   glColor3f(0.5,0.5,0.5);
   glBegin(GL_POINTS);
     glVertex2f(scenter[0],ht-scenter[1]);
   glEnd();


   glColor3f(0,0,0);
   glBegin(GL_POINTS);
     glVertex2f(dtcenter[0],ht-dtcenter[1]);
   glEnd();

   glColor3f(1,0,1);
   glBegin(GL_POINTS);
     glVertex2f(maximp_pt[0],ht-maximp_pt[1]);
   glEnd();
   
   glutSwapBuffers();
}




