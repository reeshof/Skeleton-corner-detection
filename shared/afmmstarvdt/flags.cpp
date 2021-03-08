#include <stdio.h>
#include <iostream>
#include "flags.h"
#include "genrl.h"
#include "stack.h"




FLAGS::FLAGS(FIELD<float>& f,float low): FIELD<int>(f.dimX(),f.dimY())
//Construct this and adjust f by thresholding f with value low. f will be set to
//0 outside the evolved region, 1 on its boundary, and INFINITY inside.
{									
   float* vptr = f.data(); int*   fptr = data();

   nband = ninside = 0;

   for(float *vend =vptr+f.dimX()*f.dimY();vptr<vend;vptr++,fptr++)
   {
	if ((low>0 && *vptr<low) || (low<0 && *vptr>-low))
  	{ *vptr = 0; *fptr = ALIVE; }
	else
	{ *vptr = MYINFINITY; *fptr = FAR_AWAY; ninside++; }
   }

   for(int j=0;j<dimY();j++)
      for(int i=0;i<dimX();i++)
      if (faraway(i,j))
	 if (alive(i-1,j) || alive(i+1,j) || alive(i,j-1) || alive(i,j+1))
	 { value(i,j) = NARROW_BAND; f.value(i,j) = 1; nband++; }

   ninside -= nband;
   
   computeCenters();
}





FLAGS::FLAGS(FIELD<float>& f,float low,bool dummy): FIELD<int>(f.dimX(),f.dimY())
{									
   float* vptr = f.data(); int* fptr = data();

   nband = ninside = 0;
   for(float *vend =vptr+f.dimX()*f.dimY();vptr<vend;vptr++,fptr++)
   {
	if ((low>0 && *vptr<low) || (low<0 && *vptr>-low))
  	{ *vptr = 1; *fptr = NARROW_BAND; nband++; }
	else
	{ *vptr = MYINFINITY; *fptr = FAR_AWAY; ninside++; }
   }

   ninside -= nband;

   computeCenters();
}




FLAGS::FLAGS(FIELD<float>& f,const FIELD<float>& t,float low): FIELD<int>(f.dimX(),f.dimY())
//Construct this and adjust f by thresholding f with value low. f will be set to
//the signal t outside and on the boundary of the evolved region, and INFINITY inside.
{
   int i,j;

   for(j=0;j<dimY();j++)
      for(i=0;i<dimX();i++)
         value(i,j) = FAR_AWAY;
      
   for(j=0;j<dimY();j++)						//Find boundary and outside of region to evolve.
      for(i=0;i<dimX();i++)						//Set f outside this region to t.
      {									//Set f inside this region to INFINITY.
	if ((low>0 && f.value(i,j)<low) || (low<0 && f.value(i,j)>-low))
	   value(i,j) = ALIVE;
      }

   for(j=0;j<dimY();j++)
      for(i=0;i<dimX();i++)
      if (faraway(i,j))  
         if (alive(i-1,j) || alive(i+1,j) || alive(i,j-1) || alive(i,j+1))
           value(i,j) = NARROW_BAND;
 	 else f.value(i,j) = MYINFINITY;

   computeCenters();
}

void FLAGS::computeCenters()
{
   bcenter[0] = bcenter[1] = 0;
   scenter[0] = scenter[1] = 0;
   int NB=0,NS=0;

   for(int j=0;j<dimY();j++)
      for(int i=0;i<dimX();i++)
	  {
		if (value(i,j) == NARROW_BAND) 
		{
		  bcenter[0] += i; bcenter[1] += j; ++NB;
		}
		
		if (value(i,j)!=ALIVE)
		{
		  scenter[0] += i; scenter[1] += j; ++NS;
		}
   }
   
   bcenter[0] /= NB; bcenter[1] /= NB;
   scenter[0] /= NS; scenter[1] /= NS;   
}



FIELD<int>* FLAGS::tagConnect(int* c_) const	//create field coding points by cluster-number
{
   FIELD<int>* f = new FIELD<int>(dimX(),dimY());
  
   (*f)  = CONN_UNKNOWN;			//initialize f to unknown connectivity

   int c = CONN_UNKNOWN+1;			//start coding clusters from 0	
   for(int j=0;j<f->dimY();j++)			//for every pixel compute its connectivity
      for(int i=0;i<f->dimX();i++)
         if (!alive(i,j) && f->value(i,j)==CONN_UNKNOWN)	
	 {					//if pixel belongs to some cluster, but is not yet labelled,
	    flood_fill(*f,i,j,c);		//do a flood-fill from it with current label
	    c++;				//advance the label
	 }

   if (c_) *c_ = c;				//return #clusters if desired

   return f;
}


int FLAGS::flood_fill(FIELD<int>& f,int i,int j,int val) const
{						//Fills all pixels connected (i.e. with same value)
   STACK<Coord> s(70000);			//with pixel (i,j) of this. The fill value is 'val'.
   s.Push(Coord(i,j));				//Start from pixel (i,j)

   int nx = f.dimX()-1,ny = f.dimY()-1;
   int k  = 0;
 
   while (s.Count())				//For all pixels to fill from:
   {
      Coord c = s.Pop();			//Get pixel to fill
      if (alive(c.i,c.j)) continue;		//If pixel is background continue
      f.value(c.i,c.j) = val; k++;		//Fill current pixel with 'val'

      if (c.i>0  && f.value(c.i-1,c.j)!=val)     s.Push(Coord(c.i-1,c.j));
      if (c.i<nx && f.value(c.i+1,c.j)!=val)     s.Push(Coord(c.i+1,c.j));
      if (c.j<ny && f.value(c.i,c.j+1)!=val)     s.Push(Coord(c.i,c.j+1));
      if (c.j>0  && f.value(c.i,c.j-1)!=val)     s.Push(Coord(c.i,c.j-1));
   }

   return k;
}


void FLAGS::writeRGBCodedPPM(char* fname)
{
   FILE* fp = fopen(fname,"w");
   if (!fp) return;

   const int SIZE = 3000;
   unsigned char buf[SIZE]; int b=0;

   fprintf(fp,"P6 %d %d 255\n",dimX(),dimY());
   for(int* vend=data()+dimX()*dimY(),*vptr=data();vptr<vend;vptr++)
   {
      switch (int(*vptr))
      {
	 case FLAGS::ALIVE:	    buf[b++] = 255; buf[b++] = 0;   buf[b++] = 0;   break;
	 case FLAGS::NARROW_BAND:   buf[b++] = 0;   buf[b++] = 255; buf[b++] = 0;   break;
	 case FLAGS::FAR_AWAY:      buf[b++] = 255; buf[b++] = 255; buf[b++] = 255; break;
	 case FLAGS::EXTREMUM:	    buf[b++] = 0;   buf[b++] = 0;   buf[b++] = 255; break;
      }
      if (b==SIZE) { fwrite(buf,1,SIZE,fp); b = 0; }
   }
   if (b) fwrite(buf,1,b,fp);

   fclose(fp);
}	




