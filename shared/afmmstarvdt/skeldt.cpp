#include "flags.h"
#include "field.h"
#include "stack.h"
#include "genrl.h"
#include "mfmm.h"
#include <math.h>


void compute_skel_dt(const FIELD<float>* skel,const FIELD<float>* dt,FIELD<float>* skel_dt,float max_dist)
{
   *skel_dt = *skel;

   FLAGS flags(*skel_dt,-0.9,true);

   flags.FIELD<int>::operator=((int)FLAGS::FAR_AWAY);
	
   int dX = skel->dimX(), dY = skel->dimY();
   
   for(int i=0;i<dX;++i)
	for(int j=0;j<dY;++j)
	{
		if (skel->value(i,j)==1) { flags.value(i,j)=FLAGS::ALIVE; skel_dt->value(i,j)= -dt->value(i,j); }				//Skeleton points are ALIVE, and have the skel-DT==0
	}
	
	for(int i=0;i<dX;++i)
		for(int j=0;j<dY;++j)
		{
			for(int ii=i-1;ii<=i+1;++ii)
				for(int jj=j-1;jj<=j+1;++jj)
				{
					if (ii<0 || ii==dX || jj<0 || jj==dY) continue;
					if (flags.value(ii,jj)==FLAGS::ALIVE && flags.value(i,j)==FLAGS::FAR_AWAY)							//Points neighboring skeleton-points are set to NARROWBAND and get skel-DT==1
					{ 
					  flags.value(i,j) = FLAGS::NARROW_BAND; 
					  skel_dt->value(i,j) = -dt->value(i,j);
					}					
				}
		}
	
   FastMarchingMethod fmm(skel_dt,&flags); 		
   int nfail,nextr;	
   fmm.execute(nfail,nextr,0);																							//Evolve from the skeleton outwards until we reach zero. Since we initialized skel-DT with
																														//the -DT, the evolution stops when the inflation has reached the initial shape.
}


void compute_skel_dt_interpolation(const FIELD<float>* dt,const FIELD<float>* skel_dt,FIELD<float>* interp)
{
   int dX = dt->dimX(),dY=dt->dimY();

   for(int j=0;j<dY;j++)
     for(int i=0;i<dX;i++)
     {
	 float T = dt->value(i,j);				//compute smooth interpolation between dt and skel_dt
	 float D = skel_dt->value(i,j);			//(take care that both dt,skel_dt may be 0)
	 float B = 0.5*(((D)? min(T/D,1):1) + ((T)? max(1-D/T,0):0));
	 interp->value(i,j) = pow(B,0.5);
     }
}
