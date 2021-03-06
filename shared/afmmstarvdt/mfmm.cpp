#include "mfmm.h"
#include "flags.h"
#include "dqueue.h"
#include "stack.h"
#include <math.h>
#include <iostream>


struct  NewValue {  int i; int j; float value; }; 

DARRAY<Coord> from;


ModifiedFastMarchingMethod::ModifiedFastMarchingMethod(FIELD<float>* f_,FLAGS* flags_,FIELD<float>* count_,
						       FIELD<std::multimap<float,int> >* origs_)
			   :FastMarchingMethod(f_,flags_),count(count_),scan_dir(0),comp_method(AFMM_STAR),
			    label_method(ARC_LENGTH),
			    origs(origs_), dist_tol(sqrt(2.0))
{
}




void ModifiedFastMarchingMethod::init_count()
{
   count->size(*f); 
   *count  = MYINFINITY;			
   *icount = MYINFINITY;									//1. Initialize 'count' by monotonically numbering
   STACK<Coord> s; int i,j; float cc;						//   the points in initial narrowband. This will
															//   produce connected-chains of increasingly numbered 
   if(scan_dir==0)											//   points on the connected narrowband segments
   { for(j=0;j<count->dimY();j++)							//   The search for the narrowband points is done from 		
       for(i=0;i<count->dimX();i++)							//   top-to-bottom in the data, if scan_dir==0, else from
          if (flags->narrowband(i,j)) s.Push(Coord(i,j));	// bottom to top. The found points are put in a container.
   }
   else
     for(j=count->dimY()-1;j>=0;j--)			
       for(i=0;i<count->dimX();i++) 
          if (flags->narrowband(i,j)) s.Push(Coord(i,j));
	
   length = 0; 						//!!!
   int  C = 0;						
	
   const int MYINFINITY_2 = int(MYINFINITY/2);				//2. Now that we collected all narrowband points,
   for(float k=0;s.Count();)								//We start numbering them in the 'count' field.	
   {
      Coord pt = s.Pop();									//Pick narrowband-point, test if already numbered
      if ((*count)(pt.i,pt.j)<MYINFINITY_2) continue;

      int ii,jj,di,dj,ci=0,cj=0; cc = int(MYINFINITY);		//Point pt not numbered
      for(dj=-1;dj<2;dj++)									//Find min-numbered-neighbour of pt
		for(di=-1;di<2;di++)
		{  
			ii=pt.i+di; jj=pt.j+dj; 
			if (ii<0 || ii>=count->dimX() || jj<0 || jj>=count->dimY() || !flags->narrowband(ii,jj)) continue;         
			if ((*count)(ii,jj)>=cc) continue;
			ci = ii; cj = jj; cc = (*count)(ii,jj);
		}
	   
	  float dd = (label_method==ARC_LENGTH)? sqrt(float((pt.i-ci)*(pt.i-ci)+(pt.j-cj)*(pt.j-cj))) : 0;
	  float c  = (cc>MYINFINITY_2)? k+1 : cc+dd; 
      (*count)(pt.i,pt.j)  = c;
      (*icount)(pt.i,pt.j) = C; 
      from[C] = pt;											//Register all boundary-pts in from[]
      if (origs)											//Initialize VDT for boundary-pts, if we compute VDT
		  (*origs)(pt.i,pt.j).insert(std::make_pair(0,C));
      C++;

      if (c>length) length=c;								//Determine length of boundary, used for wraparound-distance-computations 
      if (k<c) k=c;

      for(dj=-1;dj<2;dj++)				
		for(di=-1;di<2;di++)
		{ 
			if (di==0 && dj==0) continue;
			ii=pt.i+di; jj=pt.j+dj; 
			if (ii<0 || ii>=count->dimX() || jj<0 || jj>=count->dimY() || count->value(ii,jj)<MYINFINITY_2 || !flags->narrowband(ii,jj)) 
				continue;  		
			s.Push(Coord(ii,jj));
        }	
   }
}



int ModifiedFastMarchingMethod::execute(int& negd_, int& nextr_, float maxf_)
{
   from.Init(flags->initialContourLength());		//Allocate memory for from[]
   from.SetCount(flags->initialContourLength());	
   icount = new FIELD<int>(f->dimX(),f->dimY());	//Allocate memory for icount[][]

   if (comp_method!=AFMM_VDT) origs = 0;			//If we don't compute the VDT, we don't use origs[] at all
   
   init_count();									//1. Initialize the 'count' field on boundary.
	
   int ret = FastMarchingMethod::execute(negd_,nextr_,maxf_);
													//2. Call inherited fast-marching-method that will
													//   do all the evolution job...
   ::from = from; //!!!  

   from.Init(0);									//Ready with from[], shrink to release memory
   delete icount;									//Ready with icount
   return ret;
}




void ModifiedFastMarchingMethod::add_to_narrowband(int i,int j,int active_i,int active_j)
{
    FastMarchingMethod::add_to_narrowband(i,j,active_i,active_j);

    static std::multimap<float,int> om;				//candidates for origins of (i,j)
    om.erase(om.begin(),om.end());					//we declare this static since it's faster

    if (comp_method!=ONE_POINT)						//AVERAGING/AFMM_VDT COMPUTATION METHODS
    {												//
      float m=MYINFINITY,M=-m,MYINFINITY_2 = MYINFINITY/2; 
      float cv=0; int cc=0,pt_i,pt_j;		        //    
      
      for(int ii=i-1;ii<=i+1;ii++)					//For all known neighbors of crt point (i,j):		
      for(int jj=j-1;jj<=j+1;jj++)
	  if (flags->alive(ii,jj) && count->value(ii,jj)<MYINFINITY_2)
	  {						
	   if (comp_method==AFMM_VDT)
	   {
	     std::multimap<float,int>& o = origs->value(ii,jj);	//All origins of neighbor (ii,jj)
	     for(std::multimap<float,int>::iterator oi=o.begin();oi!=o.end();oi++)
	     {
	       int   O = (*oi).second;
	       Coord c = from[O];						//Find where neighbor came from on the initial boundary
	       int   d = (c.i-i)*(c.i-i)+(c.j-j)*(c.j-j); //Find distance from current point (i,j) to its potential origin
	     
	       bool go=true;							//Collect (i,j)'s potential origins & their distances
	       for(std::multimap<float,int>::iterator it=om.begin();go && it!=om.end();it++)
			 if ((*it).second==O) go=false;			//Take care we don't insert same origin twice
	       if (go)
	         om.insert(std::multimap<float,int>::value_type(sqrt(float(d)),O));
	     }
	   }
	   else											//AVERAGING method:
	   {
	     float v = count->value(ii,jj);
	     if (m>v) m=v; if (M<v) M=v;
	     cv += v; cc++;
	   }
	  }

      if (comp_method==AFMM_VDT)					//If we compute VDT, update origins of (i,j)
      {												//These are simply all points in our potential-list om[]
        float dmin = (*om.begin()).first + dist_tol;//with smallest distance from the crt-point, using dist_tol.
        pt_i = from[(*om.begin()).second].i;		//Since om[] is sorted on distance, this means its first elements
        pt_j = from[(*om.begin()).second].j;		//until we reach (*om.begin()).first + dist_tol
		std::multimap<float,int>& oa = origs->value(i,j); 
        for(std::multimap<float,int>::iterator oi=om.begin();oi!=om.end() && (*oi).first<dmin;oi++)
            oa.insert(*oi); 
      }	
	  else											//If AVERAGING method used, use original AFMM propagation (suboptimal)
	  if (comp_method==AVERAGING)
	  {
		count->value(i,j)  = (M-m<10)? cv/cc : count->value(active_i,active_j);
        icount->value(i,j) = icount->value(active_i,active_j);
	  }
    }						
    else											//ONE-POINT COMPUTATION METHOD (really obsolete)
      count->value(i,j) = count->value(active_i,active_j);
}




int ModifiedFastMarchingMethod::diffuse()
{
    static NewValue newp[20]; 

    //*** 1. FIND MIN-POINT IN NARROWBAND WITH LOWEST VALUE
    int min_i,min_j;
    std::multimap<float,Coord>::iterator it=map.begin();
    if (it==map.end()) return 0;

    min_i  = it->second.i;
    min_j  = it->second.j;
    map.erase(it);					//erase point from 'map', since we'll make it alive in step 2

    //*** 2. MAKE MIN-POINT ALIVE; Its distance value is now frozen, not to be updated any longer
    (*flags)(min_i,min_j) = FLAGS::ALIVE;		
    if ((*f)(min_i,min_j)>=maxf) return 1;		//stop evolution if we reached the user-prescribed threshold
   
    //*** 3. FIND ALL ITS STILL-TO-BE-UPDATED NEIGHBOURS
    Coord nbs[4]; int nn = 0;
    tag_nbs(min_i-1,min_j,min_i,min_j,nbs,nn);
    tag_nbs(min_i+1,min_j,min_i,min_j,nbs,nn);
    tag_nbs(min_i,min_j-1,min_i,min_j,nbs,nn);
    tag_nbs(min_i,min_j+1,min_i,min_j,nbs,nn);

    if (!nn) 						//no more alive-neighbors of point (min_i,min_j) found,
    {								//so it should be an extremum point...
        nextr++;
    }

    //*** 4. UPDATE VALUES OF NEIGHBOURS OF MIN-POINT
    NewValue* nnewp = newp;				//start updating neighbours. Their new values will be saved
    for(nn--;nn>=0;nn--)				//in nnewp[] and pasted back in 'f' at the update end.
    {
	int i = nbs[nn].i, j = nbs[nn].j;

	float vi_1j = f->value(i-1,j),     vijx1 = f->value(i,j+1);
	float vix1j = f->value(i+1,j),     vij_1 = f->value(i,j-1);
	int   fi_1j = flags->value(i-1,j), fijx1 = flags->value(i,j+1);
	int   fix1j = flags->value(i+1,j), fij_1 = flags->value(i,j-1);

	float sol = MYINFINITY; 
	solve(fi_1j,fij_1,vi_1j,vij_1,sol);
	solve(fix1j,fij_1,vix1j,vij_1,sol);
	solve(fi_1j,fijx1,vi_1j,vijx1,sol); 
	solve(fix1j,fijx1,vix1j,vijx1,sol); 

	if (sol < MYINFINITY/2) 
	{ nnewp->i = i; nnewp->j = j; nnewp->value = sol; nnewp++; }
    }

    //***5. Write updated values back in field.
	--nnewp;
    for(NewValue* x=nnewp;x>=newp;--x)						//for all updated neighbours:
    {
       map.erase(ptrs.value(x->i,x->j));					//remove the neighbour's entry from the sorted map...
       std::multimap<float,Coord>::value_type v(x->value,Coord(x->i,x->j));
       ptrs.value(x->i,x->j) = map.insert(v);				//...and insert it back since its field-value changed		
       f->value(x->i,x->j) = x->value;						//update the field too
    }
	
	
	if (comp_method==AFMM_STAR)								//AFMM_STAR method: update count[] and icount[] for all points with new distances
	for(NewValue* x=nnewp;x>=newp;--x)
	{
		int dmin = MYINFINITY; int pt_i,pt_j;	
		for(int ii=x->i-1;ii<=x->i+1;++ii)					//For all known neighbors of crt point (i,j):		
			for(int jj=x->j-1;jj<=x->j+1;++jj)
				if (!flags->faraway(ii,jj) && count->value(ii,jj)<MYINFINITY/2)
				{
					int   V = icount->value(ii,jj);			//Find where (ii,jj) came from on the initial boundary
					const Coord& c = from[V];
					int   d = (c.i-x->i)*(c.i-x->i)+(c.j-x->j)*(c.j-x->j);
					if (d<dmin)					
					{  dmin=d; pt_i=ii; pt_j=jj; }			//Find boundary-point closest to (ii,jj)
				}
	   	
		count->value(x->i,x->j)  = count->value(pt_i,pt_j);
        icount->value(x->i,x->j) = icount->value(pt_i,pt_j);									
	}
	
    return 1;
}


