#ifndef FMM_H
#define FMM_H

#include "genrl.h"
#include "darray.h"
#include "field.h"
#include <map>




class 	FastMarchingMethod
	{
	public:	
	
        typedef FIELD<std::multimap<float,Coord>::iterator> POINTERS;
	
			FastMarchingMethod(FIELD<float>*,FLAGS*);	//Ctor
		virtual	~FastMarchingMethod();				//Dtor
		virtual int     
			execute(int&,int&,float=MYINFINITY);		//Do diffusion init'd by ctor, return #iters executed,
									//#failures, #extracted-points. A stop-threshold can be given
									//to stop the marching when the constructed signal reaches it.
									//This is useful e.g. when we reconstruct a curve knowing the
									//distance to it (see FLAGS).	
		void   setVerbose(bool);				//Tell if should display info during execution or not
	protected:
	
		virtual void						//Called by execute() whenever a FAR_AWAY
			add_to_narrowband(int,int,int,int);		//point is first added to narrowband.	
									//First 2 args: point to be added. 
									//Last 2 args: active-point-nb causing addition of above.
									//Subclasses could enhance this if they want
									//to do more stuff when adding a point to
									//the narrowband
	protected:

		void    	      tag_nbs(int,int,int,int,Coord*,int&);
		virtual int           diffuse();
		virtual void          solve(int,int,float,float,float&);		
		
		std::multimap<float,Coord> map;	  	//Narrowband points sorted in ascending distance order
		FIELD<float>*	      f;		//Distance field
		FLAGS*		      flags;		//Help field (FMM)
		POINTERS	      ptrs;		//
		int		      N;		//Max #iteations allowed
		int 		      negd;		//Number of failures in solve2()
		int		      nextr;		//Number of extremum points detected in diffuse()
		float		      maxf;		//Threshold to stop evolution (see execute()).
		bool		      verbose;		//If true, displays info during execution
	};	


inline void FastMarchingMethod::setVerbose(bool b)
{ verbose = b; }

	
#endif				
