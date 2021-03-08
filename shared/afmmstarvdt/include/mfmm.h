#ifndef MFMM_H
#define MFMM_H

#include "fmm.h"
#include "darray.h"
#include <map>


class 	ModifiedFastMarchingMethod : public FastMarchingMethod
	{
	public:	
		enum METHOD							//Method for computing (propagating) the count-field:
			{								//
			  AFMM_VDT,						// - same as AFMM_STAR, but computes complete VDT (origins too, thus)
			  AFMM_STAR,					// - by finding boundary-point the propagaton came from (better)
			  AVERAGING,					// - by averaging the counts of already computed neighbours (suboptimal)
			  ONE_POINT						// - by simply propagating the count of the one active point (obsolete)
			};  
		
		enum LABELING						//Method for labeling boundary:
			{
			  ARC_LENGTH,					// - arc-length parameterization of all boundary connected components
			  PER_COMPONENT					// - same value assigned for an enture boundary connected component
			};
	
			ModifiedFastMarchingMethod(FIELD<float>*,FLAGS*,FIELD<float>*,FIELD<std::multimap<float,int> >* =0);
						   			//Ctor
		int     execute(int&,int&,float=MYINFINITY);		//Enh inherited to compute the
									//count field
		void    setScanDir(int);				//Set scanning direction. arg can be 0 or 1, thus generating
									//hopefully totally different 'connection-lines'
		void    setMethod(METHOD);				//Set way in which the count-field is computed
		void    setBoundaryLabeling(LABELING);	//Set way in which boundary is labeled
		float	getLength();					//Get boundary-length computed in execute()
		void    setDistTolerance(float);			//Set tolerance for comparing distances when computing VDT-origins
		float   getDistTolerance();
									
	protected:
	
		void	add_to_narrowband(int,int,int,int);		//Enh inherited to update 'count'

	private:

		FIELD<float>*	count;					//Maintains the propagated boundary-parametrization
		FIELD<int>*     icount;					//Maintains the propagation of 'from'
		DARRAY<Coord>   from;					//Given a boundary-label, tells where, on initial boundary, it came from
		FIELD<std::multimap<float,int> >* origs;		//Origin set
		int		scan_dir;						//Used in init_count()	
		METHOD	comp_method;					//Skeletonization method
		LABELING label_method;					//Boundary labeling method
		float	length;							//Boundary-length, used for wraparound-distance computations
		float	dist_tol;						//Tolerance under which two distances are equal. Influences #origins computed by VDT
		
		void	init_count();					//Initializes 'count' from initial boundary		
		int		diffuse();
	};	


inline void ModifiedFastMarchingMethod::setBoundaryLabeling(LABELING l)
{  label_method = l;  }

inline void ModifiedFastMarchingMethod::setScanDir(int d)
{  scan_dir = d;  }

inline void ModifiedFastMarchingMethod::setMethod(METHOD m)
{  comp_method = m;  }
   
inline float ModifiedFastMarchingMethod::getLength()
{  return length;  }

inline void ModifiedFastMarchingMethod::setDistTolerance(float dt)
{  dist_tol = dt;  }

inline float ModifiedFastMarchingMethod::getDistTolerance()
{  return dist_tol;  }


	
#endif				



