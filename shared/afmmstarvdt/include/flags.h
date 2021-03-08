#pragma once


// FLAGS:	Implements the flagging of the field's cells for the narrowband representation.
//		Every field cell used in the fast marching method can be:	
//
//		ALIVE:		its field value is known
//		NARROW_BAND:	its value is in the narrow band, i.e. it is still in the updating process
//		FAR_AWAY:	its value is not yet known, i.e. has not yet entered the narrow band
//		EXTREMUM:	this is a special case of known values. Extremum values are known values which
//				are also detected as extremum points of the constructed signal.
//		
//		The ctor of FLAGS constructs this from one or several fields and also adjusts these fields
//		to be evolved by the fast marching method, as follows:
//
//		FLAGS(f,low):	 Evolution starts from and inside curve where f == low, if low > 0.
//				 Evolution starts from and outside curve where f == -low, if low < 0.
//				 The evolved signal is set to 0 outside the evolved region, 1 on the initial curve,
//				 and grows from 1 monotonically in the evolved region.
//
//		FLAGS(f,g,low):  As above, but the evolved signal is set to g on the initial curve and outside 
//				 the evolved region, and INFINITY inside the evolved region. The signal grows then
//				 not from 1, as above, but from g's values on the initial curve.
//

#include "field.h"


class FLAGS : public FIELD<int>
	{
	public:
	
		enum FLAG_TYPE { NARROW_BAND,ALIVE,FAR_AWAY,EXTREMUM };


		     FLAGS(FIELD<float>& f,float low);		//Ctor. See info above. 
		     FLAGS(FIELD<float>& f,			//Ctor. See info above.
			  const FIELD<float>& t,float low);
			 FLAGS(FIELD<float>& f,float low,bool dummy);
			 FLAGS(FIELD<float>& f);

		bool alive(int i,int j) const 		{ return value(i,j)==ALIVE; }
		bool narrowband(int i,int j) const	{ return value(i,j)==NARROW_BAND; }
		bool faraway(int i,int j) const		{ return value(i,j)==FAR_AWAY;  }
		bool extremum(int i,int j) const    { return value(i,j)==EXTREMUM; }
		bool connected(int i,int j) const;
		int  initialContourLength() const 	{ return nband; }
		int  initialObjectSize() const      { return ninside; } 
		const float* boundaryCenter() const	{ return bcenter; }
		const float* surfaceCenter() const  { return scenter; }
	    FIELD<int>*  tagConnect(int* =0) const;	//produce connectivity-field from this, return # produced clusters if desired
		void writeRGBCodedPPM(char*);	//write this as a special color-coded PPM file


	private:
	
	int 	 flood_fill(FIELD<int>&,int i,int j,int val) const;
	void	 computeCenters();
	int		 nband;
	int      ninside;
	float	 bcenter[2];
	float	 scenter[2];
	};


inline bool FLAGS::connected(int min_i,int min_j) const
{
	if (faraway(min_i-1,min_j))   return true;
	if (faraway(min_i  ,min_j-1)) return true;
	if (faraway(min_i+1,min_j))   return true;
	if (faraway(min_i  ,min_j+1)) return true;
	return false;
}

