1. Overview
===========

This directory contains a full copy of the AFMM family of algorithms for computing distance transforms, feature transforms,
skeletons, and related quantities on binary images.

For a detailed description of the algorithms behind, see papers 27 and 85 on www.cs.rug.nl/~alext (in this order). 
The first paper describes the core AFMM algorithm. The second paper describes several important variations thereof,
such as the AFMM Star (corrects some important bugs of AFMM) and AFMM VDT (computes the full feature-transform, not just
the distance transform). Depending on the application you have, you can use different algorithms in this code base,
just by changing a few lines in the main file skeleton.cpp.


2. Compilation
==============

Under Mac OSX, just run make with gcc. For Linux, it may be necessary to change the locations of a few header files,
such as the GLUT headers.

The software is pure C++, and only needs GLUT and OpenGL for the visualization of the results.


3. Running
==========

skeleton <name of pgm file>

You have some examples in the DATA/ directory. Consider only the PGM files (the software cannot read reliably other formats
at this moment). The image you are computing the skeleton of should be black. The background of the image should be white.
The images should be monochrome PGM.

After the program runs, you get one window showing one of the computed datasets. Interaction:

-SPACE in the graphics window to cycle through the computed fields. The name of the computed field is shown in the window
 title bar.
 
- '+' and '-' to increase/decrease the skeleton simplification level. SHIFT and '+', '-' to increase/decrease more rapidly.
 Skeletons are simplified based on the computed importance metric (see paper 27)
 
- 'c' to toggle between rainbow color-coding and grayscale color-coding. 
 
- ESC to exit.


4. Code structure
=================

The entire code works in an 'implicit' image setting. That is, everything is an image, or 2D scalar field. Different fields
are computed for different tasks. The advantage, as compared to a 'geometric' explicit setting is that any kind of 2D
shape can be treated easily, including complex topologies and forms, with the same (simple) code.

The code is structured as a few classes which implement image-processing operations. Below are the most important classes:

FIELD<T> (field.h):  

A 2D image of pixel type T. Core class of most other algorithms. Provides functions for image loading/saving and simple
image arithmetic. Most used instantiation is FIELD<float>, a grayscale image.

FLAGS (flags.h):

Specializes FIELD to an image of labels. Used for the fast marching method (FMM) which is at the core of the skeletonization
and distance transform computatons. A FLAGS object can be created from a FIELD object by simple value thresholding.

FastMarchingMethod (fmm.h):

Implements the well-known FMM method of J. Sethian (see refs of paper 27). The FMM takes as input some binary image, on which
pixels are labeled as outside, inside, and boundary, and visits all inside pixels in strictly increasing order of the distance
to the boundary. As a side-effect, this computes the distance transform of the boundary. However, more operations can be added
to the pixel visiting, which allows implementing a wide range of distance-based image manipulations (see below). 

ModifiedFastMarchingMethod (mfmm.h):

Implements the modified, or augmented, FMM (which I call AFMM). This is the core class for computing skeletons and feature
transforms. The class implements several types of algorithms. To select the desired algorithm, set the desired algorithm using
the setMethod() method of this class. Options are:

ONE_POINT: simply propagates the label of the closest boundary point together with the distance. Kind of outdated, used only for historic reasons.

AVERAGING: propagates the label of the closest boundary point and, if two such labels 'meet', i.e. the current pixel is at equal distance from 
           two boundary points, propagates the average of the two labels. Also outdated and present only for historical reasons. This is actually
		   the precise implementation from paper 27.

AFMM_STAR: propagates the label of the closest boundary point among all neighbors of the currently visited point. It can be shown that this 
		   produces perfect skeletons for arbitrary objects, as opposed to the ONE_POINT and AVERAGING methods (see paper 85). This is the 
		   so-called AFMM Star algorithm, which is my current best skeletonization implementation. Used in the EuroVis'10 paper.
		   		   
AFMM_VDT:  all three methods described above record only ONE closest-boundary-point per interior pixel. This is the so-called one-point feature transform.
           However, there are cases when one wants to find ALL closest-boundary-points. This is the so-called complete feature transform. The AFMM_VDT
		   computes the complete feature transform. Hence, its output is not just a one-label-per-point image, but a multiple-label-per-point image.
		   This is stored as a FIELD<std::multimap<float,int> >, that is an image where at each pixel one has a sorted list of boundary labels, indicating
		   all pixels closest to the current point. The sorting order is in increasing order of distances. These distances will NOT be exactly the same,
		   numerically, given discretization effects on digital images. The AFMM_VDT is used more rarely than the AFMM_STAR, i.e. only in those cases when
		   one really needs to know all feature points of an interior image point.

skeleton.cpp:

The main program. Reads an image, creates a FLAGS labeling, selects a skeletonization method, and computes the distance transform and skeleton.
The skeleton is actually computed in a two-pass process (see for details paper 27).

I also added here, for illustration purposes, the code which computes the so-called smooth interpolation between the skeleton and shape. See main(), and
specifically compute_skel_dt and compute_skel_dt_interpolation. The idea is to compute a smooth distance function which is always 0 on the shape's boundary
(like the distance transform) and always 1 on the skeleton (like 1 - the skeleton's distance transform) and smoothly changes in between. For details,
see paper 28 on my webpage. 

This function is the key idea in computing shaded cushions of arbitrary shapes. Cushions are nothing else than this function interpreted as a luminance
profile. Of course, you can add variations to this, like transparency, texturing, hues, etc.


Remarks:
=========

There are some other utilities used here and there, such as a dynamic array, a stack, etc. Probably one can remove them and just use simpler implementations
readily provided by the C++ STL instead of these.

