#pragma once
#include "include/Image.hpp"
#include "shared/CUDASkel2D/include/skelcomp.h"
#include <opencv2/opencv.hpp>

//Node class used to traverse a skeleton
class node {
private:
    std::vector<coord2D_t> elements;
    coord2D_t getAvgCoordinate() const;

public:
    node() { elements = std::vector <coord2D_t>(0); }
    node(coord2D_t a) { elements.push_back(a); };
    
    void addElement(coord2D_t e) {
        elements.push_back(e);
    }

    //Remove all the elements of the node from the provided set of coordinates
    void removeFromSet(std::set < coord2D_t>& neighbours) const;
    //Set the pixels in the skeleton image to value a
    void setElementsSkel(FIELD<float>* skel, const std::set<coord2D_t>& neighbours, float a) const;

    //checks if any of the neighbours of elements of the node has been marked as a treePoint (a node with more than 1 neighbour)
    bool hasFoundNeighbour(FIELD<float>* skel) const;
    bool isFound(FIELD<float>* skel) const;
    bool isEmpty() const {
        (elements.size() == 0) ? true : false;
    };

    //returns the coordinates of the node. It can have multiple pixel locations so multiple ways to choose the value. Currently just choose the last one added
    coord2D_t getCoordinates() const;
    std::set<coord2D_t> getNeighbours(FIELD<float>* skel) const;
    coord2D_t operator [](int x) const {return elements[x];}
    int size() const {return elements.size(); }

    static float distanceNodes(const node& n1, const node& n2);
    void showNode() const;
};

//Class used to trace a skeleton
class tracer {
private:
    SkelEngine* skelE;
	FIELD<float>* skel; 
	FIELD<float>* imp; 
	FIELD<float>* dt; 
	FIELD<std::vector<cornerData>>& skeletonEndpoints; 
	FIELD<float>& skeletonEndpoints2; 
	vector<coord2D_t>& startingPoints; 
	bool useStartPoints;

	void addSkeletonEndpoint(const node& n);
	void addSkeletonEndpoint(const coord2D_t& p);
	int addTreeNode(std::vector<std::vector<std::tuple<int, int, int>>>& treeNodes, std::vector<std::tuple<int, int, int>>& nodeLabels, int lastTreeNode, int value, int label, const std::pair<int, int>& p);//value: 0=juncture 1=endpoint
	std::vector<node> createNodes(const std::set<coord2D_t>& pixels);
    void traceNode(const node& n, bool firstNode, std::vector<std::vector<std::tuple<int, int, int>>>& treeNodes, std::vector<std::tuple<int, int, int>>& nodeLabels,
        int lastTreeNode, int timeSinceLastNode, float dist, const std::pair<int, int>& startPoint);


public:
    tracer(SkelEngine* nskelE, FIELD<float>* nskel, FIELD<float>* nimp, FIELD<float>* ndt, FIELD<std::vector<cornerData>>& nskeletonEndpoints, FIELD<float>& nskeletonEndpoints2, vector<coord2D_t>& nstartingPoints, bool nuseStartPoints) :
        skelE(nskelE), skel(nskel), imp(nimp), dt(ndt), skeletonEndpoints(nskeletonEndpoints), skeletonEndpoints2(nskeletonEndpoints2), startingPoints(nstartingPoints), useStartPoints(nuseStartPoints) {
    };

    std::vector<cv::Point2f> traceSkeleton();

	
};

//some functions
bool isNeighbour(const coord2D_t& a, const coord2D_t& b);

std::vector<coord2D_t> findLargestComponent(FIELD<float>* binaryImage, FIELD<float>* importance, FIELD<float>* dt);