#include <iostream>
#include "include/skeletonTracer.h"
#include <vector>

#include "include/ImageWriter.hpp"//for coord2D_t

#include <limits>
#include <queue>
#include <stack>

class UF {
    int* id, cnt, * sz, *next;

public:
    // Create an empty union find data structure with N isolated sets.
    UF(int N) {
        cnt = N; id = new int[N]; sz = new int[N]; next = new int[N];
        for (int i = 0; i < N; i++)  id[i] = i, sz[i] = 1, next[i]=i;
    }
    ~UF() { delete[] id; delete[] sz; delete[] next; }

    // Return the id of component corresponding to object p.
    int find(int p) {
        int root = p;
        while (root != id[root])    root = id[root];
        while (p != root) { int newp = id[p]; id[p] = root; p = newp; }
        return root;
    }
    // Replace sets containing x and y with their union.
    void merge(int x, int y) {
        int i = find(x); int j = find(y); if (i == j) return;
        //combine the double linked lists for each union set
        int tempi = next[i];
        next[i] = next[j];
        next[j] = tempi;
        // make smaller root point to larger one
        if (sz[i] < sz[j]) { id[i] = j, sz[j] += sz[i]; }
        else { id[j] = i, sz[i] += sz[j]; }
        cnt--;
    }
    // Are objects x and y in the same set?
    bool connected(int x, int y) { return find(x) == find(y); }
    // Return the number of disjoint sets.
    int count() { return cnt; }

    std::vector<int> getComponent(int p, std::vector<bool>& checked) {
        if (checked[p]) {
            //std::cout << "already did this one" << std::endl;
            return std::vector<int>(0);
        }
        std::vector<int> elements;
        //std::cout << "new cluster: " << std::endl;
        int root = p;
        checked[p] = true;
        elements.push_back(p);

        //std::cout << "next node: " << p << std::endl;
        while (next[p] != root) {
            p = next[p];
            checked[p] = true;
            elements.push_back(p);
            //std::cout << "next node: " <<  p << std::endl;
        }
        return elements;
    }
};

bool isNeighbour(const coord2D_t& a, const coord2D_t& b) {
    if (a == b)
        return false;
    if((abs(a.first - b.first) <= 1) && (abs(a.second - b.second) <= 1))
        return true;
    return false;
}

/***************************************************
    Functions for the node class
***************************************************/

std::set<coord2D_t> node::getNeighbours(FIELD<float>* skel) const{
    std::set<coord2D_t> neighbours;
        for (int i = 0; i < elements.size(); i++) {
            neighboursSet(elements[i].first, elements[i].second,skel, neighbours);
        }
        return neighbours;
}

void node::removeFromSet(std::set < coord2D_t>& neighbours) const{
        for (auto el : elements) {
            neighbours.erase(el);
        }
    }

    //checks if any of the neighbours of elements of the node has been marked as a treePoint (a node with more than 1 neighbour)
bool node::hasFoundNeighbour(FIELD<float>* skel) const {
        for (int i = 0; i < elements.size(); i++) {
            int n[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

            int x = elements[i].first;
            int y = elements[i].second;

            if (x <= 0) { n[0] = 0;        n[3] = 0;        n[5] = 0; }
            if (x >= skel->dimX() - 1) { n[2] = 0;        n[4] = 0;        n[7] = 0; }
            if (y <= 0) { n[0] = 0;        n[1] = 0;        n[2] = 0; }
            if (y >= skel->dimY() - 1) { n[5] = 0;        n[6] = 0;        n[7] = 0; }

            /* For all valid coordinates in the 3x3 region: check for neighbours*/
            if ((n[0] != 0) && (skel->value(x - 1, y - 1) == 2)) { return true; }
            if ((n[1] != 0) && (skel->value(x, y - 1) == 2)) { return true; }
            if ((n[2] != 0) && (skel->value(x + 1, y - 1) == 2)) { return true; }
            if ((n[3] != 0) && (skel->value(x - 1, y) == 2)) { return true; }
            if ((n[4] != 0) && (skel->value(x + 1, y) == 2)) { return true; }
            if ((n[5] != 0) && (skel->value(x - 1, y + 1) == 2)) { return true; }
            if ((n[6] != 0) && (skel->value(x, y + 1) == 2)) { return true; }
            if ((n[7] != 0) && (skel->value(x + 1, y + 1) == 2)) { return true; }
        }
        return false;
    }

bool node::isFound(FIELD<float>* skel) const {
        for (int i = 0; i < elements.size(); i++) {
            int x = elements[i].first;
            int y = elements[i].second;

            float value = skel->value(x, y);

            if (value == 2) {
                return true;
            }
        }
        return false;
    }

void node::showNode() const {
        std::cout << "SHOWING ALL PIXELS FOR THIS NODE, size is: " << elements.size() << std::endl;
        for (int i = 0; i < elements.size(); i++) {
            std::cout << elements[i].first << "|" << elements[i].second << std::endl;
        }
    }

//returns the coordinates of the node. It can have multiple pixel locations so multiple ways to choose the value. Currently just choose the last one added
coord2D_t node::getCoordinates() const {
    if (elements.size() > 0) {
        auto coords = getAvgCoordinate();
        return coords;
    }
    else   return coord2D_t(-1, -1);
}

coord2D_t node::getAvgCoordinate() const {
    assert(elements.size() > 0);
    float x = 0;
    float y = 0;
    for (auto el : elements) {
        x += el.first;
        y += el.second;
    }
    x /= (float)elements.size();
    y /= (float)elements.size();
    return coord2D_t(std::round(x), std::round(y));
}

void node::setElementsSkel(FIELD<float>* skel, const std::set<coord2D_t>& neighbours, float a) const {
    for (int i = 0; i < elements.size(); i++) {
        int nx = elements[i].first;
        int ny = elements[i].second;
        skel->value(nx, ny) = a;
    }
}

float node::distanceNodes(const node& n1, const node& n2){
    auto p1 = n1.getAvgCoordinate();
    auto p2 = n2.getAvgCoordinate();

    float x = p1.first - p2.first; 
    float y = p1.second - p2.second;
    float dist = pow(x, 2) + pow(y, 2);       
    dist = sqrt(dist);

    return dist;
}

/***************************************************
    Functions for the tracer class
***************************************************/

std::vector<node> tracer::createNodes(const std::set<coord2D_t>& pixels) {
    int n = pixels.size();
    auto uf = UF(n);

    std::vector<coord2D_t> pixelVector(n);
    std::copy(pixels.begin(), pixels.end(), pixelVector.begin());

    for (auto i = 0 ; i < n; i++) {
        for (auto j = i; j < n; j++) {
            if (isNeighbour(pixelVector[i], pixelVector[j]))
                uf.merge(i, j);
        }
    }

    std::vector<node> nodes;

    std::vector<bool> checked(n);//used to iterate over components
    for (int i = 0; i < checked.size(); ++i) {
        auto component = uf.getComponent(i, checked);
        if (component.size() > 0) {
            node newNode;
            for (int j = 0; j < component.size(); ++j) {
                newNode.addElement(pixelVector[component[j]]);
            }
            nodes.push_back(newNode);
        }
    }
    return nodes;
}

int tracer::addTreeNode(std::vector<std::vector<std::tuple<int, int, int>>>& treeNodes, std::vector<std::tuple<int, int, int>>& nodeLabels, int lastTreeNode, int value, int label, const std::pair<int,int>& p) {//value: 0=juncture 1=endpoint
    treeNodes.push_back(std::vector<std::tuple<int,int,int>>());//create a new treeNode
    nodeLabels.push_back({ label,p.first,p.second});
    int index = treeNodes.size() - 1;
    treeNodes[index].push_back(std::tuple<int, int, int>(index, lastTreeNode, value));
    treeNodes[lastTreeNode].push_back(std::tuple<int, int, int>(lastTreeNode, index, value));

    return index;
}

void tracer::addSkeletonEndpoint(const node& n) {
    coord2D_t location = n.getCoordinates();
    //first check if its not very close to the boundary (these points are mostly caused by wrong skeletons
    if (location.first < 5 || location.second < 5 || location.first > skeletonEndpoints.dimX() - 5 || location.second > skeletonEndpoints.dimY() -5)
        return;
    skeletonEndpoints2.value(location.first, location.second)++;
}

void tracer::addSkeletonEndpoint(const coord2D_t& p) {
    //first check if its not very close to the boundary (these points are mostly caused by wrong skeletons
    if (p.first < 5 || p.second < 5 || p.first > skeletonEndpoints.dimX() - 5 || p.second > skeletonEndpoints.dimY() - 5)
        return;
    skeletonEndpoints2.value(p.first, p.second)++;
}

struct QObject {
    node n;
    int lastTreeNode;
    int timeSinceLastNode;
    float dist;
};

void tracer::traceNode(const node& newnode, bool firstNode, std::vector<std::vector<std::tuple<int, int, int>>>& treeNodes, std::vector<std::tuple<int, int, int>>& nodeLabels,
        int lastTreeNode_t, int timeSinceLastNode_t, float dist_t, const std::pair<int,int>& startPoint) {
    auto neighbours = newnode.getNeighbours(skel);
    auto newNodes = createNodes(neighbours);
    
    firstNode = false;
    
    nodeLabels.push_back({ -1,startPoint.first,startPoint.second });
    treeNodes.push_back(std::vector<std::tuple<int, int, int>>());//create a new treeNode
    int index = treeNodes.size() - 1;

    firstNode = true;
    newnode.setElementsSkel(skel, neighbours, 2);

    std::stack< QObject> Q;

    for (int i = 0; i < newNodes.size(); i++) {
        newNodes[i].addElement({ startPoint.first, startPoint.second });
        Q.push({ newNodes[i], index, 0, 0 });
    }

    while (Q.size()>0) {
        QObject currentObject = Q.top();
        Q.pop();

        auto neighbours = currentObject.n.getNeighbours(skel);
        currentObject.n.removeFromSet(neighbours);

        auto newNodes = createNodes(neighbours);

        if (newNodes.size() == 1 && newNodes[0].isFound(skel)) {
            addTreeNode(treeNodes, nodeLabels, currentObject.lastTreeNode, currentObject.dist, 1, currentObject.n.getCoordinates());
            currentObject.n.setElementsSkel(skel, neighbours, 500);
        }

        int nUnexploredNodes = 0;
        for (auto& newNode : newNodes)
            if (!newNode.isFound(skel))
                ++nUnexploredNodes;

        currentObject.n.setElementsSkel(skel, neighbours, 2);

        int label = 0;

        if (newNodes.size() > 2) {
            currentObject.lastTreeNode = addTreeNode(treeNodes, nodeLabels, currentObject.lastTreeNode, currentObject.dist, label, currentObject.n.getCoordinates());
            currentObject.timeSinceLastNode = 0;
            currentObject.dist = 0;
        }
        
        for (int i = 0; i < newNodes.size(); i++) {
            
            if (!newNodes[i].isFound(skel)) {
                float newDist = node::distanceNodes(currentObject.n, newNodes[i]);
                 Q.push({ newNodes[i],currentObject.lastTreeNode,currentObject.timeSinceLastNode + 1,currentObject.dist + newDist });
            }
        }
    }
}


std::vector<cv::Point2f> tracer::traceSkeleton() {
    int xDim = skel->dimX();
    int yDim = skel->dimY();

    std::vector<std::vector<std::tuple<int, int, int>>> treeNodes;//from, to, distance
    std::vector<std::tuple<int,int,int>> nodeLabels;//label (1=skeleton 0=intersection), x, y

    if (useStartPoints) {
        for (auto startPoint : startingPoints) {
            node newNode(coord2D_t(startPoint.first, startPoint.second));
            skel->value(startPoint.first, startPoint.second) = 2;//mark is visited
            
            traceNode(newNode, false, treeNodes, nodeLabels, 0, 0, 0,startPoint);
           
        }
    }

    //Getting information from the created skeleton tree, containing endpoints and junctures
    std::vector<float> componentSize;//total length of the skeleton
    std::vector<int> endPointsInComponent;

    std::vector<int> isVisited(nodeLabels.size(), -1);
    auto nextNode = std::find(isVisited.begin(), isVisited.end(), -1);

    while (nextNode != isVisited.end()) {
        componentSize.push_back(0);
        endPointsInComponent.push_back(0);
        int indexOfComponent = componentSize.size() - 1;

        int index = std::distance(isVisited.begin(), nextNode);//get the index location of the newfound node
        isVisited[index] = indexOfComponent;
        float totalDistance = 0;

        if (get<0>(nodeLabels[index]) == 1)
            ++endPointsInComponent[indexOfComponent];

        std::queue<int> exploredNodes;

        for (const auto edge : treeNodes[index]) {
            totalDistance += get<2>(edge);
            if (isVisited[get<1>(edge)] == -1) {
                exploredNodes.push(get<1>(edge));
                isVisited[get<1>(edge)] = indexOfComponent;
                if (get<0>(nodeLabels[get<1>(edge)]) == 1)
                    ++endPointsInComponent[indexOfComponent];
            }
        }

        while (!exploredNodes.empty()) {
            int newNode = exploredNodes.front();
            exploredNodes.pop();

            for (const auto edge : treeNodes[newNode]) {
                totalDistance += get<2>(edge);
                if (isVisited[get<1>(edge)] == -1) {
                    exploredNodes.push(get<1>(edge));
                    isVisited[get<1>(edge)] = indexOfComponent;
                    if (get<0>(nodeLabels[get<1>(edge)]) == 1)
                        ++endPointsInComponent[indexOfComponent];
                }
            }
        }

        componentSize[indexOfComponent] = totalDistance/2.0f;//each edge will be counted twice
        nextNode = std::find(isVisited.begin(), isVisited.end(), -1);
    }

    //Go through all the endpoints
    std::vector<cv::Point2f> endPoints;
    for (int i = 0; i < nodeLabels.size(); ++i) {
        if (get<0>(nodeLabels[i]) == 1) {//marked with 1 are the skeleton endpoints
            int x = get<1>(nodeLabels[i]);
            int y = get<2>(nodeLabels[i]);

            float currentDt = dt->value(x, y);
            float currentImp = imp->value(x, y);
            float currentSaliency = currentImp / currentDt;

            float distance = 0;
            float branchLength = 0;

            auto edges = treeNodes[i];
            if (edges.size() != 1)
                std::cout << "something wrong 1" << "\n";
            int label = get<0>(nodeLabels[get<1>(edges[0])]);
            distance += get<2>(edges[0]);
            //branchLength += get<2>()

            edges = treeNodes[get<1>(edges[0])];
            if(label == -1 && edges.size() == 2) {
                for (const auto e : edges) {
                    if (!(get<1>(e) == i)) {
                        distance += get<2>(e);
                    }
                }
            }
            float currentNormDt;
            if (distance == 0) currentNormDt = 0;
            else  currentNormDt = currentDt / distance;
            float currentObjectSizeNorm = distance / componentSize[isVisited[i]];
            float currentObjectEndpoints = endPointsInComponent[isVisited[i]];

            //std::pair<int,int> adjustedCorner = addSkeletonPoint(x, y, skel, dt, skeletonEndpoints2, -10, skelE).first;
            std::pair<int, int> adjustedCorner = { x,y };

            if (!(adjustedCorner.first < 10 || adjustedCorner.first > skel->dimX() - 10 || adjustedCorner.second < 10 || adjustedCorner.second > skel->dimY() - 10)) {
                cornerData newData{ adjustedCorner.first,adjustedCorner.second,currentSaliency,currentNormDt,currentObjectSizeNorm,currentObjectEndpoints,distance };

                skeletonEndpoints.value(adjustedCorner.first, adjustedCorner.second).push_back(newData);
                ++skeletonEndpoints2.value(adjustedCorner.first, adjustedCorner.second);

                endPoints.push_back({ (float)adjustedCorner.first, (float)adjustedCorner.second });
            }
        }
    }
    return endPoints;
}

std::vector<coord2D_t> findLargestComponent(FIELD<float>* binaryImage, FIELD<float>* importance, FIELD<float>* skel) {
    int yDim = binaryImage->dimY();
    int xDim = binaryImage->dimX();

    int maxImportance = 0;
    std::pair<int, int> maxLocation = std::pair<int, int>(-1, -1);

    std::vector<coord2D_t> startPoints;

    for (int y = 0; y < yDim; ++y) {
        for (int x = 0; x < xDim; ++x) {

            float currentValue = binaryImage->value(x, y);
            if (currentValue >= 0) {//found a new connected component
                maxImportance = 0;
                maxLocation = std::pair<int, int>(-1, -1);

                std::queue<std::pair<unsigned int, unsigned int>> q;
                q.push(std::make_pair(x, y)); //Push the initial position into the queue.
                binaryImage->value(x, y) = -1; //mark as visited
                while (!q.empty()) {//Keep looking at relevant neighbours so long as there is something in the queue. 
                    auto pt = q.front(); //Collect the first entry.
                    q.pop(); //Remove it, we don't want to keep processing the same point. 

                    if (importance->value(pt.first, pt.second) > maxImportance && skel->value(pt.first, pt.second) > 0) {//also has to be a skeleton point
                        maxImportance = importance->value(pt.first, pt.second);
                        maxLocation = pt;
                    }

                    //Now add neighbours if they match our initial point. 
                    if (pt.first > 0 && binaryImage->value(pt.first - 1, pt.second) == currentValue) {
                        q.push(std::make_pair(pt.first - 1, pt.second));
                        binaryImage->value(pt.first - 1, pt.second) = -1; //Replace the value here to avoid pushing the same point twice. 
                    }
                    if (pt.first + 1 < xDim && binaryImage->value(pt.first + 1, pt.second) == currentValue) {
                        q.push(std::make_pair(pt.first + 1, pt.second));
                        binaryImage->value(pt.first + 1, pt.second) = -1;
                    }
                    if (pt.second > 0 && binaryImage->value(pt.first, pt.second - 1) == currentValue) {
                        q.push(std::make_pair(pt.first, pt.second - 1));
                        binaryImage->value(pt.first, pt.second - 1) = -1;
                    }
                    if (pt.second + 1 < yDim && binaryImage->value(pt.first, pt.second + 1) == currentValue) {
                        q.push(std::make_pair(pt.first, pt.second + 1));
                        binaryImage->value(pt.first, pt.second + 1) = -1;
                    }
                }
                coord2D_t newStartPoint = coord2D_t(maxLocation.first, maxLocation.second);
                if (maxImportance > 0)
                    startPoints.push_back(newStartPoint);
            }
        }
    }
    return startPoints;
}