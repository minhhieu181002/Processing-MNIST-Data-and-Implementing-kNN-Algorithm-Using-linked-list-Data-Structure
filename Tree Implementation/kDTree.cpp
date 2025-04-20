#include "kDTree.hpp"



// ofstream cout;


kDTree::kDTree(int k)
{
    this->k = k;
    root = nullptr;
    count = 0;
}

kDTree::~kDTree()
{
   deleteTree(root);
}

void kDTree::deleteTree(kDTreeNode *node)
{
    if (node == nullptr)
        return;
    // Recursively delete the left and right subtrees
    deleteTree(node->left);
    deleteTree(node->right);

    // Delete the current node
    delete node;
}
kDTreeNode* kDTree::deepCopyTree(const kDTreeNode *source){
    // Base case
    if(source == nullptr){
        return nullptr;
    }
    // create a new node with the same value as the source
    kDTreeNode *newNode = new kDTreeNode(source->data);
    // Recursively deep copy the left and right subtrees
    newNode->left = deepCopyTree(source->left);
    newNode->right = deepCopyTree(source->right);
    return newNode;
}
const kDTree &kDTree::operator=(const kDTree &other)
{
    // Assignment operator implementation
    this->k = other.k;
    this->count = other.count;
    this->root = this->deepCopyTree(other.root);
    return *this;
}

kDTree::kDTree(const kDTree &other)
{
    // Copy constructor implementation
    this->k = other.k;
    this->count = other.count;
    this->root = this->deepCopyTree(other.root);
}

int kDTree::nodeCount() const
{
    // count number of nodes 
    return this->count;
}
int kDTree::findHeight(kDTreeNode* node) const {
    // Base case
    if(node == nullptr){
        return 0;
    }
    // Recursively find the height of the left and right subtrees
    int leftHeight = findHeight(node->left);
    int rightHeight = findHeight(node->right);
    // Return the height of the current node
    return 1 + max(leftHeight, rightHeight);
}
int kDTree::height() const
{
    // height
    return findHeight(this->root);
}
int kDTree::findLeaves(kDTreeNode* node) const {
    if(node == nullptr){
        return 0;
    }
    if(node->left == nullptr && node->right == nullptr){
        return 1;
    }
    else {
        return findLeaves(node->left) + findLeaves(node->right);
    }
}
int kDTree::leafCount() const
{
    return this->findLeaves(this->root);
}
void kDTree::inorderTraversalRecursion(kDTreeNode *node) const
{
    if (node != nullptr) {
        inorderTraversalRecursion(node->left);
        cout <<*node << " ";
        inorderTraversalRecursion(node->right);
    }
}
void kDTree::inorderTraversal() const
{
    // inorderTraversal implementation
    this->inorderTraversalRecursion(root);
}
void kDTree::preorderTraversalRecursion(kDTreeNode *node) const
{   
    if (node != nullptr) {
        cout <<*node << " ";
        // cout << node->data[0] << "\n";
        preorderTraversalRecursion(node->left);
        preorderTraversalRecursion(node->right);
    }
}
void kDTree::preorderTraversal() const
{
    this->preorderTraversalRecursion(root);
}
void kDTree::postorderTraversalRecursion(kDTreeNode *node) const
{
    if (node != nullptr) {
        postorderTraversalRecursion(node->left);
        postorderTraversalRecursion(node->right);
        cout <<*node << " ";

    }
}
void kDTree::postorderTraversal() const
{
    this->postorderTraversalRecursion(root);
}
kDTreeNode* kDTree::insertRecursion(kDTreeNode *node,const vector<int> &point,unsigned depth){
    // cout << "insertRecursion\n";
    if(node == nullptr){
        node = new kDTreeNode(point);
        return node;
    }
    unsigned curDim = depth % this->k;
    if(point[curDim] < node->data[curDim]){
        node->left = insertRecursion(node->left,point,depth+1);
    }
    else{
        node->right =insertRecursion (node->right,point,depth+1);
    }
    return node;
}
void kDTree::insert(const vector<int> &point)
{
    // cout << "start to insert\n";
    this->root = this->insertRecursion(root,point,0);
    this->count++;
}
bool kDTree::searchRecursion(kDTreeNode *node,const vector<int> &point,unsigned depth){
    if(node == nullptr){
        return false;
    }
    if(node->data == point){
        return true;
    }
    unsigned curDim = depth % this->k;
    if(point[curDim] < node->data[curDim]){
        return searchRecursion(node->left,point,depth+1);
    }
    else{
        return searchRecursion(node->right,point,depth+1);
    }

}
bool kDTree::search(const vector<int> &point)
{
    return this->searchRecursion(root,point,0);
}
kDTreeNode* kDTree::findMin(kDTreeNode* node, int d, int depth) const {
    if (node == nullptr)
        return nullptr;

    int cd = depth % k;

    if (cd == d) {
        if (node->left == nullptr)
            return node;
        return findMin(node->left, d, depth + 1);
    }

    kDTreeNode* left = findMin(node->left, d, depth + 1);
    kDTreeNode* right = findMin(node->right, d, depth + 1);

    kDTreeNode* min = node;

    if (left != nullptr && left->data[d] < min->data[d])
        min = left;
    if (right != nullptr && right->data[d] < min->data[d])
        min = right;

    return min;
}
kDTreeNode* kDTree::removeRecursion(kDTreeNode* node, const vector<int>& point, unsigned depth) {
    if (node == nullptr)
        return nullptr;

    int cd = depth % k;

    if (node->data == point) {
        if (node->right != nullptr) {
            kDTreeNode* min = findMin(node->right, cd, depth + 1);
            node->data = min->data;
            node->right = removeRecursion(node->right, min->data, depth + 1);
        } else if (node->left != nullptr) {
            kDTreeNode* min = findMin(node->left, cd, depth + 1);
            node->data = min->data;
            node->right = removeRecursion(node->left, min->data, depth + 1);
            node->left = nullptr;
        } else {
            delete node;
            this->count--;
            return nullptr;
        }
    } else if (point[cd] < node->data[cd]) {
        node->left = removeRecursion(node->left, point, depth + 1);
    } else {
        node->right = removeRecursion(node->right, point, depth + 1);
    }

    return node;
}
void kDTree::remove(const vector<int> &point)
{
    root = removeRecursion(root, point, 0);
}
void kDTree::merge(vector<vector<int>>& points, int left, int mid, int right, int dim) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<vector<int>> L(n1), R(n2);

    for (i = 0; i < n1; i++)
        L[i] = points[left + i];
    for (j = 0; j < n2; j++)
        R[j] = points[mid + 1 + j];

    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2) {
        if (L[i][dim] <= R[j][dim]) {
            points[k] = L[i];
            i++;
        } else {
            points[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        points[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        points[k] = R[j];
        j++;
        k++;
    }
}

void kDTree::mergeSort( vector<vector<int>>& points, int left, int right, int dim) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(points, left, mid, dim);
        mergeSort(points, mid + 1, right, dim);

        merge(points, left, mid, right, dim);
    }
}
kDTreeNode* kDTree::buildTreeRecursion(vector<vector<int>>& points, int start, int end, int depth) {
    if (start > end)
        return nullptr;

    int cd = depth % k;

    // Sort points according to current dimension using mergeSort
    mergeSort(points, start, end, cd);

    int mid = (start + end) / 2;
    // chose the left most median if the multiple same value
    while (mid > start && points[mid][cd] == points[mid - 1][cd]){
        mid--;
        if(mid == start){
            break;
        }
    }
    kDTreeNode* node = new kDTreeNode(points[mid]);

    node->left = buildTreeRecursion(points, start, mid - 1, depth + 1);
    node->right = buildTreeRecursion(points, mid + 1, end, depth + 1);

    return node;
}
void kDTree::buildTree(const vector<vector<int>> &pointList)
{
    // cout << "start to buildTree\n";
    vector<vector<int>> pointListCopy = pointList;
    this->root = buildTreeRecursion(pointListCopy, 0, pointListCopy.size() - 1, 0);
    this->count = pointListCopy.size();
    // printTreePattern();
}
double kDTree::distance(const vector<int>& point1, const vector<int>& point2) {
    double sum = 0;
    for (int i = 0; i < k; i++) {
        sum += pow(point1[i] - point2[i], 2);
    }
    return sqrt(sum);
}
void kDTree::nearestNeighbourRecursion(kDTreeNode *temp, const vector<int> &target, kDTreeNode* &best, int depth){
    if (temp == nullptr) {
        return;
    }

    int cd = depth % k;
    kDTreeNode* next_node = nullptr;
    kDTreeNode* opposite_node = nullptr;

    if (target[cd] < temp->data[cd]) {
        next_node = temp->left;
        opposite_node = temp->right;
    } else {
        next_node = temp->right;
        opposite_node = temp->left;
    }

    nearestNeighbourRecursion(next_node, target, best, depth + 1);

    if (best == nullptr || distance(target, best->data) > distance(target, temp->data)) {
        best = temp;
    }

    if (opposite_node != nullptr) {
        double d = abs(target[cd] - temp->data[cd]);
        double R = distance(target, best->data);

        if (d < R) {
            nearestNeighbourRecursion(opposite_node, target, best, depth + 1);
        }
    }
}
void kDTree::nearestNeighbour(const vector <int >& target , kDTreeNode*& best)
{
    best = nullptr;
    nearestNeighbourRecursion(root, target, best, 0);
}

// write a function to calculate the distance between the target and the node in BestList
double calculateDistance(const vector<int>& target, kDTreeNode* node) {
    double distance = 0.0;
    for (size_t i = 0; i < target.size(); i++) {
        distance += pow(target[i] - node->data[i], 2);
    }
    return sqrt(distance);
}
kDTreeNode* findFurthestNode(const vector<int>& target, const vector<kDTreeNode*>& bestList) {
    kDTreeNode* furthestNode = nullptr;
    double maxDistance = -1.0;

    for (kDTreeNode* node : bestList) {
        double distance = calculateDistance(target, node);
        if (distance >= maxDistance) {
            maxDistance = distance;
            furthestNode = node;
        }
    }

    return furthestNode;
}

int findFurthestNodeIndex(const vector<int>& target, const vector<kDTreeNode*>& bestList) {
    int furthestNodeIndex = -1;
    double maxDistance = -1.0;

    for (size_t i = 0; i < bestList.size(); i++) {
        double distance = calculateDistance(target, bestList[i]);
        if (distance >= maxDistance) {
            maxDistance = distance;
            furthestNodeIndex = i;
        }
    }

    return furthestNodeIndex;
}
void kDTree::kNearestNeighbourRecursion(kDTreeNode *temp, const vector<int> &target, int k, vector<kDTreeNode*> &bestList, int depth){
    if(temp == nullptr){
        return;
    }
    kDTreeNode *next_node = nullptr;
    kDTreeNode* opposite_node = nullptr;
    int cd = depth % this->k;

    if(target[cd] < temp->data[cd]){
        next_node = temp->left;
        opposite_node = temp->right;
        kNearestNeighbourRecursion(next_node,target,k,bestList,depth+1);
    }
    else{
        next_node = temp->right;
        opposite_node = temp->left;
        kNearestNeighbourRecursion(next_node,target,k,bestList,depth+1);
    }
    double r = distance(target,temp->data);
    if(bestList.size() < k){
        bestList.push_back(temp);  
        kNearestNeighbourRecursion(opposite_node,target,k,bestList,depth+1);
    }
    else{
        double d = abs(target[cd] - temp->data[cd]);
        // calculate the distance between the target and the node that have the largest distance to the target
        kDTreeNode* furthestNode = findFurthestNode(target, bestList);
        double R = calculateDistance(target, furthestNode);
        if(r < R){
            int idx = findFurthestNodeIndex(target, bestList);
            if (furthestNode != nullptr) {
                bestList.erase(bestList.begin() + idx);
            }
            bestList.push_back(temp);
        }
        if(d < R){
            kNearestNeighbourRecursion(opposite_node,target,k,bestList,depth+1);
        }
    }
}

void kDTree::sortBestList(vector<kDTreeNode*> &bestList, const vector<int> &target) {
    int n = bestList.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            double distA = 0.0;
            double distB = 0.0;
            for (size_t k = 0; k < target.size(); ++k) {
                distA += (bestList[j]->data[k] - target[k]) * (bestList[j]->data[k] - target[k]);
                distB += (bestList[j + 1]->data[k] - target[k]) * (bestList[j + 1]->data[k] - target[k]);
            }
            if (distA > distB) {
                swap(bestList[j], bestList[j + 1]);
            }
        }
    }
}
void kDTree::kNearestNeighbour ( const vector <int >& target , int k, vector <kDTreeNode*> & bestList ){
    kNearestNeighbourRecursion(root, target, k, bestList, 0);  
    sortBestList(bestList, target);
}
void kDTree::printTree(kDTreeNode *node, int space) {
    if (node == nullptr) return;

    space += 10;

    printTree(node->right, space);

    cout << endl;
    for (int i = 10; i < space; ++i) cout << " ";
    cout << "(" << node->data[0];
    for (size_t j = 1; j < node->data.size(); ++j) {
        cout << ", " << node->data[j];
    }
    cout << ")" << endl;

    printTree(node->left, space);
}
//######################-BUILD TREE LABEL-###################
kDTreeNode* kDTree::buildTreeLabelRecursion(vector<vector<int>> &pointList,vector<int> &label, int depth) {
    if (pointList.empty()) {
        return nullptr;
    }

    int dim = depth % pointList[0].size();

    // Bubble sort along the dim
    for (int i = 0; i < pointList.size(); i++) {
        for (int j = 0; j < pointList.size() - i - 1; j++) {
            if (pointList[j][dim] > pointList[j + 1][dim]) {
                swap(pointList[j], pointList[j + 1]);
                swap(label[j], label[j + 1]);
            }
        }
    }

    int median = pointList.size() / 2;
    // there are multiple same value, chose the left most median
    while (median > 0 && pointList[median][dim] == pointList[median - 1][dim]) {
        median--;
    }
    kDTreeNode* node = new kDTreeNode(pointList[median], label[median]);

    vector<vector<int>> leftPoints(pointList.begin(), pointList.begin() + median);
    vector<int> leftLabels(label.begin(), label.begin() + median);

    vector<vector<int>> rightPoints(pointList.begin() + median + 1, pointList.end());
    vector<int> rightLabels(label.begin() + median + 1, label.end());

    node->left = buildTreeLabelRecursion(leftPoints, leftLabels, depth + 1);
    node->right = buildTreeLabelRecursion(rightPoints, rightLabels, depth + 1);

    return node;
}

void kDTree::buildTreeLabel(vector<vector<int>> &pointList, vector<int> &label) {
    this->root = buildTreeLabelRecursion(pointList, label, 0);
    this->count = pointList.size();
}
//####################Class kNN####################
kNN::kNN(int k){
    this->k = k;
}
void kNN::fit(Dataset &X_train, Dataset &y_train){
    this->X_train = &X_train;
    this->y_train = &y_train;
    vector<vector<int>> pointList;

    if (X_train.data.size() > 0)
    {
        int dim = X_train.data.front().size();
        kdtree.k = dim;
        vector<int> label;
        // convert X_train.data to pointList,y_train.data (get all elements of the first column) to label
        for (const auto &sublist : X_train.data) {   
            if (!sublist.empty()) {
                vector<int> vec(sublist.begin(), sublist.end());
                pointList.push_back(vec);
            }
        }
        
        for (const auto &row : y_train.data)
        {
            if (!row.empty())
            {
                label.push_back(row.front());
            }
        }

        kdtree.buildTreeLabel(pointList, label);
    }
}
vector<int> calculateFrequency(const vector<kDTreeNode*>& bestList) {
    vector<int> frequency(10, 0); // Adjusted for labels from 0 to 9
    for (const auto &node : bestList) {
        frequency[node->label]++;
    }
    return frequency;
}

// Function to find the most frequent label
int findMostFrequentLabel(const vector<int>& frequency) {
    int maxFrequency = 0;
    int mostFrequentLabel = -1;
    for (int i = 0; i < frequency.size(); i++) {
        if (frequency[i] > maxFrequency) {
            maxFrequency = frequency[i];
            mostFrequentLabel = i;
        }
    }
    return mostFrequentLabel;
}
Dataset kNN::predict(Dataset &X_test){
    Dataset result;
    result.columnName.push_back("label");
    //convert X_test.data to vector 
    vector<vector<int>> x_test_vec;
    for (const auto &sublist : X_test.data) {   
            vector<int> vec(sublist.begin(), sublist.end());
            x_test_vec.push_back(vec);
    }
    for (const auto &target : x_test_vec) {
            vector<kDTreeNode *> bestList;
            kdtree.kNearestNeighbour(target, this->k,bestList);
            vector<int> frequency = calculateFrequency(bestList);
            int mostFrequentLabel = findMostFrequentLabel(frequency);
            result.data.push_back({mostFrequentLabel});
        }
    return result;
}
double kNN::score(const Dataset &y_test, const Dataset &y_pred){
    if(y_test.data.size() != y_pred.data.size()){
        return -1;
    }
    int cnt = 0;
    int n = y_test.data.size();
    auto row1  = y_test.data.begin();
    auto row2 = y_pred.data.begin();    
    while (row1 != y_test.data.end() && row2 != y_pred.data.end())
    {
        if ((*row1).front() == (*row2).front())
        {
            cnt++;
        }
        row1++;
        row2++;
    }
    return cnt * 1.0 / y_test.data.size();
}