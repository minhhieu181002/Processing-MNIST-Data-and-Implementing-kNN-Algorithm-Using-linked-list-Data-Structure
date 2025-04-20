#include "main.hpp"
#include "Dataset.hpp"
/* TODO: Please design your data structure carefully so that you can work with the given dataset
 *       in this assignment. The below structures are just some suggestions.
 */
struct kDTreeNode
{
    vector<int> data;
    int label;
    kDTreeNode *left;
    kDTreeNode *right;
    kDTreeNode(vector<int> data, kDTreeNode *left = nullptr, kDTreeNode *right = nullptr)
    {
        this->data = data;
        this->label = 0;
        this->left = left;
        this->right = right;
    }
    kDTreeNode(vector<int> data, int label, kDTreeNode *left = nullptr, kDTreeNode *right = nullptr)
    {
        this->data = data;
        this->label = label;
        this->left = nullptr;
        this->right = nullptr;
    }

    friend ostream &operator<<(ostream &os, const kDTreeNode &node)
    {
        os << "(";
        for (int i = 0; i < node.data.size(); i++)
        {
            os << node.data[i];
            if (i != node.data.size() - 1)
            {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }
};

class kDTree
{
private:
    int k;
    kDTreeNode *root;
    int count;
private:
        // hàm phụ nếu cần
        kDTreeNode* deepCopyTree(const kDTreeNode *source);
        void deleteTree(kDTreeNode *kDTreeNode);
        int findHeight(kDTreeNode *kDTreeNode) const;
        int findLeaves(kDTreeNode *kDTreeNode) const;
        void inorderTraversalRecursion(kDTreeNode *kDTreeNode) const;
        void preorderTraversalRecursion(kDTreeNode *kDTreeNode) const;
        void postorderTraversalRecursion(kDTreeNode *kDTreeNode) const;
        kDTreeNode* insertRecursion(kDTreeNode *kDTreeNode, const vector<int> &point, unsigned depth);
        bool searchRecursion(kDTreeNode *kDTreeNode, const vector<int> &point, unsigned depth);
        kDTreeNode* findMin(kDTreeNode *kDTreeNode, int d, int depth) const;
        kDTreeNode* removeRecursion(kDTreeNode *kDTreeNode, const vector<int> &point, unsigned depth);
        void merge(vector<vector<int>> &points, int left, int mid, int right, int dim);
        void mergeSort(vector<vector<int>> &points, int left, int right, int dim);

        kDTreeNode* buildTreeRecursion(vector<vector<int>> &pointList, int left, int right, int depth);
        kDTreeNode* buildTreeLabelRecursion(vector<vector<int>> &pointList,vector<int> &label, int depth);

        

        double distance(const vector<int> &point1, const vector<int> &point2);
        void nearestNeighbourRecursion(kDTreeNode *temp, const vector<int> &target, kDTreeNode* &best, int depth);
        void sortBestList(vector<kDTreeNode*> &bestList, const vector<int> &target);
        void kNearestNeighbourRecursion(kDTreeNode *temp, const vector<int> &target, int k, vector<kDTreeNode*> &bestList, int depth);
        void printTree(kDTreeNode *node, int space);
public:
    kDTree(int k = 2);
    ~kDTree();
    const kDTree &operator=(const kDTree &other);
    kDTree(const kDTree &other);
    int nodeCount() const;
    int height() const;
    int leafCount() const;
    void inorderTraversal() const;
    void preorderTraversal() const;
    void postorderTraversal() const;
    void insert(const vector<int> &point);
    void remove(const vector<int> &point);
    bool search(const vector<int> &point);

    void buildTree(const vector<vector<int>> &pointList);

    void buildTreeLabel(vector<vector<int>> &points,vector<int> &labels);

    void nearestNeighbour(const vector<int> &target, kDTreeNode*& best);
    void kNearestNeighbour(const vector<int> &target, int k, vector<kDTreeNode*> &bestList);
    void buildTreeLable(vector<vector<int>> &pointList, vector<int> &label);
    friend class kNN;
};



class kNN
{
private:
    int k;
    Dataset *X_train;
    Dataset *y_train;
    kDTree  kdtree;
public:
    kNN(int k = 5);
    void fit(Dataset &X_train, Dataset &y_train);
    Dataset predict(Dataset &X_test);
    double score(const Dataset &y_test, const Dataset &y_pred);

};

// Please add more or modify as needed
