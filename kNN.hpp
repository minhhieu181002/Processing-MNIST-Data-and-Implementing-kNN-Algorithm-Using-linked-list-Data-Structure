#include "main.hpp"

template <typename T>
class List
{
public:
    virtual ~List() = default;
    virtual void push_back(T value) = 0;
    virtual void push_front(T value) = 0;
    virtual void insert(int index, T value) = 0;
    virtual void remove(int index) = 0;
    virtual T &get(int index) const = 0;
    virtual int length() const = 0;
    virtual void clear() = 0;
    virtual void print() const = 0;
    virtual void reverse() = 0;

    virtual List<T> *subList(int start, int end) = 0;
    virtual void printStartToEnd(int start, int end) const = 0;
};

template <typename T>
class Container : public List<T>
{
private:
    class Node
    {
    public:
        T data;
        Node *next;

    public:
        Node(T data, Node *next = nullptr) : data(data), next(next) {}
    };

private:
    Node *head;
    Node *tail;
    int size;

public:
    Container()
    {
        head = tail = nullptr;
        size = 0;
    }
    ~Container()
    {
        this->clear();
    }
    void push_back(T value)
    {
        Node* node = new Node(value);
        if (size == 0) {
            head = tail = node;
        } else {
            tail->next = node;
            tail = node;
        }
        size++;

    }
    
    void push_front(T value)
    {
        Node* node = new Node(value);
        if (size == 0) {
            head = tail = node;
        } else {
            node->next = head;
            head = node;
        }
        size++;
    
    }
    void insert(int index, T value)
    {
        
        if (index < 0 || index > size) {
            // Index out of bounds
            return;
        } else if (index == 0) {
            push_front(value);
        } else if (index == size) {
            push_back(value);
        } else {
            Node* temp = head;
            for (int i = 1; i < index; i++) {
                temp = temp->next;
            }
            Node* node = new Node(value);
            node->next = temp->next;
            temp->next = node;
            size++;
        }
    }
    void remove(int index)
    {
        Node *delNode;
        if (index < 0 || index >= size)
            return;
        // case 1: remove when list has one element
        else if (size == 1)
        {
            delNode = head;
            head = tail = NULL;
        }
        //case 2: remove at head
        else if(index == 0){
            delNode = head;
            head = head->next;
        }
        // case 3: remove at the middle and tail
        else {
            Node *cur = head;
            int i = 1;
            while(i < index){
                cur = cur->next;
                i++;
            }
            delNode = cur->next;
            cur->next = delNode->next;
            if(index == size - 1){
                tail = cur;
            }
        }
        delete delNode;
        size--;
    }
    T &get(int index) const
    {
        if(index < 0 || index >= this->size)  
        {       
                throw std::out_of_range("get(): Out of range");
        }
        Node *cur = head;
        int i = 0;
        while(i < index){
            cur = cur->next;
            i++;
        }
        return cur->data;
    }
    int length() const
    {
        if(size == 0){
            return 0;
        }
        return size;
    }
    void clear()
    {
        while(head!= nullptr){
            Node* temp = head;
            head = head->next;
            delete temp;
        }
        head = tail = nullptr;
        size = 0;
    }
    void print() const
    {
        
        Node *temp = head;
        if (size == 0)
            return;
        for (int i = 0; i < this->size; i++)
        {   
            if(i == this->size-1){
                cout << temp->data;
            }
            else{
                cout << temp->data << " ";
            }

            temp = temp->next;
        }
    }
    void reverse()
    {
        if(size == 0) return;
        Node* prev = nullptr;
        Node* next = nullptr;
        Node* cur = head;
        while(cur!= nullptr){
            next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        tail = head;
        head = prev;
    }
    void printStartToEnd(int start, int end) const
    {
        Node *temp = head;
        for (int i = 0; i < start; i++)
            temp = temp->next;
        for (int i = start; i < end && i < this->size; i++)
        {
            
            if (i == end - 1 || i == this->size-1)
                cout << temp->data;

            else
                cout << temp->data << " ";
            temp = temp->next;

        }
    }
    List<T> *subList(int start, int end)
    {
        if(this->size <= start) return nullptr;
        if(end > this->size){
            end = this->size;
        }
        List<T>* result = new Container<T>();
        Node *cur = head;
        for(int i = 0; i < start; i++){
            cur = cur->next;
        }
        for(int i = start; i < end; i++){
            result->push_back(cur->data);
            cur = cur->next;
        }
        return result;
    }
};
int mostFrequent(int* arr, int n) 
{ 
    int maxcount = 0; 
    int element_having_max_freq = arr[0]; 
    if(n == 1)
        return element_having_max_freq;
    else{
        for (int i = 0; i < n; i++) { 
        int count = 0; 
        for (int j = 0; j < n; j++) { 
            if (arr[i] == arr[j]) 
                count++; 
        } 
  
        if (count > maxcount || (count == maxcount && arr[i] < element_having_max_freq)) { 
            maxcount = count; 
            element_having_max_freq = arr[i]; 
        } 
    } 
  
    return element_having_max_freq;
    }
     
}
class Dataset
{
private:
    List<List<int> *> *data;
    List<string> *columnName;
public:
    Dataset()
    {
        this->columnName = new Container<string>();
        this->data = new Container<List<int> *>();
    }
    ~Dataset()
    {
            for (int i=0;i<data->length();i++){
            data->get(i)->clear();

        }
            for(int i = 0; i< data->length(); i++){
            delete this->data->get(i);
        }
            delete data;
            delete columnName;
    }
    Dataset(const Dataset &other)
    {
        this->columnName = new Container<string>();
        this->data = new Container<List<int>*>();
        int nRows = other.data->length();
        int nCols = other.columnName->length();
        if(other.data->length() == 0){
            return;
        }
        else{
    
            for(int i = 0; i < nCols; i++){
                columnName->push_back(other.columnName->get(i));
            }
            for(int j = 0; j < nRows; j++){
                data->push_back(other.data->get(j)->subList(0, nCols));
            }
        }
    }
    Dataset &operator=(const Dataset &other)
    {
        this->~Dataset();
        if(other.data->length() == 0 ){
            this->columnName = new Container<string>();
            this->data = new Container<List<int>*>();
            return *this;
        }
        int col = other.columnName->length();
        int row = other.data->length();
        this->columnName = new Container<string>();
        this->data = new Container<List<int> *>();
        for (int i = 0; i < row;i++){
            this->data->push_back(other.data->get(i)->subList(0, col));
        }
        this->columnName = other.columnName->subList(0, col);
        return *this;
    }
    List<List<int> *> *getData() const
    {
        return data;
    }
    bool loadFromCSV(const char *fileName)
    {
        
        ifstream csvFile(fileName);
        if (!csvFile)
        {
            return false;
        }

        string lineData;
        int value;

        // Process the header line
        getline(csvFile, lineData);
        stringstream lineStream(lineData);

        string cell;
        while (getline(lineStream, cell, ','))
        {
            columnName->push_back(cell);
        }

        // Read and process each subsequent line
        while (getline(csvFile, lineData))
        {
            stringstream cellStream(lineData);
            auto* row = new Container<int>();

            while (getline(cellStream, cell, ','))
            {
                value = stoi(cell);
                row->push_back(value);
            }

            data->push_back(row);
        }

        csvFile.close();
        return true;
        
    }
    void getShape(int &nRows, int &nCols) const
    {
        nRows = this->data->length();
       
        if(this->data->length() == 0){
            nCols = 0;
        }
        else{
            nCols = this->data->get(0)->length();
        }
        
    }
    void columns() const
    {
        this->columnName->print();
    }
    void printHead(int nRows = 5, int nCols = 5) const
    {
        if(nRows <= 0 || nCols <= 0) return;
        int columnNameLength = 0;
        int dataLength = 0;
        getShape(dataLength, columnNameLength);
        if(nCols > columnNameLength){
            nCols = columnNameLength;
        }
        if(nRows > dataLength){
            nRows = dataLength;
        }       
        if(dataLength == 0){
            return;
        }
        this->columnName->printStartToEnd(0,nCols);

        cout << endl;
        for(int i = 0; i < nRows; i++){
                if(i == nRows - 1){
                    this->data->get(i)->printStartToEnd(0, nCols);
                }
                else{
                    this->data->get(i)->printStartToEnd(0, nCols);
                cout << endl;
                }
        }
    }
    void printTail(int nRows = 5, int nCols = 5) const
    {
        if(nRows <= 0 || nCols <= 0)  return;
        int columnNameLength = 0;
        int dataLength = 0;
        getShape(columnNameLength, dataLength);
        if(nCols > columnName->length()){
            nCols = columnName->length();
        }
        if(nRows > data->length()){
            nRows = data->length();
        }  
        if(this->data->length() == 0){
            return;
        }
        else{

        this->columnName->printStartToEnd(this->columnName->length() - nCols, this->columnName->length());
        cout << endl;
        }
        for(int i = data->length()-nRows; i <data->length() ; i++){

            
            if(i == data->length() - 1){
                this->data->get(i)->printStartToEnd(this->data->get(i)->length() - (nCols), this->data->get(i)->length());

            }
            else{
                this->data->get(i)->printStartToEnd(this->data->get(i)->length() - (nCols), this->data->get(i)->length());
            cout << endl;

            }
        }
    }
    bool drop(int axis = 0, int index = 0, std::string columns = "")
    {
        if(this->data->length() == 0){

            return false;
        }
        else {
        int nRows = 0;
        int nCols = 0;
        getShape(nRows, nCols);
        // std::string delCol = "";
        if (axis == 0 || axis == 1)
        {
            if(axis == 0) {
                if(index >= nRows || index < 0) {
                    return false;
                }
                if(this->data->length()  == 0){
                    return false;
                }
                this->data->remove(index);
                return true;
            }
            else if(axis == 1){
                int cnt = 0;
                
                bool flag = false;
                for (int i = 0; i < nCols;i++){
                    if(this->columnName->get(i) == columns){
                        flag = true;
                        break;
                    }
                    cnt++;
                }
                this->columnName->remove(cnt);
                for (int i = 0; i < nRows; i++)
                {
                    this->data->get(i)->remove(cnt);
                }
                if(!flag){
                    return false;
                }
                else{
                    // cout << delCol << endl;
                    return true;
                }
                
            }
        }
        else{
            return false;
        }
        return false;
        }
    }
    Dataset extract(int startRow = 0, int endRow = -1, int startCol = 0, int endCol = -1) const
    {
        Dataset result;
        int nRows;
        int nCols;
        getShape(nRows, nCols);
        if(endRow < -1 || endCol < -1){
             throw std::out_of_range("get(): Out of range");
        }
        if((endRow > -1 && startRow > endRow) || (endCol > -1 && startCol > endCol)){
            throw std::out_of_range("get(): Out of range");
        }
       if ((endRow == -1))
       {
           endRow = nRows;
        }
        if(endRow > nRows){
            endRow = nRows;
        }
        if(endCol == -1) {
            endCol = nCols;
        }
        if(endCol > nCols){
            endCol = nCols;
        }
        if(startRow >= nRows || startCol >= nCols){
            return result;
        }
        delete result.columnName; 
        result.columnName = columnName->subList(startCol, endCol+1);
        for (int i = startRow; i < endRow+1;++i){
            if(i == nRows){
                break;
            }
            else{
                result.data->push_back(data->get(i)->subList(startCol, endCol+1));
            }
        }
        int resRow = 0;
        int resCol = 0;
        result.getShape(resRow, resCol);
        return result;
    }

    double distanceEuclidean(const List<int> *a, const List<int> *b) const
    {
         double distance = 0.0;
        int len1 = a->length();
        int len2 = b->length();
        int mainLen = 0;
        if(len1 >= len2){
            mainLen = len1;
        }
        else{
            mainLen = len2;
        }
        //Node *cur1 = head;
        double val_A = 0.0;
        double val_B = 0.0;
        for(int i = 0; i < mainLen; i++){
            if(i < len1){
             val_A = a->get(i);
            }
            else{
                val_A = 0;
            } 
            if(i < len2){
                val_B = b->get(i);
            }
            else{
                val_B = 0;
            }
            distance += pow(val_A - val_B, 2);
        }
        return sqrt(distance);
    }
    
    Dataset predict(const Dataset &X_train, const Dataset &Y_train, const int k) const
    {
         Dataset y_pred;
         int y_trainRow = 0;
         int y_trainCol = 0;
         int y_train_length = Y_train.data->length();
        int newY_train[y_train_length];
         int lengthTrain = k;
         if(k > X_train.data->length()){
             throw std::out_of_range("get(): Out of range");
         }
        if(y_train_length < k){
            lengthTrain = y_train_length;
        }
         Y_train.getShape(y_trainRow, y_trainCol);
         if(y_trainRow != 0 && y_trainCol != 0){
             y_pred.columnName->push_back(Y_train.columnName->get(0));
         }
         int lenX_train = X_train.data->length();
         int listDistance[lenX_train];
         int indexOfdistance[lenX_train];
         
             for (int i = 0; i < this->data->length(); i++)
             {
                for (int j = 0;j < lenX_train;j++){
                    indexOfdistance[j] = j;
                }
                 for (int j = 0; j < lenX_train; j++)
                 {
                     listDistance[j] = distanceEuclidean(this->data->get(i), X_train.data->get(j));
                   
                 }
                 for (int u = 0; u < lenX_train - 1; ++u)
                 {
                     for (int e = 0; e < lenX_train - u - 1; ++e)
                     {
                         if (listDistance[e] > listDistance[e + 1])
                         {
                             swap(listDistance[e], listDistance[e + 1]);
                             swap(indexOfdistance[e], indexOfdistance[e + 1]);
                         }
                     }
                 }
                int checkDistance = lengthTrain-1;
                 
                if(listDistance[checkDistance] == listDistance[checkDistance+1] && (indexOfdistance[checkDistance] > indexOfdistance[checkDistance+1])){
                    if( (indexOfdistance[checkDistance] > indexOfdistance[checkDistance+1]) ){
                        swap(indexOfdistance[checkDistance], indexOfdistance[checkDistance+1]);
                    }
                }
                for (int m = 0; m <y_train_length ; m++)
                {
                    newY_train[m] = Y_train.data->get(indexOfdistance[m])->get(0);
                }
                for (int n = 0;n<lengthTrain;n++){
                }
                int most = mostFrequent(newY_train, lengthTrain);
                List<int> *temp = new Container<int>();
                temp->push_back(most);
                y_pred.data->push_back(temp);

            
             }
                
             
        
        return y_pred;

    }
    double score(const Dataset &y_predict) const
    {
        // TODO: implement Task 3
        int yTestLength = this->data->length();
        int yPredictLength = y_predict.data->length();
        double score = 0.0;
        double cnt = 0;
        if(yTestLength == 0 || yPredictLength == 0)
            return -1;
        else if(yTestLength != yPredictLength){
            return -1;
        }
        else{
            for(int i = 0; i < yTestLength; i++){
                if(this->data->get(i)->get(0) == y_predict.data->get(i)->get(0)){
                    cnt++;
                }
            }
            score = cnt/yTestLength;
            return score;
        }
        // return -1;
    }
};

class kNN
{
private:
    int k;
    Dataset X_train;
    Dataset Y_train;
    // You may need to define more
public:
    kNN(int k = 5) : k(k){};
    void fit(const Dataset &X_train, const Dataset &y_train)
    {
        this->X_train = X_train;
        this->Y_train = y_train;
    }
    Dataset predict(const Dataset &X_test)
    {
        return X_test.predict(this->X_train, this->Y_train, this->k);
    }
    double score(const Dataset &y_test, const Dataset &y_pred)
    {
        return y_test.score(y_pred);
    }
};

void train_test_split(Dataset &X, Dataset &Y, double test_size,
                      Dataset &X_train, Dataset &X_test, Dataset &Y_train, Dataset &Y_test)
{
    if (X.getData()->length() != Y.getData()->length() || test_size >= 1 || test_size <= 0)
        return;

    double minDouble = 1.0e-15;
    int nRow = X.getData()->length();
    double rowSplit = nRow * (1 - test_size);

    if (abs(round(rowSplit) - rowSplit) < minDouble * nRow)
        rowSplit = round(rowSplit);

    X_train = X.extract(0, rowSplit - 1, 0, -1);
    Y_train = Y.extract(0, rowSplit - 1, 0, -1);
    X_test = X.extract(rowSplit, -1, 0, -1);
    Y_test = Y.extract(rowSplit, -1, 0, -1);
}

// Please add more or modify as needed