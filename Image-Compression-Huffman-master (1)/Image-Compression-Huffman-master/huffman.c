#include <stdio.h>
#include <stdlib.h>
#define MAX_TREE_HT 100
struct MinHeapNode
{
    int data;
    unsigned freq;
    struct MinHeapNode *left, *right;
};
struct MinHeap
{
    unsigned size;
    unsigned capacity;
    struct MinHeapNode **array;
};
struct MinHeapNode* newNode(int data, unsigned freq)
{
    struct MinHeapNode* temp =
          (struct MinHeapNode*) malloc(sizeof(struct MinHeapNode));
    temp->left = temp->right = NULL;
    temp->data = data;
    temp->freq = freq;
    return temp;
}
struct MinHeap* createMinHeap(unsigned capacity)
{
    struct MinHeap* minHeap =
         (struct MinHeap*) malloc(sizeof(struct MinHeap));
    minHeap->size = 0;
    minHeap->capacity = capacity;
    minHeap->array =
     (struct MinHeapNode**)malloc(minHeap->capacity * sizeof(struct MinHeapNode*));
    return minHeap;
}
void swapMinHeapNode(struct MinHeapNode** a, struct MinHeapNode** b)
{
    struct MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}


int isSizeOne(struct MinHeap* minHeap)
{
    return (minHeap->size == 1);
}
struct MinHeapNode* extractMin(struct MinHeap* minHeap)
{
    struct MinHeapNode* temp = minHeap->array[0];
    minHeap->array[0] = minHeap->array[minHeap->size - 1];
    --minHeap->size;
    minHeapify(minHeap, 0);
    return temp;
}
void insertMinHeap(struct MinHeap* minHeap, struct MinHeapNode* minHeapNode)
{
    ++minHeap->size;
    int i = minHeap->size - 1;
    while (i && minHeapNode->freq < minHeap->array[(i - 1)/2]->freq)
    {
        minHeap->array[i] = minHeap->array[(i - 1)/2];
        i = (i - 1)/2;
    }
    minHeap->array[i] = minHeapNode;
}


int isLeaf(struct MinHeapNode* root)
{
    return !(root->left) && !(root->right) ;
}
struct MinHeap* createAndBuildMinHeap(int data[], int freq[], int size)
{
    struct MinHeap* minHeap = createMinHeap(size);
    for (int i = 0; i < size; ++i)
        minHeap->array[i] = newNode(data[i], freq[i]);
    minHeap->size = size;
    buildMinHeap(minHeap);
    return minHeap;
}
struct MinHeapNode* buildHuffmanTree(int data[], int freq[], int size)
{
    struct MinHeapNode *left, *right, *top;
    struct MinHeap* minHeap = createAndBuildMinHeap(data, freq, size);
    while (!isSizeOne(minHeap))
    {
        left = extractMin(minHeap);
        right = extractMin(minHeap);
        top = newNode('$', left->freq + right->freq);
        top->left = left;
        top->right = right;
        insertMinHeap(minHeap, top);
    }
    return extractMin(minHeap);
}
void printCodes(struct MinHeapNode* root, int arr[], int top)
{
    if (root->left)
    {
        arr[top] = 0;
        printCodes(root->left, arr, top + 1);
    }
    if (root->right)
    {
        arr[top] = 1;
        printCodes(root->right, arr, top + 1);
    }
    if (isLeaf(root))
    {
        printf("%d: ", root->data);
        printArr(arr, top);
    }
}
void HuffmanCodes(int data[], int freq[], int size)
{
   struct MinHeapNode* root = buildHuffmanTree(data, freq, size);
   int arr[MAX_TREE_HT], top = 0;
   printCodes(root, arr, top);
}

void printArr(int arr[], int n)
{
    int i;
    for (i = 0; i < n; ++i)
        printf("%d", arr[i]);
    printf("\n");
}

void decodeHuffman(struct MinHeapNode* root, const char* compressedFilename) {
    FILE* compressedFile = fopen(compressedFilename, "r");
    FILE* outputFile = fopen("decompressed.txt", "w");
    struct MinHeapNode* current = root;
    char bit;
    while ((bit = fgetc(compressedFile)) != EOF) {
        if (bit == '0') current = current->left;
        else if (bit == '1') current = current->right;

        if (isLeaf(current)) {
            fprintf(outputFile, "%d\n", current->data);
            current = root;
        }
    }
    fclose(compressedFile);
    fclose(outputFile);
}


int main()
{
    int r,i=0;
    int arr1[1000],freq1[1000];
    FILE *fp;
    fp=fopen("test2.txt","r");
    r=fscanf(fp,"%d,%d\n",&arr1[i],&freq1[i]);
    while(r!=EOF){
        i++;
        r=fscanf(fp,"%d,%d\n",&arr1[i],&freq1[i]);
    }
    int k;
    int arr[i],freq[i];
    for(k=0;k<i;k++)
    	arr[k]=arr1[k];
   	for(k=0;k<i;k++)
   		freq[k]=freq1[k];
    int size = sizeof(arr)/sizeof(arr[0]);
    HuffmanCodes(arr, freq, size);
    return 0;
}

































































void buildMinHeap(struct MinHeap* minHeap)
{
    int n = minHeap->size - 1;
    int i;
    for (i = (n - 1) / 2; i >= 0; --i)
        minHeapify(minHeap, i);
}

void minHeapify(struct MinHeap* minHeap, int idx)
{
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    if (left < minHeap->size &&
        minHeap->array[left]->freq < minHeap->array[smallest]->freq)
      smallest = left;
    if (right < minHeap->size &&
        minHeap->array[right]->freq < minHeap->array[smallest]->freq)
      smallest = right;
    if (smallest != idx)
    {
        swapMinHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);
        minHeapify(minHeap, smallest);
    }
}
