#include <iostream>
#include <fstream>
#include <memory>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <complex>
#include <vector>
#include <set>
#include <omp.h>
#include <mkl.h>
#include <cstring>

// Utility functions

void matrixXmatrixMKLfull(const std::complex<float> *matrixA,
                          const std::complex<float> *matrixB,
                          std::complex<float> *result,
                          const int &rowSize, const int &colSize,
                          const int &midSize,
                          const int &lda, const int &ldb, const int &ldc,
                          char transA, char transB)
{
    if (rowSize <= 0 || colSize <= 0 || midSize <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0)
        return;
    printf("\t\t\trowSize: %d, colSize: %d, midSize: %d, lda: %d, ldb: %d, ldc: %d\n", rowSize, colSize, midSize, lda, ldb, ldc);

    const std::complex<float> alpha(1, 0);
    const std::complex<float> beta(0, 0);
    int least_lda = (transA == 'N') ? rowSize : midSize;
    int true_k = (transA == 'N') ? midSize : rowSize;
    int least_ldb = (transB == 'N') ? midSize : colSize;
    int true_n = (transB == 'N') ? colSize : midSize;

    cgemm(&transA, &transB, &least_lda, &true_n, &true_k, (MKL_Complex8 *)&alpha,
          (MKL_Complex8 *)matrixA, &lda, (MKL_Complex8 *)matrixB, &ldb,
          (MKL_Complex8 *)&beta, (MKL_Complex8 *)result, &ldc);
}

double matrixXmatrixMKLPT(const std::complex<float> *matrixA,
                          const std::complex<float> *matrixB,
                          std::complex<float> *result,
                          const int &rowSize, const int &colSize,
                          const int &midSize,
                          const int &lda, const int &ldb, const int &ldc,
                          char transA, char transB, int numRepeat)
{
    if (rowSize <= 0 || colSize <= 0 || midSize <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0)
        return 0.0;

    const std::complex<float> alpha(1, 0);
    const std::complex<float> beta(0, 0);
    int least_lda = rowSize;
    int true_k = midSize;
    int true_n = colSize;

    double time_st = dsecnd();
    for (auto i = 0; i < numRepeat; i++)
    {
        cgemm(&transA, &transB, &least_lda, &true_n, &true_k, (MKL_Complex8 *)&alpha,
              (MKL_Complex8 *)matrixA, &lda, (MKL_Complex8 *)matrixB, &ldb,
              (MKL_Complex8 *)&beta, (MKL_Complex8 *)result, &ldc);
    }
    double time_end = dsecnd();
    double time_avg = (time_end - time_st) / numRepeat;
    double gflop = (2.0 * least_lda * true_n * true_k) * 1E-9;

    return gflop / time_avg;
}

void vectorPlusEqualVector_MKL(std::complex<float> *v1, const int &size, const std::complex<float> *v2)
{
    int incx = 1;
    int incy = 1;
    float alpha = +1;
    int complexSize = size * 2;

    saxpy(&complexSize, &alpha, (float *)v2, &incx, (float *)v1, &incy);
}

std::vector<std::string> SplitStringOnWhiteSpace(const std::string &input)
{
    std::vector<std::string> result;
    std::istringstream istr(input);
    while (istr)
    {
        std::string data;
        istr >> data;
        if (!data.empty())
            result.push_back(data);
    }
    return result;
}

// Data Structures

class TreeNode
{
public:
    TreeNode(int level, int indexInLevel) : m_level(level), m_leftChild(nullptr),
                                            m_rightChild(nullptr), m_indexInLevel(indexInLevel)
    {
        ;
    }
    virtual ~TreeNode()
    {
        if (m_leftChild)
        {
            delete (m_leftChild);
            m_leftChild = nullptr;
        }
        if (m_rightChild)
        {
            delete (m_rightChild);
            m_rightChild = nullptr;
        }
    }
    void grow(int targetLevel)
    {
        if (m_level < targetLevel)
        {
            m_leftChild = new TreeNode(m_level + 1, 2 * m_indexInLevel);
            m_rightChild = new TreeNode(m_level + 1, 2 * m_indexInLevel + 1);
            m_leftChild->grow(targetLevel);
            m_rightChild->grow(targetLevel);
        }
        else
            m_val = std::sqrt(2e-8 * (m_indexInLevel + 1));
        return;
    }

    std::complex<float> getVal(int index, int maxLevel) const
    {
        if (m_level == maxLevel)
            return std::complex<float>(m_val, 2 * m_val);

        int bitShifted = index;
        for (int ii = m_level; ii < maxLevel; ++ii)
            bitShifted /= 2;

        if (bitShifted == 2 * m_indexInLevel)
            return m_leftChild->getVal(index, maxLevel);
        else
            return m_rightChild->getVal(index, maxLevel);
    }

    void printTree(int depth = 0) const
    {
        // Print the current node with indentation based on depth
        std::cout << std::string(depth * 2, ' ') << "Level: " << m_level
                  << ", Index: " << m_indexInLevel
                  << ", Value: " << m_val << std::endl;

        if (m_leftChild)
            m_leftChild->printTree(depth + 1);
        if (m_rightChild)
            m_rightChild->printTree(depth + 1);
    }

private:
    int m_level;
    int m_indexInLevel;
    float m_val;
    TreeNode *m_leftChild;
    TreeNode *m_rightChild;
};

class Task
{
public:
    Task(const int &m, const int &n, const int &r, const int &id) : m_M(m), m_N(n), m_R(r), m_id(id)
    {
        m_cost = (r == 0) ? (m * n) : (r * (m + n));
        m_size = m;
        m_size *= n;
    }

    const int &M() const { return m_M; }
    const int &N() const { return m_N; }
    const int &R() const { return m_R; }
    bool operator<(const Task &other) const
    {
        return m_cost < other.m_cost;
    }
    const long int &size() const { return m_size; }
    int d_createBasisMap(int index, bool isSource, int maxSize)
    {
        long long hash = (2 * m_id + 23);
        hash *= index;
        hash *= index;

        long long t2 = m_id;
        t2 *= m_id;
        t2 += 45;
        t2 *= index;

        hash += t2;
        hash += 43;
        if (isSource)
        {
            hash *= 100;
            hash += 1423;
        }
        return (int)(hash % maxSize);
    }
    std::complex<float> d_matrixEntry(int rIndex, int cIndex,
                                      const TreeNode *tHead,
                                      int maxLevel, bool local)
    {
        std::complex<float> result;
        for (int ii = 0; ii < 10; ii++)
        {
            long long hash = (cIndex + 1001) * rIndex + ii * 500;
            if (!local)
            {
                hash *= 100;
                hash += 1423;
            }
            long max = (long)std::pow(2, maxLevel);
            int tIndex = (int)(hash % max);
            std::complex<float> val = tHead->getVal(tIndex, maxLevel);
            if (std::abs(val) > 1e-31)
                val /= std::sqrt(std::abs(val));
            result += val;
        }
        return result;
    }

    void matmat(const std::complex<float> *globalB,
                std::complex<float> *globalCPerBin,
                std::complex<float> *localB,
                std::complex<float> *localC,
                std::complex<float> *localMat,
                int nRHS,
                int matrixSize)
    {
        printf("\t\tMatrix multiplication for task %d\n", m_id);
        for (int j = 0; j < nRHS; ++j)
            for (int i = 0; i < m_N; ++i)
            {
                localB[i + m_N * j] = globalB[m_srcBasisMap[i] + matrixSize * j];
            }
        if (m_R == 0)
        {
            matrixXmatrixMKLfull(&m_pDense[0], &localB[0], &localC[0], m_M, nRHS, m_N, m_M, m_N, m_M, 'N', 'N');
        }
        else
        {
            matrixXmatrixMKLfull(&m_Rmat[0], &localB[0], &localMat[0], m_R, nRHS, m_N, m_R, m_N, m_R, 'N', 'N');
            matrixXmatrixMKLfull(&m_Qmat[0], &localMat[0], &localC[0], m_M, nRHS, m_R, m_M, m_R, m_M, 'N', 'N');
        }
        for (int j = 0; j < nRHS; ++j)
            for (int i = 0; i < m_M; ++i)
            {
                globalCPerBin[m_sinkBasisMap[i] + matrixSize * j] += localC[i + m_M * j];
            }
    }

    void setUp(int matrixSize, const TreeNode *tHead, const int &tLevel)
    {
        m_srcBasisMap.resize(m_N);
      //  printf("\t\t\tSetting up %d source basis maps\n", m_N);
        for (int ii = 0; ii < m_N; ++ii)
        {
            m_srcBasisMap[ii] = d_createBasisMap(ii, true, matrixSize);
            //printf("Source basis map %d: %d\n", ii, m_srcBasisMap[ii]);
        }

      //  printf("\t\t\tSetting up %d sink basis maps\n", m_M);
        m_sinkBasisMap.resize(m_M);
        for (int ii = 0; ii < m_M; ++ii)
        {
            m_sinkBasisMap[ii] = d_createBasisMap(ii, false, matrixSize);
            //printf("Sink basis map %d: %d\n", ii, m_sinkBasisMap[ii]);
        }

        if (m_R == 0)
        {
            m_pDense = (std::complex<float> *)mkl_malloc(m_M * m_N * sizeof(std::complex<float>), 128);
            printf("\t\t\tSetting up dense matrix of size %d x %d\n", m_M, m_N);
            for (int ii = 0; ii < m_M; ii++)
            {
                for (int jj = 0; jj < m_N; jj++)
                    m_pDense[ii + jj * m_M] = d_matrixEntry(m_sinkBasisMap[ii],
                                                            m_srcBasisMap[jj],
                                                            tHead, tLevel, true);
            }
        }
        else
        {
            m_Qmat = (std::complex<float> *)mkl_malloc(m_M * m_R * sizeof(std::complex<float>), 128);
            m_Rmat = (std::complex<float> *)mkl_malloc(m_R * m_N * sizeof(std::complex<float>), 128);
            printf("\t\t\tSetting up Q matrix of size %d x %d\n", m_M, m_R);
            printf("\t\t\tSetting up R matrix of size %d x %d\n", m_R, m_N);

            for (int ii = 0; ii < m_M; ++ii)
            {
                for (int jj = 0; jj < m_R; ++jj)
                    m_Qmat[ii + jj * m_M] = d_matrixEntry(m_sinkBasisMap[ii],
                                                          jj,
                                                          tHead, tLevel, false);
            }
            for (int ii = 0; ii < m_R; ++ii)
            {
                for (int jj = 0; jj < m_N; ++jj)
                    m_Rmat[ii + jj * m_R] = d_matrixEntry(m_srcBasisMap[jj], ii, tHead, tLevel, false);
            }
        }
    }

    double cost() const
    {
        return (m_R != 0) ? 1.0 * m_R * (m_M + m_N) : 1.0 * m_M * m_N;
    }

private:
    Task() { ; }
    int m_M;
    int m_N;
    int m_R;
    int m_cost;
    int m_id;
    long int m_size;
    std::vector<int> m_srcBasisMap;
    std::vector<int> m_sinkBasisMap;

    std::complex<float> *m_pDense, *m_Qmat, *m_Rmat;
};

class TaskBins
{
public:
    TaskBins() : m_OutputbufferSize(0) { ; }
    virtual ~TaskBins() { ; }
    void reserve(int size) { m_tasks.reserve(size); }
    void push_back(const Task &newTask) { m_tasks.push_back(newTask); }
    void setUp(int matrixSize, const TreeNode *treeHead, int numLevel)
    {
        printf("\t\tSetting up tasks with matrixSize of %d\n", matrixSize);
        printf("\t\tTotal number of tasks need to be setup: %d\n", m_tasks.size());
        for (int jj = 0; jj < m_tasks.size(); ++jj)
        {
            printf("\t\t\tSetting up task jj=%d\n", jj);
            m_tasks[jj].setUp(matrixSize, treeHead, numLevel);
        }
    }
    void prepareInternalBuffers(int matrixSize, int nRhs)
    {
        int maxSrcBuffer = -1;
        int maxTestBuffer = -1;
        int maxRankBuffer = -1;

        for (auto task : m_tasks)
        {
            maxSrcBuffer = std::max(maxSrcBuffer, task.N());
            maxTestBuffer = std::max(maxTestBuffer, task.M());
            maxRankBuffer = std::max(maxRankBuffer, task.R());
        }

        m_inputBuffer = (std::complex<float> *)mkl_malloc(nRhs * maxSrcBuffer * sizeof(std::complex<float>), 128);
        m_outputBuffer = (std::complex<float> *)mkl_malloc(nRhs * maxTestBuffer * sizeof(std::complex<float>), 128);
        m_internalBuffer = (std::complex<float> *)mkl_malloc(nRhs * maxRankBuffer * sizeof(std::complex<float>), 128);
        m_matmat = (std::complex<float> *)mkl_malloc(nRhs * matrixSize * sizeof(std::complex<float>), 128);

        printf("\t\tAllocated input buffer of size %d x %d\n", nRhs, maxSrcBuffer);
        printf("\t\tAllocated output buffer of size %d x %d\n", nRhs, maxTestBuffer);
        printf("\t\tAllocated internal buffer of size %d x %d\n", nRhs, maxRankBuffer);
        printf("\t\tAllocated matmat buffer of size %d x %d\n", nRhs, matrixSize);

        m_OutputbufferSize = maxTestBuffer;
    }
    void clearBuffer(int outputSize)
    {
        memset(&m_outputBuffer[0], 0, sizeof(std::complex<float>) * m_OutputbufferSize);
        memset(&m_matmat[0], 0, sizeof(std::complex<float>) * outputSize);
    }
    void matmat(const std::complex<float> *globalB, int nRHS, int matrixSize)
    {
        int jj = 0;
        for (auto task : m_tasks)
        {
            task.matmat(globalB, &m_matmat[0], &m_inputBuffer[0], &m_outputBuffer[0], &m_internalBuffer[0], nRHS, matrixSize);
        }
    }
    void addResult(std::complex<float> *result, int size)
    {
        vectorPlusEqualVector_MKL(&(result[0]), size, &(m_matmat[0]));
    }

private:
    int m_OutputbufferSize;
    std::vector<Task> m_tasks;

    std::complex<float> *m_matmat, *m_inputBuffer, *m_outputBuffer, *m_internalBuffer;
};

void runMIL_MatrixMultiplication(const std::string &MILFile,
                                 int numThreads,
                                 int numRepeat,
                                 int numRHS,
                                 const std::string &filter,
                                 const bool &useAffinity,
                                 const std::string &scheme,
                                 const bool &HT)
{
    std::fstream fMil;
    fMil.open(MILFile, std::fstream::in);
    std::string thisLine = "";
    std::multiset<Task> allTasks;
    long long int numElements = 0;
    int numTasks = 0;
    std::set<int> includedMILs;

    double totalCost = 0;
    while (!fMil.eof())
    {
        std::getline(fMil, thisLine);
        std::vector<std::string> words = SplitStringOnWhiteSpace(thisLine);
        int M, N, R, index;
        if (filter != "")
        {
            if (words.size() < 7)
                continue;
            if (filter != words[3])
                continue;
        }

        index = atoi(words[2].c_str());
        M = atoi(words[4].c_str());
        N = atoi(words[5].c_str());
        R = atoi(words[6].c_str());

        printf("M: %d, N: %d, R: %d\n", M, N, R);

        if (includedMILs.find(index) == includedMILs.end())
            includedMILs.insert(index);
        else
            continue;
        Task thisTask = Task(M, N, R, numTasks);
        totalCost += thisTask.cost();
        numElements += thisTask.size();

       // printf("Task cost: %f\n", thisTask.cost());
        //printf("Task size: %ld\n", thisTask.size());

        allTasks.insert(thisTask);
        ++numTasks;
    }

    printf("Total cost: %f\n", totalCost);
    printf("Total elements: %lld\n", numElements);

    fMil.close();
    int matrixSize = int(ceil(std::sqrt(1.0 * numElements)));
    printf("Matrix size: %d\n", matrixSize);
    int numLevel = 15;
    std::unique_ptr<TreeNode> treeHead;
    treeHead.reset(new TreeNode(0, 0));
    treeHead->grow(numLevel);

   // treeHead->printTree();

    int numBins = numThreads;
    printf("Number of bins: %d\n", numBins);

    std::vector<TaskBins> allTaskBins;
    allTaskBins.resize(numBins);
    for (auto entry : allTaskBins)
        entry.reserve(numTasks);
    int nextBin = 0;
    for (auto it = allTasks.rbegin(); it != allTasks.rend(); ++it)
    {
        allTaskBins[nextBin].push_back((*it));
        nextBin++;
        nextBin = nextBin % numBins;
    }

    clock_t beginTime = clock();
#pragma omp parallel num_threads(numBins)
    {
#pragma omp for
        for (int ii = 0; ii < numBins; ii++)
        {
            auto threadId = omp_get_thread_num();
            printf("\tStart to setting up all tasks for bin %d\n", ii);
            allTaskBins[threadId].setUp(matrixSize, treeHead.get(), numLevel);
        }
    }
#pragma omp barrier

    clock_t endTime = clock();
    double elapsedTimeSetup = ((double)(endTime - beginTime) / (double)(CLOCKS_PER_SEC));
    std::cout << "Set up done for taskcount : " << allTasks.size() << " in time(s): " << elapsedTimeSetup << std::endl;

    std::vector<std::complex<float>> RHS, tmp;

    RHS.resize(matrixSize * numRHS);
    tmp.resize(matrixSize * numRHS);

    printf("Setting up RHS matrix of size %d x %d\n", matrixSize, numRHS);
    printf("Setting up tmp matrix of size %d x %d\n", matrixSize, numRHS);
    for (int ii = 0; ii < numRHS; ii++)
    {
        for (int jj = 0; jj < matrixSize; jj++)
            RHS[jj + matrixSize * ii] = std::complex<float>(0.0001 * (ii + 1), -2e-7 * (jj + 1));
    }

#pragma omp parallel num_threads(numBins)
    {
#pragma omp for

        for (int ii = 0; ii < numBins; ii++)
        {
            auto threadId = omp_get_thread_num();
            allTaskBins[threadId].prepareInternalBuffers(matrixSize, numRHS);
        }
    }

    beginTime = clock();

    mkl_set_dynamic(0);
    omp_set_nested(true);
    mkl_set_num_threads(1);

#pragma omp parallel num_threads(numBins)
    for (int kk = 0; kk < numRepeat; kk++)
    {
        int tID = omp_get_thread_num();
        if (tID == 0)
            std::cout << "In iteration " << kk << std::endl;

#pragma omp for
        for (int jj = 0; jj < numBins; ++jj)
        {
            auto threadId = omp_get_thread_num();
            printf("\tStart to matmat for bin %d\n", jj);
            allTaskBins[threadId].matmat(&RHS[0], numRHS, matrixSize);
        }
#pragma omp barrier

#pragma omp single
        {
            memset(&tmp[0], 0, matrixSize * numRHS * sizeof(std::complex<float>));
            for (int jj = 0; jj < numBins; ++jj)
                allTaskBins[jj].addResult(&tmp[0], matrixSize * numRHS);

            float max = 0;
            for (int jj = 0; jj < matrixSize * numRHS; ++jj)
            {
                float val = std::abs(tmp[jj]);
                if (val > max)
                    max = val;
            }
            for (int jj = 0; jj < matrixSize * numRHS; ++jj)
            {
                RHS[jj] = tmp[jj];
                RHS[jj] /= max;
            }
        }
    }
    endTime = clock();
    double elapsedTimeMatMat = ((double)(endTime - beginTime) / (double)(CLOCKS_PER_SEC));
    std::cout << "Matmat done in time(s): " << elapsedTimeMatMat << std::endl;

    std::fstream fOut;
    fOut.open("Results.txt", std::fstream::out);
    for (int ii = 0; ii < matrixSize; ii++)
    {
        for (int jj = 0; jj < numRHS; jj++)
            fOut << RHS[ii + jj * matrixSize] << " ";
        fOut << "\n";
    }
    totalCost *= (numRepeat * numRHS / 1e9);
    fOut.close();
    fOut.open("Stat.txt", std::fstream::out);
    fOut << "Setup time " << elapsedTimeSetup << std::endl;
    fOut << "Matmat time " << elapsedTimeMatMat << std::endl;
    fOut << "GFLOPS " << totalCost << std::endl;
    fOut << "SPEED  " << totalCost / elapsedTimeMatMat << std::endl;
    fOut.close();
}

int main(int argc, char *argv[])
{
    if (argc < 8)
    {
        std::cerr << "Usage: " << argv[0] << " <MILFile> <numThreads> <numRepeat> <numRHS> <filter> <useAffinity> <scheme> <HT>" << std::endl;
        return 1;
    }

    std::string MILFile = argv[1];
    int numThreads = std::atoi(argv[2]);
    int numRepeat = std::atoi(argv[3]);
    int numRHS = std::atoi(argv[4]);
    std::string filter = argv[5];
    bool useAffinity = std::atoi(argv[6]);
    std::string scheme = argv[7];
    bool HT = std::atoi(argv[8]);

    runMIL_MatrixMultiplication(MILFile, numThreads, numRepeat, numRHS, filter, useAffinity, scheme, HT);

    return 0;
}
