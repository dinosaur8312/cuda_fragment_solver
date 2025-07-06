#include "GPUTask.h"
#include <tuple>
class GPUBatchTask
{
public:
    GPUBatchTask(int M_pad, int N_pad, int R_pad);
    GPUBatchTask(const std::tuple<int, int, int>& dims)
        : GPUBatchTask(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims)) {}
    ~GPUBatchTask();
    void addTask(GPUTask* task);
    std::vector<GPUTask*>& getTasks() { return m_tasks; }

private:
    int m_M_pad, m_N_pad, m_R_pad;
    std::vector<GPUTask*> m_tasks;
    cudaStream_t m_stream;


    void uploadBatchToDevice();
};