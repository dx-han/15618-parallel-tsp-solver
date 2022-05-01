#include <stdio.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

#include "CycleTimer.h"

#define BLOCKSIZE 512

extern float toBW(int bytes, float sec);

const float pheromone_decay = .1f;
const float dist_coef = 2.f;
const float local_coef = .1f;
const int candidate_list_len = 64;
const float exploration_prob = 0.1f;
const int num_ant = 614;
const int seed = 1234;
const bool use_candidate_list = true;
const float t0 = 0.001f;
const int num_iter = 256;
const float pheromone_production = 1.f;
bool run_2opt_for_each_iter = false;

struct GlobalConstants {
    int *city_x;
    int *city_y;
    float *distance;
    float *adjusted_distance;
    float *pheromone;
    int *candidate_list;
    int *ant_tour;
    float *ant_tour_length;
    curandState *states;
    int N;
};

__constant__ GlobalConstants cuConsts;

struct Heap {
    int *data;
    int size;
    int x;
};

__device__ __forceinline__ void heap_push(Heap *heap, int id) {
    int current = heap->size;
    if (current >= candidate_list_len) {
        printf("Heap is full: Push rejected\n");
        return;
    }
    heap->size++;
    int x = heap->x;
    int N = cuConsts.N;
    int *data = heap->data;
    float *distance = cuConsts.distance + x * N;
    data[current] = id;
    float current_dist = distance[id];
    while (current > 0) {
        int parent = (current - 1) >> 1;
        if (current_dist > distance[data[parent]]) {
            int tmp = data[parent];
            data[parent] = data[current];
            data[current] = tmp;
        } else {
            break;
        }
        current = parent;
    }
}

__device__ __forceinline__ void heap_pop(Heap *heap) {
    if (heap->size <= 0) {
        printf("Heap is empty: Pop rejected\n");
        return;
    }
    int size = heap->size--;
    int x = heap->x;
    int N = cuConsts.N;
    int *data = heap->data;
    float *distance = cuConsts.distance + x * N;
    int top = data[0];
    data[0] = data[size];
    data[size] = top;
    int current = 0;
    float current_dist = distance[data[0]];
    while (current < size) {
        int child1 = (current << 1) + 1;
        int child2 = child1 + 1;
        if (child1 >= size) {
            break;
        }
        float dist1, dist2;
        if (child2 >= size) {
            dist2 = -FLT_MAX;
        } else {
            dist2 = distance[data[child2]];
        }
        dist1 = distance[data[child1]];
        int greater_child;
        if (dist1 > dist2) {
            greater_child = child1;
        } else {
            greater_child = child2;
        }
        if (distance[data[greater_child]] > current_dist) {
            int tmp = data[current];
            data[current] = data[greater_child];
            data[greater_child] = tmp;
        } else {
            break;
        }
        current = greater_child;
    }
}

struct NodeWeight {
    int node_id;
    float weight;
    __device__ bool operator<(const NodeWeight &r) const {
        return this->weight < r.weight;
    }
    __device__ bool operator>(const NodeWeight &r) const {
        return this->weight > r.weight;
    }
};

__global__ void initDistKernel() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int N = cuConsts.N;
    int i = index / N;
    int j = index % N;
    if (i >= N) {
        return;
    }
    if (i == j) {
        cuConsts.distance[index] = FLT_MAX;
        cuConsts.pheromone[index] = 0.f;
    }
    int *city_x = cuConsts.city_x;
    int *city_y = cuConsts.city_y;
    int diff_x = city_x[i] - city_x[j];
    int diff_y = city_y[i] - city_y[j];
    cuConsts.distance[index] = sqrtf(diff_x * diff_x + diff_y * diff_y);
    cuConsts.adjusted_distance[index] = pow(cuConsts.distance[index], -dist_coef);
    cuConsts.pheromone[index] = 1.f;
}

__global__ void initCandListKernel() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int N = cuConsts.N;
    if (index >= N) {
        return;
    }
    Heap heap;
    heap.size = 0;
    heap.x = index;
    float *distance = cuConsts.distance + index * N;
    heap.data = cuConsts.candidate_list + index * candidate_list_len;
    int j;
    for (j = 0; heap.size < candidate_list_len && j < N; ++j) {
        if (j != index) {
            heap_push(&heap, j);
        }
    }
    for (; j < N; ++j) {
        if (j != index) {
            int heap_top = heap.data[0];
            if (distance[j] < distance[heap_top]) {
                heap_pop(&heap);
                heap_push(&heap, j);
            }
        }
    }
    while (heap.size > 0) {
        heap_pop(&heap);
    }
}

__device__ void atomicAxpy(float *address, float a, float y) {
    int *address_as_int = (int*) address;
    int old = *address_as_int;
    int assumed, target;
    do {
        assumed = old;
        target = __float_as_int(a * __int_as_float(assumed) + y);
        old = atomicCAS(address_as_int, assumed, target);
    } while (assumed != old);
}

__global__ void acsKernel(int iter) {
    typedef cub::BlockScan<float, BLOCKSIZE> BlockScan;
    typedef cub::BlockReduce<NodeWeight, BLOCKSIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ typename BlockScan::TempStorage scan_storage;
    int ant_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int N = cuConsts.N;
    curandState *states = cuConsts.states;
    if (ant_id >= num_ant) {
        return;
    }
    if (iter == 0 && thread_id == 0) {
        curand_init(seed, ant_id, 0, &states[ant_id]);
    }
    extern __shared__ char shared[];
    float *acc_prob = (float*)shared;
    bool *visited = (bool*)(shared + sizeof(float) * N);
    for (int i = thread_id; i < N; i += BLOCKSIZE) {
        visited[i] = false;
    }
    __syncthreads();
    int *tour = cuConsts.ant_tour + ant_id * N;
    if (thread_id == 0) {
        tour[0] = curand(&states[ant_id]) % N;
        visited[tour[0]] = true;
    }
    float tour_length = 0.f;
    __shared__ bool is_exploration;
    __syncthreads();
    for (int i = 1; i < N; ++i) {
        __syncthreads();
        int from = tour[i - 1];
        float *pheromone = cuConsts.pheromone + from * N;
        float *distance = cuConsts.distance + from * N;
        float *adjusted_distance = cuConsts.adjusted_distance + from * N;

        if (thread_id == 0) {
            float r = curand_uniform(&states[ant_id]);
            is_exploration = (r < exploration_prob);
        }
        __syncthreads();
        // probability design: two-levels (BLOCKSIZE^2)
        // Thread i gathers prob using pheromone and distance
        // for all nodes whose id satisties (id % BLOCKSIZE == i).
        // All the nodes gathered by a thread forms a group.
        // First random number decides the group, 2nd random
        // number decides the node in the group.
        // __shared__ acc_prob stores accumulated pmf in group
        // init prob
        __shared__ float group_pmf[BLOCKSIZE];
        NodeWeight nw;
        nw.weight = -1.f;
        float group_sum = 0.f;
        for (int j = thread_id; j < N; j += BLOCKSIZE) {
            float p;
            if (visited[j]) {
                p = 0.f;
            } else {
                p = pheromone[j] * adjusted_distance[j];
            }
            group_sum += p;
            acc_prob[j] = group_sum;
            if (!visited[j] && p > nw.weight) {
                nw.weight = p;
                nw.node_id = j;
            }
        }
        group_pmf[thread_id] = group_sum;
        NodeWeight global_max = BlockReduce(reduce_storage).Reduce(nw, cub::Max());
        __syncthreads();
        int next_node;
        // 1. Inclusive Scan group pmf
        BlockScan(scan_storage).InclusiveSum(group_pmf[thread_id], group_pmf[thread_id]);
        __syncthreads();
        if (is_exploration && group_pmf[BLOCKSIZE - 1] > 0.f) {
            // 2. Thread 0 generates a random number 0-1 and scale to sum of pmf
            __shared__ float r;
            __shared__ int group;
            if (thread_id == 0) {
                r = curand_uniform(&states[ant_id]) * group_pmf[BLOCKSIZE - 1];
            }
            __syncthreads();
            // 3. Each thread examine a group to decide which group gets luck
            if (r < group_pmf[thread_id] && (thread_id == 0 || r >= group_pmf[thread_id - 1])) {
                group = thread_id;
            } else if (thread_id == BLOCKSIZE - 1 && r >= group_pmf[BLOCKSIZE - 1]) {
                // should not happen
                group = thread_id;
            }
            __syncthreads();
            // 4. The selected group broadcast the group_sum
            __shared__ float shared_group_sum;
            if (thread_id == group) {
                shared_group_sum = group_sum;
            }
            __syncthreads();
            // 5. Thread 0 generates a random number 0-1 and scale to group_sum
            __shared__ int shared_next_node;
            if (thread_id == 0) {
                r = curand_uniform(&states[ant_id]) * shared_group_sum;
            }
            __syncthreads();
            // 6. Each thread examine a node and decide the next node
            int test_node = group + thread_id * BLOCKSIZE;
            if (test_node < N) {
                if (r < acc_prob[test_node] && (test_node == group || r >= acc_prob[test_node - BLOCKSIZE])) {
                    shared_next_node = test_node;
                } else if (test_node + BLOCKSIZE >= N && r >= acc_prob[test_node]) {
                    // should not happen
                    shared_next_node = test_node;
                }
            }
            __syncthreads();
            next_node = shared_next_node;
        } else {
            next_node = global_max.node_id;
        }
        // Thread 0 prepares next node
        if (thread_id == 0) {
            visited[next_node] = true;
            tour[i] = next_node;
            atomicAxpy(&pheromone[next_node], 1 - local_coef, local_coef * t0);
            tour_length += distance[next_node];
        }
        __syncthreads();
    }
    if (thread_id == 0) {
        tour_length += cuConsts.distance[tour[N - 1] * N + tour[0]];
        cuConsts.ant_tour_length[ant_id] = tour_length;
        // printf("Length of tour %d: %f\n", ant_id, tour_length);
    }
}

__global__ void globalDecayKernel() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int N = cuConsts.N;
    int i = index / N;
    int j = index % N;
    if (i >= N) {
        return;
    }
    if (i == j) {
        return;
    }
    cuConsts.pheromone[index] *= 1.f - pheromone_decay;
}

__global__ void addPheromoneKernel(int best_ant, int tour_length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int N = cuConsts.N;
    if (index >= N) {
        return;
    }
    int *tour = cuConsts.ant_tour + best_ant * N;
    int from = tour[index];
    int to = tour[(index + 1) % N];
    cuConsts.pheromone[from * N + to] += pheromone_decay * pheromone_production / tour_length;
}

__global__ void check2OptKernel(int best_ant, int *improveTarget) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int N = cuConsts.N;
    int halfN = N >> 1;
    int edge1 = index / halfN;
    if (edge1 >= N) {
        return;
    }
    int edge2 = index % halfN;
    if (edge1 <= edge2) {
        edge1 = N - 1 - edge1;
        edge2 = halfN + halfN - 1 - edge2;
    }
    if (edge1 >= N || edge2 >= N || edge1 <= edge2 || 
        std::abs(edge1 - edge2) == 1 || std::abs(edge1 - edge2) == N - 1) {
        return;
    }

    int *tour = cuConsts.ant_tour + best_ant * N;
    float *distance = cuConsts.distance;
    int v1 = tour[edge1];
    int v2 = tour[(edge1 + 1) % N];
    int v3 = tour[edge2];
    int v4 = tour[(edge2 + 1) % N];
    float old_sum = distance[v1 * N + v2] + distance[v3 * N + v4];
    float new_sum = distance[v1 * N + v3] + distance[v2 * N + v4];
    if (new_sum < old_sum) {
        *improveTarget = edge1 * N + edge2;
    }
}

__global__ void run2OptKernel(int best_ant, int target, float *distance_diff) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int N = cuConsts.N;
    int edge1 = target / N;
    int edge2 = target % N;
    int diff = edge1 - edge2;
    int max_index = (diff >> 1);
    if (index >= max_index) {
        return;
    }
    int *tour = cuConsts.ant_tour + best_ant * N;
    if (index == 0) {
        float *distance = cuConsts.distance;
        int v1 = tour[edge1];
        int v2 = tour[(edge1 + 1) % N];
        int v3 = tour[edge2];
        int v4 = tour[(edge2 + 1) % N];
        float old_sum = distance[v1 * N + v2] + distance[v3 * N + v4];
        float new_sum = distance[v1 * N + v3] + distance[v2 * N + v4];
        *distance_diff = new_sum - old_sum;
        // printf("e1=%d, e2=%d\n", edge1, edge2);
    }
    int tmp = tour[edge2 + 1 + index];
    tour[edge2 + 1 + index] = tour[edge1 - index];
    tour[edge1 - index] = tmp;
}

void acsCuda(int N, int *x, int *y, int *result, float *total_cost) {
    int totalBytes = sizeof(int) * 3 * N;

    // allocate internal tmp arrays
    float *device_distance;
    float *device_pheromone;
    int *device_city_x;
    int *device_city_y;
    float *device_adjusted_distance;
    int *device_candidate_list;
    int *device_ant_tour;
    float *device_ant_tour_length;
    curandState *device_states;
    int *device_improve_target;
    float *device_distance_diff;
    cudaMalloc(&device_distance, sizeof(float) * N * N);
    cudaMalloc(&device_pheromone, sizeof(float) * N * N);
    cudaMalloc(&device_city_x, sizeof(int) * N);
    cudaMalloc(&device_city_y, sizeof(int) * N);
    cudaMalloc(&device_adjusted_distance, sizeof(float) * N * N);
    cudaMalloc(&device_candidate_list, sizeof(int) * N * N);
    cudaMalloc(&device_ant_tour, sizeof(int) * num_ant * N);
    cudaMalloc(&device_ant_tour_length, sizeof(float) *num_ant);
    cudaMalloc(&device_states, sizeof(curandState) * num_ant);
    cudaMalloc(&device_improve_target, sizeof(int));
    cudaMalloc(&device_distance_diff, sizeof(float));

    //
    

    GlobalConstants params;
    params.distance = device_distance;
    params.pheromone = device_pheromone;
    params.city_x = device_city_x;
    params.city_y = device_city_y;
    params.adjusted_distance = device_adjusted_distance;
    params.candidate_list = device_candidate_list;
    params.ant_tour = device_ant_tour;
    params.ant_tour_length = device_ant_tour_length;
    params.states = device_states;
    params.N = N;
    cudaMemcpyToSymbol(cuConsts, &params, sizeof(GlobalConstants));

    cudaMemcpy(device_city_x, x, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_city_y, y, sizeof(int) * N, cudaMemcpyHostToDevice);

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    // run kernel
    double startKernelTime = CycleTimer::currentSeconds();
    dim3 blockDim(BLOCKSIZE, 1, 1);
    dim3 NSquareGridDim((N * N + blockDim.x - 1) / blockDim.x);
    dim3 NGridDim((N + blockDim.x - 1) / blockDim.x);
    initDistKernel<<<NSquareGridDim, blockDim>>>();
    cudaDeviceSynchronize();
    float *host_distance = new float[N * N];
    cudaMemcpy(host_distance, device_distance, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    float host_tour_length[num_ant];
    float best_tour_length = FLT_MAX;
    int *best_tour = result;
    for (int i = 0; i < num_iter; ++i) {
        dim3 antGridDim(num_ant);
        acsKernel<<<antGridDim, blockDim, N * (sizeof(bool) + sizeof(int))>>>(i);
        cudaDeviceSynchronize();
        cudaMemcpy(host_tour_length, device_ant_tour_length, sizeof(float) * num_ant, cudaMemcpyDeviceToHost);
        float best_ant_length = FLT_MAX;
        int best_ant = 0;
        for (int j = 0; j < num_ant; ++j) {
            if (host_tour_length[j] < best_ant_length) {
                best_ant = j;
                best_ant_length = host_tour_length[j];
            }
        }
        dim3 check2optGridDim((N * (N >> 1) + blockDim.x - 1) / blockDim.x);
        int improveTarget;
        do {
            if (!run_2opt_for_each_iter && i < num_iter - 1) {
                break;
            }
            cudaMemset(device_improve_target, 0, sizeof(int));
            check2OptKernel<<<check2optGridDim, blockDim>>>(best_ant, device_improve_target);
            cudaDeviceSynchronize();
            cudaMemcpy(&improveTarget, device_improve_target, sizeof(int), cudaMemcpyDeviceToHost);
            if (improveTarget > 0) {
                int edge1 = improveTarget / N;
                int edge2 = improveTarget % N;
                dim3 run2optGridDim(((edge1 - edge2) / 2 + blockDim.x - 1) / blockDim.x);
                run2OptKernel<<<run2optGridDim, blockDim>>>(best_ant, improveTarget, device_distance_diff);
                cudaDeviceSynchronize();
                float distance_diff;
                cudaMemcpy(&distance_diff, device_distance_diff, sizeof(float), cudaMemcpyDeviceToHost);
                host_tour_length[best_ant] += distance_diff;
                // printf("2-opt: e1=%d, e2=%d improved %f\n", edge1, edge2, -distance_diff);
            }
        } while (improveTarget > 0);
        globalDecayKernel<<<NSquareGridDim, blockDim>>>();
        cudaDeviceSynchronize();
        addPheromoneKernel<<<NGridDim, blockDim>>>(best_ant, host_tour_length[best_ant]);
        cudaDeviceSynchronize();
        if (host_tour_length[best_ant] < best_tour_length) {
            best_tour_length = host_tour_length[best_ant];
            cudaMemcpy(best_tour, &device_ant_tour[best_ant * N], sizeof(int) * N, cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        printf("Stage %d complete\n", i);
    }
    double endKernelTime = CycleTimer::currentSeconds();
    *total_cost = best_tour_length;
    //

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double kernelDuration = endKernelTime - startKernelTime;
    printf("Kernel: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));
    double overallDuration = endTime - startTime;
    printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

    delete[] host_distance;
    cudaFree(device_distance);
    cudaFree(device_pheromone);
    cudaFree(device_city_x);
    cudaFree(device_city_y);
    cudaFree(device_adjusted_distance);
    cudaFree(device_candidate_list);
    cudaFree(device_ant_tour);
    cudaFree(device_ant_tour_length);
    cudaFree(device_states);
    cudaFree(device_improve_target);
    cudaFree(device_distance_diff);
}

void printCudaInfo() {
    // For fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
