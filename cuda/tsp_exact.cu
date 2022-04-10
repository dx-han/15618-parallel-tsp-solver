#include <stdio.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define BLOCKSIZE 512

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0') 

extern float toBW(int bytes, float sec);

struct TspState {
    float value;
    // int next;
};

struct GlobalConstants {
    float *distance;
    TspState *states;
    unsigned long searchSpace;
};

__constant__ GlobalConstants cuConsts;

__device__ __inline__ unsigned long nCr(int n, int r) {
    if (r + r > n) {
        r = n - r;
    }
    unsigned long ret = 1;
    for (int i = n; i > n - r; i--) {
        ret *= i;
    }
    for (int i = 2; i <= r; i++) {
        ret /= i;
    }
    return ret;
}

unsigned long host_nCr(int n, int r) {
    if (r + r > n) {
        r = n - r;
    }
    unsigned long ret = 1;
    for (int i = n; i > n - r; i--) {
        ret *= i;
    }
    for (int i = 2; i <= r; i++) {
        ret /= i;
    }
    return ret;
}

__device__ unsigned long getEncodedState(int count1, unsigned long kth) {
    unsigned long ret = 0;
    while (count1 > 0) {
        if (kth <= 1) {
            ret += (1ULL << count1) - 1;
            return ret;
        }
        int count0 = 0;
        unsigned long accumulated = 0;
        while (true) {
            unsigned long cnt = nCr(count0 + count1 - 1, count0);
            if (accumulated + cnt >= kth) {
                break;
            }
            accumulated += cnt;
            ++count0;
        }
        ret += (1ULL << (count0 + count1 - 1));
        count1--;
        kth -= accumulated;
    }
    return ret;
}

__device__ __inline__ unsigned long rightBitMask(int start) {
    return (1ULL << (start - 1)) - 1;
}

__device__ __inline__ unsigned long leftBitMask(int start) {
    return ~0ULL - rightBitMask(start);
}

__device__ __inline__ unsigned long expandState(unsigned long state, int start) {
    unsigned long leftMask = leftBitMask(start);
    unsigned long rightMask = rightBitMask(start);
    return ((state & leftMask) << 1) + (state & rightMask);
}

__device__ __inline__ unsigned long removeFromState(unsigned long expandedState, int vertexId) {
    unsigned long lMask = leftBitMask(vertexId) << 1;
    unsigned long rMask = rightBitMask(vertexId);
    // printf("Remove: %d: " BYTE_TO_BINARY_PATTERN " -> " BYTE_TO_BINARY_PATTERN "\n", vertexId, BYTE_TO_BINARY(expandedState), BYTE_TO_BINARY((((expandedState & lMask) >> 1) + (expandedState & rMask))));
    return ((expandedState & lMask) >> 1) + (expandedState & rMask);
}

__device__ __inline__ void debugPrintState(unsigned long expandedState, int N, int cur, bool old) {
    if (old)
        printf("Current: %d: " BYTE_TO_BINARY_PATTERN "\n", cur, BYTE_TO_BINARY(expandedState));
    else 
       printf("New: %d: " BYTE_TO_BINARY_PATTERN "\n", cur, BYTE_TO_BINARY(expandedState));
}

// __device__ __inline__ void debugPrintExpand(unsigned long state, int cur, unsigned long expand) {
//     printf("Expand: %d: " BYTE_TO_BINARY_PATTERN " -> " BYTE_TO_BINARY_PATTERN "\n", cur, BYTE_TO_BINARY(state), BYTE_TO_BINARY(expand));
// }

__global__ void helKarpStageKernel(int stage, int N, int stageSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int start = (index % (N - 1)) + 1;
    int stateId = index / (N - 1);
    TspState *states = cuConsts.states;
    if (stateId >= stageSize) {
        return;
    }
    if (stage == 0) {
        TspState s;
        s.value = cuConsts.distance[start];
        states[start - 1] = s;
        return;
    }

    unsigned long stateCode = getEncodedState(stage, stateId + 1);
    unsigned long expandedState = expandState(stateCode, start);
    // debugPrintState(stateCode, N, start, true);
    // debugPrintExpand(stateCode, start, expandedState);
    TspState s{FLT_MAX};
    float *distance = cuConsts.distance;
    
    for (int i = 0; i < N - 1; ++i) {
        unsigned long mask = 1ULL << i;
        if ((expandedState & mask) == 0) {
            continue;
        }
        int vertexId = i + 1;
        unsigned long newState = removeFromState(expandedState, vertexId);
        // debugPrintState(newState, N, vertexId, false);

        float cost = distance[start * N + vertexId] + states[newState * (N - 1) + i].value;
        if (cost < s.value) {
            s.value = cost;
        }
    }

    states[stateCode * (N - 1) + start - 1] = s;

}

__global__ void getResultKernel(int N, int *result, float *total_cost) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > 0) {
        return;
    }
    TspState *states = cuConsts.states;
    float *distance = cuConsts.distance;
    unsigned long stateCode = (1ULL << (N - 2)) - 1;
    int next_hop = 0;
    float best_cost = FLT_MAX;
    for (int i = 0; i < N - 1; ++i) {
        float cost = states[stateCode * (N - 1) + i].value;
        if (cost < best_cost) {
            best_cost = cost;
            next_hop = i + 1;
        }
    }
    *total_cost = best_cost + cuConsts.distance[next_hop];
    result[0] = 0;
    result[1] = next_hop;
    for (int i = 2; i < N; ++i) {
        int current = result[i - 1];
        unsigned long expandedState = expandState(stateCode, current);
        best_cost = FLT_MAX;
        next_hop = 0;
        unsigned long best_state;
        for (int j = 0; j < N - 1; ++j) {
            unsigned long mask = 1ULL << j;
            if ((expandedState & mask) == 0) {
                continue;
            }
            int vertexId = j + 1;
            unsigned long targetState = removeFromState(expandedState, vertexId);
            float cost = states[targetState * (N - 1) + j].value + distance[current * N + vertexId];
            if (cost < best_cost) {
                best_cost = cost;
                next_hop = vertexId;
                best_state = targetState;
            }
        }
        stateCode = best_state;
        result[i] = next_hop;
    }
}

void heldKarpCuda(int N, int *x, int *y, int *result, float *total_cost) {
    int totalBytes = sizeof(int) * 3 * N;
    int eachBytes = sizeof(int) * N;
    int searchSpace = 1 << (N - 1);

    float *device_cost;
    int *device_result;

    TspState *tspStates;

    //
    cudaMalloc(&device_cost, sizeof(float));
    cudaMalloc(&device_result, sizeof(int) * N);
    //

    cudaMalloc(&tspStates, sizeof(TspState) * searchSpace * (N - 1));

    float *device_distance;
    cudaMalloc(&device_distance, sizeof(float) * N * N);
    float *distance = new float[N * N];
    for (int i = 0; i < N; ++ i) {
        for (int j = 0; j < N; ++j) {
            int xdiff = x[i] - x[j];
            int ydiff = y[i] - y[j];
            float square_dist = xdiff * xdiff + ydiff * ydiff;
            distance[i * N + j] = sqrtf(square_dist);
        }
    }
    cudaMemcpy(device_distance, distance, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    delete[] distance;

    GlobalConstants params;
    params.distance = device_distance;
    params.states = tspStates;
    params.searchSpace = searchSpace;
    cudaMemcpyToSymbol(cuConsts, &params, sizeof(GlobalConstants));


    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    // run kernel
    double startKernelTime = CycleTimer::currentSeconds();
    dim3 blockDim(BLOCKSIZE, 1, 1);
    for (int i = 0; i < N; ++i) {
        unsigned long stageSize = host_nCr(N - 2, i);
        unsigned long threadCnt = stageSize * (N - 1);
        dim3 gridDim((threadCnt + blockDim.x - 1) / blockDim.x);
        helKarpStageKernel<<<gridDim, blockDim>>>(i, N, stageSize);
        int ret = cudaDeviceSynchronize();
        if (ret != 0) {
            printf("Failed!\n");
            break;
        }
        printf("Stage %d complete\n", i);
    }
    dim3 singleDim(1, 1, 1);
    getResultKernel<<<singleDim, singleDim>>>(N, device_result, device_cost);

    cudaDeviceSynchronize();
    double endKernelTime = CycleTimer::currentSeconds();
    //
    cudaMemcpy(result, device_result, eachBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(total_cost, device_cost, sizeof(float), cudaMemcpyDeviceToHost);

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

    cudaFree(device_result);    
    cudaFree(device_cost);
    cudaFree(device_result);
    cudaFree(tspStates);
    cudaFree(device_distance);
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
