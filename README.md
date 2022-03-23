# CMU 15-618 Final Project: Paralle TSP Solver
##### Team members: Yujun Qin and Dongxiao Han

## Project Proposal

### Summary
- We are going to implement a TSP solver with exact and heuristic algorithms.
- We are going to parallelize these algorithms using CUDA, OpenMP, MPI.
- We are going to compare the speedup with different libraries, algorithms and problem sizes.

### Background
The Traveling Salesman Problem (TSP) is a famous NP-hard problem. It asks for the shortest route to visit all the cities in the list exactly once. The TSP problem has applications in various fields for planning optimization. Since the TSP problem is NP-hard, finding exact solutions to this problem is very time consuming. Therefore, we expect that parallelism can be exploited to accelerate the process of solving the TSP problem.We would like to see how TSP algorithms can benefit from parallelism of GPU and multi-core CPU.

An exact solution to the TSP problem is the Held–Karp algorithm, which is a dynamic programming solution of time complexity O(n^2^2^n^) where n is the problem size. This problem requires exploring a wide range of permutations and thus we expect to see high speedups for this algorithm.

For a large problem size, the exact solution would take unreasonably long time to run (even with parallelism), therefore many heuristic algorithms were invented to approximate the exact solution, and they are much faster than the exact solution. We also plan to investigate how parallelism can accelerate these heuristic algorithms. 

We will focus on the symmetric version of the TSP problem in this project, with the distances between a pair of cities defined by their Euclidean distance.

### Challenge
Implementation of efficient CUDA versions of the algorithms is the most challenging part of the program. There are dependencies between states that need to be computed. It is challenging to encode and arrange the states so that the states can be computed in an efficient manner with CUDA kernels. Also, many concurrent states involve shared readings, which may also be exploited to reduce memory latencies.

### Resources
The list of machines we expect to use in this project:
- GHC
  - CPU: Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz, 8-core
  - GPU: GeForce RTX 2080 8G
- PSC Regular Memory Node
  - CPU: 2x AMD EPYC 7742 (2.25-3.40 GHz, 2x64 cores per node)
- PSC V100 GPU Node
  - CPU: 2x Intel Xeon Gold 6148 (2.40-3.70 GHz, 2x20 cores per node)
  - GPU: 8x NVIDIA V100-16GB

We expect to run OpenMP and MPI programs on PSC Regular Memory Nodes, and CUDA programs will run on GHC machines. We may need to run CUDA programs on PSC V100 GPU Nodes if we find it necessary.

### Goals & Deliverables
#### Plan to achieve (100% Goal)
1.  Implement exact and heuristic algorithms with different optimized strategies.
1. Implement these algorithms using CUDA, OpenMP, and MPI.
1. The speedup is decent for OpenMP and MPI.
1. The CUDA versions achieve reasonable speedup compared to CPU solutions.
1. Compare the difference between and within the libraries.
#### Extra goal to achieve (125% Goal)
1. Achieve a significant superlinear speedup for a specific optimized algorithm.
1. Explore more than two algorithms.
#### Minimum goal to achieve (75% Goal)
1. If the project moves slowly, we may sacrifice CUDA and MPI versions of the heuristic algorithm.

#### Deliverables
1. Speedup graphs with different libraries, algorithms, and problem sizes.
1. Analysis for the trend of the speedup on the graphs.
1. Visualization of the solutions given by the algorithms.

### Platform Choice
- Platform: GHC, PSC
- Language: C++
- Libraries: CUDA, OpenMP, MPI
We make these choice based on our familarity with them through the process of the course. We have easy access to the GHC and PSC machines. CUDA, OpenMP and OpenMPI are commonly used for high-performance parallel computing.

### Schedule
- Week 1 (03/21-03/27): Work on proposal. Research for algorithms.
- Week 2 (03/28-04/03): Implement OpenMP version of the exact solution. Start implementing CUDA version and MPI version of the exact solution.
- Week 3 (04/04-04/10): Implement CUDA version and MPI version of the exact solution.
- Week 4 (04/11-04/17): Debug and improve previous codes. Implement OpenMP version of the heuristic solution. (04/11 - checkpoint)
- Week 5 (04/18-04/24): Implement CUDA and MPI version of the heuristic solution.
- Week 6 (04/25-05/01): Final improvements. Gather data and write the final report. (04/29 - report)
- Week 7 (05/02-05/05): Prepare the poster and presentation. (0505 - presentation)