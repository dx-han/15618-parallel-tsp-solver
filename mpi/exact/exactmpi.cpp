#include "exactmpi.h"
#include "mpi.h"
#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>



int main(int argc, char *argv[]) {
    int procID;
    int nproc;
    double startTime;
    double endTime;
    char *inputFilename = NULL;
    int opt = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Read command line arguments
    do {
        opt = getopt(argc, argv, "f:");
        switch (opt) {
        case 'f':
            inputFilename = optarg;
            break;
        case -1:
            break;
        default:
            break;
        }
    } while (opt != -1);
    
    if (inputFilename == NULL) {
        printf("Usage: %s -f <filename> [-p <P>] [-i <N_iters>]\n", argv[0]);
        MPI_Finalize();
        return -1;
    }

    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);

    // Get total number of processes specificed at start of run
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Print parameters
    if (procID == 0) {
        printf("Input file: %s\n", inputFilename);
        printf("NUmber of threads: %d\n", nproc);
    }

    // Run computation
    compute(procID, nproc, inputFilename, &startTime, &endTime);

    // Cleanup
    MPI_Finalize();
    // printf("Elapsed time for proc %d: %f\n", procID, endTime - startTime);
    return 0;
}

void compute(int procID, int nproc, char *inputFilename, double *startTime, double *endTime) {
    int dim_x, dim_y;
    int num_of_city;
    double sTime;
    double eTime;
    *startTime = MPI_Wtime();

    
    sTime = MPI_Wtime();
    FILE *input = fopen(inputFilename, "r");
    if (!input) {
        printf("Unable to open file: %s.\n", inputFilename);
        return;
    }
    fscanf(input, "%d\n", &num_of_city);
    fscanf(input, "%d %d\n", &dim_x, &dim_y);
    
    std::vector<city_t> cities(num_of_city);
    for (int i = 0; i < num_of_city; i++) {
        city_t &city = cities[i];
        fscanf(input, "%d %d\n", &city.x, &city.y);
    }

    if (procID == 0) {
        printf("Number of cities: %d\n", num_of_city);
        printf("Map size: %d x %d\n", dim_x, dim_y);
    }

    /* ============= initialization =============*/

    // record distance between each city, outer map city id is smaller than inner map city id
    std::unordered_map<int, std::unordered_map<int, int>> distances;
    update_distances(distances, cities);

    /* run combinations for initialization in parallel */
    std::unordered_map<int, std::vector<std::vector<int>>> combinations;
    for (int i = 0; i < num_of_city - 1; i++) {
        int num_of_city_in_subset = i + 1;
        combinations[num_of_city_in_subset] = get_combination(num_of_city - 1, num_of_city_in_subset);
    }

    /**
    * graph: city_id: num_of_set: set (stored as string, cities are separated by ","), candidate_t
    * E.g. graph[5][3]['2,3,4,']={city:2,distance=40}: to go to city 5, only only go through city 2,3,4 from city1, the best path
    * to 1->{3,4}->2->5, and the distance from city1 to city 5 is 40
    */
    std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<std::string, candidate_t>>> graph;
    for (int i = 0; i < num_of_city - 1; i++) {
        candidate_t best_candidate;
        city_t first_city = cities[0];
        int to_city = i + 2;
        best_candidate.from_city = to_city;
        best_candidate.dist = get_euclidian_distance(first_city, cities[i + 1]);
        std::string path = "," + std::to_string(to_city) + ",";
        graph[to_city][1][path] = best_candidate;
    }
    initiate_graph(combinations, distances, graph);

    std::vector<candidate_t> last_candidates(num_of_city - 1);
    std::string other_city_except_city1 = "";
    for (int i = 2; i <= num_of_city; i++) {
        other_city_except_city1 += "," + std::to_string(i) + ",";
    }
    eTime = MPI_Wtime();
    // printf("ProcID: %d, Initialization Time: %lf.\n", procID, eTime - sTime);

    /* ============= run Held-Karp in parallel =============*/
    sTime = MPI_Wtime();
    for (int num_of_city_in_subset = 2; num_of_city_in_subset <= num_of_city - 1; num_of_city_in_subset++) {
        size_t num_of_subset = combinations[num_of_city_in_subset].size();
        std::vector<int> job_cnt;
        for (size_t i = 0; i < num_of_subset; i++) {
            job_cnt.emplace_back(i);
        }
        while (job_cnt.size() % nproc != 0) {
            job_cnt.emplace_back(-1);
        }
        int job_int_cnt = (int)job_cnt.size() / nproc;
        std::vector<int> jobs(job_int_cnt);
        if (procID == 0) {
            MPI_Scatter(job_cnt.data(), job_int_cnt, MPI_INT, jobs.data(), job_int_cnt, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatter(nullptr, job_int_cnt, MPI_INT, jobs.data(), job_int_cnt, MPI_INT, 0, MPI_COMM_WORLD);
        }
        // we need job!
        std::vector<size_t> graph_lv1_city;
        std::vector<size_t> graph_lv2_subset;
        std::vector<size_t> graph_lv3_str;
        std::vector<candidate_t> graph_lv4_candidate;
        int graph_size;

        for (size_t i = 0; i < jobs.size(); i++) {
            if (jobs[i] == -1) break;
            update_graph(combinations[num_of_city_in_subset][jobs[i]], distances, graph, 
            graph_lv1_city, graph_lv2_subset, graph_lv3_str, graph_lv4_candidate);
        }
        
        if (procID == 0) {
            graph_size = (int)graph_lv1_city.size();
        }
        MPI_Bcast(&graph_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        graph_lv1_city.resize(graph_size);
        graph_lv2_subset.resize(graph_size);
        graph_lv3_str.resize(graph_size * num_of_city_in_subset);
        graph_lv4_candidate.resize(graph_size);
        // while (graph_size > (int)graph_lv1_city.size()) {
        //     graph_lv1_city.emplace_back(-1);
        //     graph_lv2_subset.emplace_back(-1);
        //     graph_lv3_str.emplace_back(-1);
        //     candidate_t new_candidate;
        //     graph_lv4_candidate.emplace_back(new_candidate);
        // }

        std::vector<size_t> graph_lv1_city_gather(graph_size * nproc);
        std::vector<size_t> graph_lv2_subset_gather(graph_size * nproc);
        std::vector<size_t> graph_lv3_str_gather(graph_size * num_of_city_in_subset * nproc);
        std::vector<candidate_t> graph_lv4_candidate_gather(graph_size * nproc);

        MPI_Gather(graph_lv1_city.data(), graph_size * sizeof(graph_lv1_city[0]) / sizeof(short), MPI_SHORT, graph_lv1_city_gather.data(), graph_size * sizeof(graph_lv1_city[0]) / sizeof(short), MPI_SHORT, 0, MPI_COMM_WORLD);
        MPI_Gather(graph_lv2_subset.data(), graph_size * sizeof(graph_lv2_subset[0]) / sizeof(short), MPI_SHORT, graph_lv2_subset_gather.data(), graph_size * sizeof(graph_lv2_subset[0]) / sizeof(short), MPI_SHORT, 0, MPI_COMM_WORLD);
        MPI_Gather(graph_lv3_str.data(), graph_size * num_of_city_in_subset * sizeof(graph_lv3_str[0]) / sizeof(short), MPI_SHORT, graph_lv3_str_gather.data(), graph_size * num_of_city_in_subset * sizeof(graph_lv3_str[0]) / sizeof(short), MPI_SHORT, 0, MPI_COMM_WORLD);
        MPI_Gather(graph_lv4_candidate.data(), graph_size * sizeof(candidate_t) / sizeof(int), MPI_INT, graph_lv4_candidate_gather.data(), graph_size * sizeof(candidate_t) / sizeof(int), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(graph_lv1_city_gather.data(), graph_size * nproc * sizeof(graph_lv1_city[0]) / sizeof(short), MPI_SHORT, 0, MPI_COMM_WORLD);
        MPI_Bcast(graph_lv2_subset_gather.data(), graph_size * nproc * sizeof(graph_lv2_subset[0]) / sizeof(short), MPI_SHORT, 0, MPI_COMM_WORLD);
        MPI_Bcast(graph_lv3_str_gather.data(), graph_size * num_of_city_in_subset * nproc * sizeof(graph_lv3_str[0]) / sizeof(short), MPI_SHORT, 0, MPI_COMM_WORLD);
        MPI_Bcast(graph_lv4_candidate_gather.data(), graph_size * nproc * sizeof(candidate_t) / sizeof(int), MPI_INT, 0, MPI_COMM_WORLD);
        for (size_t i = 0; i < graph_lv1_city_gather.size(); i++) {
            if (graph_lv1_city_gather[i] == -1) break;
            int lv1 = (int)graph_lv1_city_gather[i];
            int lv2 = (int)graph_lv2_subset_gather[i];
            std::string lv3 = "";
            for (size_t j = 0; j < num_of_city_in_subset; j++) {
                lv3 += "," + std::to_string(graph_lv3_str_gather[num_of_city_in_subset * i + j]) + ",";
            }
            candidate_t lv4 = graph_lv4_candidate_gather[i];
            graph[lv1][lv2][lv3] = lv4;
        }
    }

    std::vector<double>compute_time_record(nproc);
    eTime = MPI_Wtime();
    double compute_time = eTime - sTime;
    // printf("Computation Time for proc %d: %lf.\n", procID, compute_time);

    MPI_Gather(&compute_time, 1, MPI_DOUBLE, compute_time_record.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // write to file
    if (procID == 0) {
        double general_computation_time = std::numeric_limits<double>::max();
        for (size_t i = 0; i < compute_time_record.size(); i++) {
            if (general_computation_time > compute_time_record[i]) {
                general_computation_time = compute_time_record[i];
            }
        }
        printf("General computation time: %.6f.\n", general_computation_time);

        for (int i = 0; i < num_of_city - 1; i++) {
            int from_city = i + 2; // 2...n
            int to_city = 1;
            int prev_dist = graph[from_city][num_of_city-1][other_city_except_city1].dist;
            int curr_dist = distances[to_city][from_city];
            candidate_t candidate;
            candidate.from_city = from_city;
            candidate.dist = curr_dist + prev_dist;
            last_candidates[i] = candidate;
        }

        // go through the last layer of graph can get the best path
        candidate_t last_best_path = find_best_path(last_candidates);

        std::vector<candidate_t> best_path;
        best_path.emplace_back(last_best_path);
        std::string path_gone = "";
        for (int num_of_city_in_subset = num_of_city - 1; num_of_city_in_subset > 1; num_of_city_in_subset--) {
            int to_city = best_path.back().from_city;
            std::string next_path = "";
            for (int i = 2; i <= num_of_city; i++) {
                std::size_t found = path_gone.find("," + std::to_string(i) + ",");
                if (found != std::string::npos) continue;
                next_path += "," + std::to_string(i) + ",";
            }
            path_gone += "," + std::to_string(to_city) + ",";
            candidate_t candidate = graph[to_city][num_of_city_in_subset][next_path];
            best_path.emplace_back(candidate);
        }

        int total_dist = last_best_path.dist;

        std::stringstream output;
        output << "output_" << std::to_string(num_of_city) << "_" << std::to_string(dim_x) << "x" << std::to_string(dim_y) << ".txt";
        std::string output_filename = output.str();
        write_output(best_path, output_filename, total_dist);
    }

    *endTime = MPI_Wtime();
    return;
}



int get_euclidian_distance(const city_t &city1, const city_t &city2) {
    int x1 = city1.x;
    int y1 = city1.y;
    int x2 = city2.x;
    int y2 = city2.y;
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

void update_distances(
    std::unordered_map<int, std::unordered_map<int, int>> &distances, 
    const std::vector<city_t> &cities) {
    // !let outer key is smaller than inner key
    for (size_t i = 0; i < cities.size(); i++) {
        for (size_t j = i + 1; j < cities.size(); j++) {
            int dist = get_euclidian_distance(cities[i], cities[j]);
            distances[i+1][j+1] = dist;
        }
    }
}

std::vector<std::vector<int>> get_combination(int n, int k) {
    std::vector<std::vector<int>> res;
    std::vector<int> curr;
    // start from city 2
    helper(2, k, n, curr, res);
    return res;
}

void helper(int idx, int k, int n, std::vector<int> &curr, std::vector<std::vector<int>> &res) {
    if ((int)curr.size() == k) {
        res.emplace_back(curr);
        return;
    }
    // start from 2, so end with n + 2 instead of n + 1
    for (int i = idx; i < n + 2; i++) {
        curr.emplace_back(i);
        helper(i + 1, k, n, curr, res);
        curr.pop_back();
    }
}

void initiate_graph(std::unordered_map<int, std::vector<std::vector<int>>> &combinations, std::unordered_map<int, std::unordered_map<int, int>> &distances, std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<std::string, candidate_t>>> &graph) {
    for (size_t num_of_city_in_subset = 2; num_of_city_in_subset <= combinations.size(); num_of_city_in_subset++) {
        std::vector<std::vector<int>> subsets = combinations[num_of_city_in_subset];
        for (size_t i = 0; i < subsets.size(); i++) {
            std::vector<int> subset = subsets[i];
            std::string path = "";
            for (size_t j = 0; j < subset.size(); j++) {
                path += "," + std::to_string(subset[j]) + ",";
            }
            candidate_t new_candidate;
            for (size_t j = 0; j < subset.size(); j++) {
                int city = subset[j];
                graph[city][num_of_city_in_subset][path] = new_candidate;
            }
        }
    }
}


void update_graph(
    const std::vector<int> &subset, 
    std::unordered_map<int, std::unordered_map<int, int>> &distances, 
    std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<std::string, candidate_t>>> &graph,
    std::vector<size_t> &graph_lv1_city,
    std::vector<size_t> &graph_lv2_subset,
    std::vector<size_t> &graph_lv3_str,
    std::vector<candidate_t> &graph_lv4_candidate) {
    size_t num_of_city = subset.size();
    for (size_t i = 0; i < subset.size(); i++) {
        int to_city = subset[i];
        int dist = std::numeric_limits<int>::max();
        std::string all_set = "";
        for (size_t j = 0; j < subset.size(); j++) {
            all_set += "," + std::to_string(subset[j]) + ",";
            graph_lv3_str.emplace_back((short)subset[j]);
        }
        candidate_t best_candidate;
        for (size_t j = 0; j < subset.size(); j++) {
            int from_city = subset[j];
            if (to_city == from_city) continue;
            std::string set = "";
            for (size_t k = 0; k < subset.size(); k++) {
                int city = subset[k];
                if (city != to_city) {
                    set += "," + std::to_string(city) + ",";
                }
            }
            int prev_dist = graph[from_city][num_of_city-1][set].dist;
            int curr_dist = from_city > to_city ? distances[to_city][from_city] : distances[from_city][to_city];
            int total_dist = prev_dist + curr_dist;
            if (total_dist < dist) {
                best_candidate.from_city = from_city;
                best_candidate.dist = total_dist;
                dist = total_dist;
            }
        }
        // printf("%d  %d  %s  %d  %d\n", to_city, (int)num_of_city, all_set.c_str(), best_candidate.from_city, best_candidate.dist);
        // #pragma omp critical
            graph[to_city][num_of_city][all_set] = best_candidate;
            graph_lv1_city.emplace_back((short)to_city);
            graph_lv2_subset.emplace_back((short)num_of_city);
            graph_lv4_candidate.emplace_back(best_candidate);
    }
}

candidate_t find_best_path(const std::vector<candidate_t> &candidates) {
    int city = 0;
    int dist = std::numeric_limits<int>::max();
    for (size_t i = 0; i < candidates.size(); i++) {
        candidate_t candidate = candidates[i];
        if (candidate.dist < dist) {
            city = candidate.from_city;
            dist = candidate.dist;
        }
    }
    candidate_t res;
    res.from_city = city;
    res.dist = dist;
    return res;
}

void write_output(const std::vector<candidate_t> &path, const std::string &filename, const int dist) {
    std::ofstream f(filename);
    // f << path.size() + 1 << std::endl;
    f << 0 << " ";
    for (size_t i = 0; i < path.size(); i++) {
        f << path[i].from_city - 1 << " ";
    }
    f << std::endl;
    f << dist << std::endl;
}