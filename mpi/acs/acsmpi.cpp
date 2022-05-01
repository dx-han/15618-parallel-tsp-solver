#include "acsmpi.h"
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

const double beta = 2; //distance over pheromone weight
const double q0 = 0.9; //ant colony system over ant system weight 
const double rou = 0.1; //local update weight
const double alpha = 0.1; //global update weight, pheromone decay param
const int num_of_iters = 2;

int main(int argc, char *argv[]) {
    int procID;
    int nproc;
    double startTime;
    double endTime;
    char *inputFilename = NULL;
    int num_of_ant = 0;
    int opt = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Read command line arguments
    do {
        opt = getopt(argc, argv, "f:a:");
        switch (opt) {
        case 'f':
            inputFilename = optarg;
            break;
        case 'a':
            num_of_ant = atoi(optarg);
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
        printf("Number of ants: %d\n", num_of_ant);
    }

    // Run computation
    compute(procID, nproc, inputFilename, num_of_ant, &startTime, &endTime);

    // Cleanup
    MPI_Finalize();
    // printf("Elapsed time for proc %d: %f\n", procID, endTime - startTime);
    return 0;
}

void compute(int procID, int nproc, char *inputFilename, int num_of_ant, double *startTime, double *endTime) {
    int dim_x, dim_y;
    int num_of_city;
    double sTime;
    double eTime;
    *startTime = MPI_Wtime();

    
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

    if (num_of_city < num_of_ant) {
        printf("The number of city must greater than the number of ant\n");
        return;
    }

    if (procID == 0) {
        printf("Number of cities: %d\n", num_of_city);
        printf("Map size: %d x %d\n", dim_x, dim_y);
    }

    /* ============= initialization =============*/
    sTime = MPI_Wtime();

    // record distance between each city, outer map city id is smaller than inner map city id
    std::unordered_map<int, std::unordered_map<int, double>> distances;
    update_distances(distances, cities);
    std::unordered_map<int, std::unordered_map<int, pheromone_t>> graph;
    pheromone_t pheromone0;
    pheromone0 = calculate_pheronome0(distances, num_of_city);
    for (int i = 1; i <= num_of_city - 1; i++) {
        for (int j = i + 1; j <= num_of_city; j++) {
            graph[i][j] = pheromone0;
        }
    }

    std::vector<std::vector<int>> ant_path(num_of_ant);
    // every ant started at one of cities from 2 to n, ant1 start at city2, ant2 start at city3, and so on
    // since cities are generated randomly, the order is meaningless, so can initiate in this way
    
    for (int i = 0; i < num_of_ant; i++) {
        ant_path[i].emplace_back(i + 2);
        for (int j = 1; j <= num_of_city; j++) {
            if ((i + 2) != j) {
                ant_path[i].emplace_back(j);
            }
        }
        ant_path[i].emplace_back(i + 2);
    }
    std::vector<int> job_cnt;
    for (size_t z = 0; z < num_of_ant; z++) {
        job_cnt.emplace_back(z);
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
    
    eTime = MPI_Wtime();
    // printf("ProcID: %d, Initialization Time: %lf.\n", procID, eTime - sTime);
    std::vector<double> ant_path_dist(job_int_cnt * nproc, 0.0);

    /* ============= run Held-Karp in parallel =============*/
    sTime = MPI_Wtime();

    std::random_device r;
    std::vector<int>curr_best_path(num_of_city + 1);
    int best_path_ant_id;
    double best_path_dist;

    // ants build tours
    std::mt19937 rand_eng;
    rand_eng.seed(r());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    for (int iter = 0; iter < num_of_iters; iter++) {
        std::vector<double> job_path_dist(job_int_cnt, 0.0);
        for (int i = 1; i <= num_of_city - 1; i++) {
            std::vector<int> curr_city_r(job_int_cnt, -1);
            std::vector<int> curr_city_s(job_int_cnt, -1);
            for (int jid = 0; jid < (int)jobs.size(); jid++) {
                if (jobs[jid] == -1) continue;
                int j = jobs[jid];
                //ant_path, graph, distances
                int city_r = ant_path[j][i-1];
                if (uniform_dist(rand_eng) <= q0) {
                    //acs state transition rule
                    double v = 0.0;
                    int city_s_pos = -1;
                    for (int k = i; k <= num_of_city - 1; k++) {
                        int candidate = ant_path[j][k];
                        int city_r_copy = city_r;
                        if (candidate < city_r_copy) {
                            std::swap(city_r_copy, candidate);
                        }
                        double prod = graph[city_r_copy][candidate] * pow(distances[city_r_copy][candidate], -beta);
                        if (prod > v) {
                            v = prod;
                            city_s_pos = k;
                        }
                    }
                    std::swap(ant_path[j][i], ant_path[j][city_s_pos]);
                } else {
                    //as state transition rule
                    double v = 0.0;
                    double acc = 0.0;
                    int city_s_pos = -1;
                    std::vector<double> probability;
                    std::vector<int> probability_city_id;
                    std::vector<double> prod_tmp;
                    for (int k = i; k <= num_of_city - 1; k++) {
                        int candidate = ant_path[j][k];
                        int city_r_copy = city_r;
                        if (candidate < city_r_copy) {
                            std::swap(city_r_copy, candidate);
                        }
                        double prod = graph[city_r_copy][candidate] * pow(distances[city_r_copy][candidate], -beta);
                        prod_tmp.emplace_back(prod);
                        probability_city_id.emplace_back(k);
                        acc += prod;
                    }
                    for (size_t k = 0; k < prod_tmp.size(); k++) {
                        probability.emplace_back(prod_tmp[k] / acc);
                    }
                    double rand = uniform_dist(rand_eng);
                    for (size_t z = 0; z < probability.size(); z++) {
                        v += probability[z];
                        if (v >= rand) {
                            city_s_pos = probability_city_id[z];
                            break;
                        }
                    }
                    std::swap(ant_path[j][i], ant_path[j][city_s_pos]);
                }
                int city_s = ant_path[j][i];
                if (city_s < city_r) {
                    std::swap(city_r, city_s);
                }
                curr_city_r[jid] = city_r;
                curr_city_s[jid] = city_s;
                job_path_dist[jid] += distances[city_r][city_s];
            }
            std::vector<int> curr_city_r_gather(job_int_cnt * nproc);
            std::vector<int> curr_city_s_gather(job_int_cnt * nproc);
            MPI_Gather(curr_city_r.data(), job_int_cnt, MPI_INT, curr_city_r_gather.data(), job_int_cnt, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(curr_city_s.data(), job_int_cnt, MPI_INT, curr_city_s_gather.data(), job_int_cnt, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(curr_city_r_gather.data(), job_int_cnt * nproc, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(curr_city_s_gather.data(), job_int_cnt * nproc, MPI_INT, 0, MPI_COMM_WORLD);
            // local update
            for (size_t j = 0; j < curr_city_r_gather.size(); j++) {
                if (curr_city_r_gather[j] == -1) continue;
                int city_r = curr_city_r_gather[j];
                int city_s = curr_city_s_gather[j];
                if (city_s < city_r) {
                    std::swap(city_r, city_s);
                }
                graph[city_r][city_s] = (1 - rou) * graph[city_r][city_s] + rou * pheromone0;
            }
        }
        // update graph from the last city to the initial city
        for (int jid = 0; jid < (int)jobs.size(); jid++) {
            if (jobs[jid] == -1) continue;
            int j = jobs[jid];
            int city_r = ant_path[j][num_of_city-1];
            int city_s = ant_path[j][num_of_city];
            if (city_s < city_r) {
                std::swap(city_r, city_s);
            }
            job_path_dist[jid] += distances[city_r][city_s];
            graph[city_r][city_s] = (1 - rou) * graph[city_r][city_s] + rou * pheromone0;
        }
        MPI_Gather(job_path_dist.data(), job_int_cnt, MPI_DOUBLE, ant_path_dist.data(), job_int_cnt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(ant_path_dist.data(), job_int_cnt * nproc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // update pheronome on the best path
        best_path_ant_id = 0;
        best_path_dist = std::numeric_limits<double>::max();
        for (int j = 0; j < num_of_ant; j++) {
            double v = ant_path_dist[j];
            if (v == 0.0) continue;
            if (v < best_path_dist) {
                best_path_ant_id = j;
                best_path_dist = v;
            }
        }
        int target_procid = best_path_ant_id / job_int_cnt;
        if (procID == target_procid) {
            curr_best_path = ant_path[best_path_ant_id];
        }
        MPI_Bcast(curr_best_path.data(), num_of_city + 1, MPI_INT, target_procid, MPI_COMM_WORLD);

        // update pheronome globally
        for (int k = 1; k <= num_of_city; k++) {
            int city_r = curr_best_path[k-1];
            int city_s = curr_best_path[k];
            if (city_s < city_r) {
                std::swap(city_r, city_s);
            }
            graph[city_r][city_s] = (1 - alpha) * graph[city_r][city_s] + alpha * (1.0 / best_path_dist);
        }
    }

    std::vector<double>compute_time_record(nproc);
    eTime = MPI_Wtime();
    double compute_time = eTime - sTime;
    // printf("Computation Time for proc %d: %lf.\n", procID, compute_time);

    MPI_Gather(&compute_time, 1, MPI_DOUBLE, compute_time_record.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (procID == 0) {
        double general_computation_time = std::numeric_limits<double>::max();
        for (size_t i = 0; i < compute_time_record.size(); i++) {
            if (general_computation_time > compute_time_record[i]) {
                general_computation_time = compute_time_record[i];
            }
        }
        printf("General computation time: %.6f.\n", general_computation_time);
        std::vector<int> res;
        double final_dist = 0.0;
        res.emplace_back(curr_best_path[0]);
        for (size_t i = 1; i < curr_best_path.size(); i++) {
            int city_r = curr_best_path[i];
            int city_s = curr_best_path[i-1];
            if (city_s < city_r) {
                std::swap(city_r, city_s);
            }
            final_dist += distances[city_r][city_s];
            res.emplace_back(curr_best_path[i]);
        }
        // write to file
        std::stringstream output;
        output << "output_" << std::to_string(num_of_city) << "_" << std::to_string(dim_x) << "x" << std::to_string(dim_y) << ".txt";
        std::string output_filename = output.str();
        write_output(res, output_filename, final_dist);
    }
    *endTime = MPI_Wtime();
    return;
}



double get_euclidian_distance(const city_t &city1, const city_t &city2) {
    int x1 = city1.x;
    int y1 = city1.y;
    int x2 = city2.x;
    int y2 = city2.y;
    return sqrt((x1 - x2) * (x1 - x2) * 1.0 + (y1 - y2) * (y1 - y2) * 1.0);
}

void update_distances(std::unordered_map<int, std::unordered_map<int, double>> &distances, const std::vector<city_t> &cities) {
    // !let outer key is smaller than inner key
    for (size_t i = 0; i < cities.size(); i++) {
        for (size_t j = i + 1; j < cities.size(); j++) {
            distances[i+1][j+1] = get_euclidian_distance(cities[i], cities[j]);
        }
    }
}

pheromone_t calculate_pheronome0(std::unordered_map<int, std::unordered_map<int, double>>&distances, int num_of_city) {
    std::unordered_set<int> path;
    for (int i = 1; i <= num_of_city; i++) {
        path.insert(i);
    }
    int r, s, c;
    r = s = c = 1;
    double dist = 0;
    double total_closest_dist = 0;
    
    while (path.size() > 1) {
        path.erase(r);
        double closest_dist = std::numeric_limits<double>::max();
        for (auto it = path.begin(); it != path.end(); it++) {
            c = *it;
            dist = r > c ? distances[c][r] : distances[r][c];
            if (dist < closest_dist) {
                s = c;
                closest_dist = dist;
            }
        }
        total_closest_dist += closest_dist;
        r = s;
    }
    total_closest_dist += distances[1][r];
    return 1.0 / (num_of_city * total_closest_dist);
}

void write_output(const std::vector<int> &path, const std::string &filename, int dist) {
    std::ofstream f(filename);
    // f << path.size() + 1 << std::endl;
    // f << 0 << " ";
    for (size_t i = 0; i < path.size(); i++) {
        f << path[i] - 1 << " ";
    }
    f << std::endl;
    f << dist << std::endl;
}