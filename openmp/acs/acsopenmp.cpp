#include "acsopenmp.h"
#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
#include <iostream>
#include <omp.h>
#include <math.h>

static int _argc;
static const char **_argv;
// const int num_of_ant = 60;
const double beta = 2; //distance over pheromone weight
const double q0 = 0.9; //ant colony system over ant system weight 
const double rou = 0.1; //local update weight
const double alpha = 0.1; //global update weight, pheromone decay param
const int num_of_iters = 1024;


const char *get_option_string(const char *option_name, const char *default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return _argv[i + 1];
    return default_value;
}

int get_option_int(const char *option_name, int default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return atoi(_argv[i + 1]);
    return default_value;
}

int main(int argc, const char *argv[]) {
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    auto init_start = Clock::now();
    double init_time = 0;

    _argc = argc - 1;
    _argv = argv + 1;

    const char *input_filename = get_option_string("-f", NULL);
    int num_of_threads = get_option_int("-n", 1);
    int num_of_ant = get_option_int("-a", 1);
    

    if (input_filename == NULL) {
        printf("Error: You need to specify -f.\n");
        return 1;
    }

    printf("Input filename: %s\n", input_filename);
    printf("Number of threads: %d\n", num_of_threads);
    printf("Number of ants: %d\n", num_of_ant);

    omp_set_num_threads(num_of_threads);
    omp_set_nested(1);

    FILE *input = fopen(input_filename, "r");
    
    if (!input) {
        printf("Unable to open file: %s.\n", input_filename);
        return 1;
    }

    int num_of_city;
    int dim_x, dim_y;
    fscanf(input, "%d\n", &num_of_city);
    fscanf(input, "%d %d\n", &dim_x, &dim_y);
    
    std::vector<city_t> cities(num_of_city);
    for (int i = 0; i < num_of_city; i++) {
        city_t &city = cities[i];
        fscanf(input, "%d %d\n", &city.x, &city.y);
    }

    if (num_of_city < num_of_ant) {
        printf("The number of city must greater than the number of ant\n");
        return 1;
    }

    printf("Number of cities: %d\n", num_of_city);
    printf("Map size: %d x %d\n", dim_x, dim_y);

    /* ============= initialization =============*/

    // record distance between each city, outer map city id is smaller than inner map city id
    std::unordered_map<int, std::unordered_map<int, double>> distances;
    update_distances(distances, cities);

    pheromone_t pheromone0 = calculate_pheronome0(distances, num_of_city);
    printf("init pheromone: %f\n", pheromone0);
    std::unordered_map<int, std::unordered_map<int, pheromone_t>> graph;
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

    // std::vector<double> ant_path_distance(num_of_ant, 0.0);

    init_time += duration_cast<dsec>(Clock::now() - init_start).count();
    printf("Initialization Time: %lf.\n", init_time);


    /* ============= run ACS in parallel =============*/
    auto compute_start = Clock::now();
    double compute_time = 0;
    std::mt19937 rand_eng;
    std::random_device r;
    rand_eng.seed(r());
    int best_path_ant_id;
    double best_path_dist;


    // ants build tours
    #pragma omp parallel
    {
        std::mt19937 rand_eng;
        #pragma omp critical
            rand_eng.seed(r());
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
        for (int iter = 0; iter < num_of_iters; iter++) {
            for (int i = 1; i <= num_of_city - 1; i++) {
                #pragma omp for schedule(static)
                    for (int j = 0; j < num_of_ant; j++) {
                        if (uniform_dist(rand_eng) <= q0) {
                            //acs state transition rule
                            double v = 0.0;
                            int city_r = ant_path[j][i-1];
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
                            int city_r = ant_path[j][i-1];
                            int city_s_pos = -1;
                            std::vector<double> probability;
                            std::vector<int> probability_city_id;
                            for (int k = i; k <= num_of_city - 1; k++) {
                                int candidate = ant_path[j][k];
                                int city_r_copy = city_r;
                                if (candidate < city_r_copy) {
                                    std::swap(city_r_copy, candidate);
                                }
                                double prod = graph[city_r_copy][candidate] * pow(distances[city_r_copy][candidate], -beta);
                                acc += prod;
                            }
                            // printf("acc:%.10f\n", acc);
                            for (int k = i; k <= num_of_city - 1; k++) {
                                int candidate = ant_path[j][k];
                                int city_r_copy = city_r;
                                if (candidate < city_r_copy) {
                                    std::swap(city_r_copy, candidate);
                                }
                                // printf("graph: %.10f, %.10f, %d, %d\n", graph[city_r_copy][candidate], pow(distances[city_r_copy][candidate], -beta), city_r, candidate);
                                double prod = graph[city_r_copy][candidate] * pow(distances[city_r_copy][candidate], -beta);
                                probability.emplace_back(prod / acc);
                                probability_city_id.emplace_back(k);
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
                    }
                #pragma omp for schedule(static)
                    //local update
                    for (int j = 0; j < num_of_ant; j++) {
                        int city_r = ant_path[j][i-1];
                        int city_s = ant_path[j][i];
                        if (city_s < city_r) {
                            std::swap(city_r, city_s);
                        }
                        #pragma omp critical
                            graph[city_r][city_s] = (1 - rou) * graph[city_r][city_s] + rou * pheromone0;
                    } 
            }
            #pragma omp for schedule(static)
                // local update for the last step to the initial city
                for (int j = 0; j < num_of_ant; j++) {
                    int city_r = ant_path[j][num_of_city-1];
                    int city_s = ant_path[j][num_of_city];
                    if (city_s < city_r) {
                        std::swap(city_r, city_s);
                    }
                    graph[city_r][city_s] = (1 - rou) * graph[city_r][city_s] + rou * pheromone0;
                }
            // update pheronome on the best path
            best_path_ant_id = 0;
            best_path_dist = std::numeric_limits<double>::max();
            for (int j = 0; j < num_of_ant; j++) {
                double v = 0.0;
                for (int k = 1; k <= num_of_city; k++) {
                    int city_r = ant_path[j][k-1];
                    int city_s = ant_path[j][k];
                    if (city_s < city_r) {
                        std::swap(city_r, city_s);
                    }
                    v += distances[city_r][city_s];
                }
                if (v < best_path_dist) {
                    best_path_dist = v;
                    best_path_ant_id = j;
                }
            }
            #pragma omp for schedule(static)
                for (int k = 1; k <= num_of_city; k++) {
                    int city_r = ant_path[best_path_ant_id][k-1];
                    int city_s = ant_path[best_path_ant_id][k];
                    if (city_s < city_r) {
                        std::swap(city_r, city_s);
                    }
                    graph[city_r][city_s] = (1 - alpha) * graph[city_r][city_s] + alpha * (1.0 / best_path_dist);
                }
        }
    }
    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);
    printf("Best total distance: %.10f\n", best_path_dist);
    
    std::vector<int> res;
    for(int i = 0; i < num_of_city; i++) {
        // printf("%d-", ant_path[best_path_ant_id][i]);
        res.emplace_back(ant_path[best_path_ant_id][i]);
    }

    // write to output
    std::stringstream output;
    output << "output_" << std::to_string(num_of_city) << "_" << std::to_string(dim_x) << "x" << std::to_string(dim_y) << ".txt";
    std::string output_filename = output.str();
    write_output(res, output_filename, best_path_dist);
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