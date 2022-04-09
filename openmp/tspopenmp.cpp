/**

#pragma omp parallel
#pragma omp barrier
#pragma omp for schedule(dynamic)
#pragmp omp critical

#pragma omp parallel for

*/

#include "tspopenmp.h"
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

static int _argc;
static const char **_argv;

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
    

    if (input_filename == NULL) {
        printf("Error: You need to specify -f.\n");
        return 1;
    }

    printf("Input filename: %s\n", input_filename);
    printf("Number of threads: %d\n", num_of_threads);

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

    printf("Number of cities: %d\n", num_of_city);
    printf("Map size: %d x %d\n", dim_x, dim_y);


    /* ============= initialization =============*/

    // record distance between each city, outer map city id is smaller than inner map city id
    std::unordered_map<int, std::unordered_map<int, int>> distances;
    update_distances(distances, cities);

    /* run combinations for initialization in parallel */
    std::unordered_map<int, std::vector<std::vector<int>>> combinations;
    #pragma omp parallel for schedule(static)
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

    init_time += duration_cast<dsec>(Clock::now() - init_start).count();
    printf("Initialization Time: %lf.\n", init_time);


    /* ============= run Held-Karp in parallel =============*/

    auto compute_start = Clock::now();
    double compute_time = 0;
    #pragma omp parallel
    {
        // use different size of candidate path set start from 2 cities to n-1 cities
        for (int num_of_city_in_subset = 2; num_of_city_in_subset <= num_of_city - 1; num_of_city_in_subset++) {
            size_t num_of_subset = combinations[num_of_city_in_subset].size();
            // step 1: calculate best path for each city in each subset
            #pragma omp barrier
            #pragma omp for schedule(static)
                for (size_t i = 0; i < num_of_subset; i++) {
                    // std::vector<int> subset = subsets[i]; // length 4 [3,4,6,7]
                    update_graph(combinations[num_of_city_in_subset][i], distances, graph);
                }
        }
        // step 2: update for the last path from city i to city 1
        #pragma omp for schedule(static)
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
    }

    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);
    
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

    // write to output
    std::stringstream output;
    output << "output_" << std::to_string(num_of_city) << "_" << std::to_string(dim_x) << "x" << std::to_string(dim_y) << ".txt";
    std::string output_filename = output.str();
    write_output(best_path, output_filename, total_dist);
}

int get_euclidian_distance(const city_t &city1, const city_t &city2) {
    int x1 = city1.x;
    int y1 = city1.y;
    int x2 = city2.x;
    int y2 = city2.y;
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

void update_distances(std::unordered_map<int, std::unordered_map<int, int>> &distances, const std::vector<city_t> &cities) {
    // !let outer key is smaller than inner key
    for (size_t i = 0; i < cities.size(); i++) {
        for (size_t j = i + 1; j < cities.size(); j++) {
            distances[i+1][j+1] = get_euclidian_distance(cities[i], cities[j]);
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

void update_graph(const std::vector<int> &subset, std::unordered_map<int, std::unordered_map<int, int>> &distances, std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<std::string, candidate_t>>> &graph) {
    size_t num_of_city = subset.size();
    for (size_t i = 0; i < subset.size(); i++) {
        int to_city = subset[i];
        int dist = std::numeric_limits<int>::max();
        std::string all_set = "";
        for (size_t j = 0; j < subset.size(); j++) {
            all_set += "," + std::to_string(subset[j]) + ",";
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
            }
        }
        // printf("%d  %d  %s  %d  %d\n", to_city, (int)num_of_city, all_set.c_str(), best_candidate.from_city, best_candidate.dist);
        // #pragma omp critical
            graph[to_city][num_of_city][all_set] = best_candidate;
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
    f << path.size() + 1 << std::endl;
    f << dist << std::endl;
    f << 1 << " -> ";
    for (size_t i = 0; i < path.size(); i++) {
        f << path[i].from_city  << " -> ";
    }
    f << 1 << std::endl;
}