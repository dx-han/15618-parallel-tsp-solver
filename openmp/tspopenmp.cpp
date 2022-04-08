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

    /**
     * graph: city_id: num_of_set: set (stored as string, cities are separated by ","), candidate_t
     * E.g. graph[5][3]['2,3,4,']={city:2,distance=40}: to go to city 5, only only go through city 2,3,4 from city1, the best path
     * to 1->{3,4}->2->5, and the distance from city1 to city 5 is 40
     */
    std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<std::string, candidate_t>>> graph;
    std::unordered_map<int, std::vector<std::vector<int>>> combinations;
    // std::unordered_map<size_t, std::unordered_map<size_t, std::vector<path_t>>> path_group;
    std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<size_t, candidate_t>>>> dist_candidate_group;
    std::vector<candidate_t> last_candidates(num_of_city - 1);

    // record distance between each city, outer map city id is smaller than inner map city id
    std::unordered_map<int, std::unordered_map<int, int>> distances;
    update_distances(distances, cities);

    std::string other_city_except_city1 = "";
    for (int i = 1; i <= num_of_city; i++) {
        other_city_except_city1 += std::to_string(i) + ",";
    }
    
    
    /* run some data structures for initialization in parallel */
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
            for (int i = 0; i < num_of_city - 1; i++) {
                int num_of_city_in_subset = i + 1;
                combinations[num_of_city_in_subset] = get_combination(num_of_city - 1, num_of_city_in_subset);
            }
        
        // initialize distance between other cities and city 1
        #pragma omp for schedule(static)
            for (int i = 0; i < num_of_city - 1; i++) {
                candidate_t best_candidate;
                city_t first_city = cities[0];
                int to_city = i + 2;
                best_candidate.from_city = to_city;
                best_candidate.dist = get_euclidian_distance(first_city, cities[i + 1]);
                std::string path = std::to_string(to_city) + ",";
                // best_candidate.path = path;
                graph[to_city][1][path] = best_candidate;
            }
    }

    // for (int num_of_city_in_subset = 2; num_of_city_in_subset <= num_of_city - 1; num_of_city_in_subset++) {
    //     std::vector<std::vector<size_t>> subsets = combinations[num_of_city_in_subset];
    //     size_t num_of_subset = subsets.size();
    //     // num_of_city_in_subset is the number of to_city, (num_of_city_in_subset - 1) is the number of from_city
    //     int num_of_combination_parallel = num_of_city_in_subset * (num_of_city_in_subset - 1);
    //     std::vector<path_t> path(num_of_combination_parallel);
    //     // initialize candidate path for each city set
    //     for (size_t i = 0; i < num_of_subset; i++) {
    //         path_group[num_of_city_in_subset][i] = path;
    //     }
    //     for (size_t i = 0; i < num_of_subset; i++) {
    //         std::vector<size_t> subset = subsets[i];
    //         std::vector<size_t> subset_parallel = padding_subset(subset);
    //         #pragma omp parallel for schedule(static)
    //             // generate candidate path from one city i to city j across city set S_ij, where |S_ij| + 2 = num_of_city_in_subset
    //             for (int j = 0; j < num_of_city_in_subset; j++) {
    //                 update_path(path_group[num_of_city_in_subset][i], subset_parallel, j, num_of_city_in_subset);
    //             }
            
    //     }
    // }

    // printf("id: %d\n", omp_get_thread_num());
    // printf("%d %d\n", graph[2][1]["2,"].city, graph[2][1]["2,"].distance);
    // printf("%d %d\n", graph[3][1]["3,"].city, graph[3][1]["3,"].distance);
    // printf("%d %d\n", graph[4][1]["4,"].city, graph[4][1]["4,"].distance);

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
            // printf("numofsubset: %d\n", (int)num_of_subset);
            // step 1: calculate best path for each city in each subset
            #pragma omp barrier
            #pragma omp for schedule(static)
                for (size_t i = 0; i < num_of_subset; i++) {
                    // std::vector<int> subset = subsets[i]; // length 4 [3,4,6,7]
                    update_graph(combinations[num_of_city_in_subset][i], distances, graph);
                }
    //         for (size_t i = 0; i < num_of_subset; i++) {
    //             std::vector<size_t> subset = subsets[i];
    //             int num_of_combination_parallel = num_of_city_in_subset * (num_of_city_in_subset - 1);
    //             // // suppose subset={2,3,4}, subset_parallel={2,3,4,2,3,4}
    //             std::vector<size_t> subset_parallel = padding_subset(subset);
    //             #pragma omp barrier
    //             // // step 1: calculate total distance for each candidate path
    //             #pragma omp for schedule(static)
    //                 for (int j = 0; j < num_of_combination_parallel; j++) {
    //                     int from_city = path_group[num_of_city_in_subset][i][j].from;
    //                     int to_city = path_group[num_of_city_in_subset][i][j].to;
    //                     std::string set = path_group[num_of_city_in_subset][i][j].set;
    //                     int curr_dist = from_city < to_city ? distances[from_city][to_city] : distances[to_city][from_city];
    //                     int prev_dist = graph[from_city][num_of_city_in_subset-2][set].distance;
    //                     candidate_t candidate;
    //                     candidate.city = from_city;
    //                     candidate.distance = curr_dist + prev_dist;
    //                     #pragma omp critical
    //                         dist_candidate_group[to_city][num_of_city_in_subset][i][from_city] = candidate;
    //                     // #pragma omp critical
    //                     //     dist_candidate_group[num_of_city_in_subset][i][to_city][from_city] = candidate;
    //                 }
    //             // step 2: find the minimum distance from different city to city i across city {2,...,i-1,i+1,...,j-1,j+1,...n}, where n = s
    //             #pragma omp for schedule(static)
    //                 for (int j = 0; j < num_of_city_in_subset; j++) {
    //                     int to_city = subset_parallel[j * num_of_city_in_subset + j];
    //                     std::string other_cities = get_other_cities(subset_parallel, j, num_of_city_in_subset);
    //                     std::unordered_map<size_t, candidate_t> candidates = dist_candidate_group[to_city][num_of_city_in_subset][i];
    //                     candidate_t best_path = find_best_path(candidates);
    //                     graph[to_city][num_of_city_in_subset-1][other_cities] = best_path;
    //                 }
    //         }
        }
        // step 3: update for the last path from city i to city 1
        // #pragma omp for schedule(static)
        //     for (int i = 0; i < num_of_city - 1; i++) {
        //         int from_city = i + 2; // 2...n
        //         int to_city = 1;
        //         int prev_dist = graph[from_city][num_of_city-1][other_city_except_city1].dist;
        //         int curr_dist = distances[to_city][from_city];
        //         candidate_t candidate;
        //         candidate.from_city = from_city;
        //         candidate.dist = curr_dist + prev_dist;
        //         last_candidates[i] = candidate;
        //     }
    }

    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);

    // // go through graph can get the best path
    // candidate_t last_best_path = find_best_path(last_candidates);
    // std::vector<candidate_t> best_path;
    // best_path.emplace_back(last_best_path);
    // int total_dist = last_best_path.dist;
    // for (int num_of_city_in_subset = num_of_city - 1; num_of_city_in_subset > 0; num_of_city_in_subset--) {
    //     int to_city = best_path.back().from_city;
    //     int city = 0;
    //     int dist = std::numeric_limits<int>::max();
    //     for (auto iter = graph[to_city][num_of_city_in_subset].begin(); iter != graph[to_city][num_of_city_in_subset].end(); iter++) {
    //         candidate_t curr = iter->second;
    //         if (curr.dist < dist) {
    //             dist = curr.dist;
    //             city = curr.from_city;
    //         }
    //     }
    //     candidate_t candidate;
    //     candidate.from_city = city;
    //     candidate.dist = dist;
    //     total_dist += dist;
    //     best_path.emplace_back(candidate);
    // }

    // // write to output
    // std::stringstream output;
    // output << "output_" << std::to_string(num_of_city) << "_" << std::to_string(dim_x) << "x" << std::to_string(dim_y) << ".txt";
    // std::string output_filename = output.str();
    // write_output(best_path, output_filename, total_dist);
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
            distances[i][j] = get_euclidian_distance(cities[i], cities[j]);
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

void update_graph(const std::vector<int> &subset, std::unordered_map<int, std::unordered_map<int, int>> &distances, std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<std::string, candidate_t>>> &graph) {
    size_t num_of_city = subset.size(); 
    for (size_t i = 0; i < subset.size(); i++) {
        int to_city = subset[i];
        int dist = std::numeric_limits<int>::max();
        std::string all_set = std::to_string(subset[i]) + ",";
        candidate_t best_candidate;
        for (size_t j = 0; j < subset.size(); j++) {
            int from_city = subset[j];
            if (from_city == to_city) continue;
            all_set += std::to_string(subset[j]) + ",";
            std::string set = std::to_string(subset[j]) + ",";
            for (size_t k = 0; k < subset.size(); k++) {
                int city = subset[k];
                if (city != from_city && city != to_city) {
                    set += std::to_string(city) + ",";
                }
            }
            int prev_dist = graph[from_city][num_of_city-1][set].dist;
            int curr_dist;
            if (from_city > to_city) {
                curr_dist = distances[to_city][from_city];
            } else {
                curr_dist = distances[from_city][to_city];
            }
            // int curr_dist = from_city > to_city ? distances[to_city][from_city] : distances[from_city][to_city];
            int total_dist = prev_dist + curr_dist;
            if (total_dist < dist) {
                best_candidate.from_city = from_city;
                best_candidate.dist = total_dist;
            }
        }
        #pragma omp critical
            graph[to_city][num_of_city][all_set] = best_candidate;
    }
}

// std::vector<size_t> padding_subset(const std::vector<size_t> &subset) {
//     std::vector<size_t> res;
//     for (size_t i = 0; i < subset.size(); i++) {
//         for (size_t j = 0; j < subset.size(); j++) {
//             res.emplace_back(subset[j]);
//         }
//     }
//     return res;
// }

// void update_path(std::vector<path_t> &path, const std::vector<size_t> &subset_parallel, size_t to_idx, size_t num_of_city_in_subset) {
//     size_t to_city = subset_parallel[to_idx * num_of_city_in_subset + to_idx];
//     size_t k = 0;
//     for (size_t i = 0; i < num_of_city_in_subset; i++) {
//         size_t from_city = subset_parallel[to_idx * num_of_city_in_subset + i];
//         if (to_city != from_city) {
//             path_t curr_path;
//             curr_path.from = from_city;
//             curr_path.to = to_city;
//             std::string set = "";
//             for (size_t j = 0; j < num_of_city_in_subset; j++) {
//                 size_t city = subset_parallel[to_idx * num_of_city_in_subset + j];
//                 if (city != to_city && city != from_city) {
//                     set += std::to_string(city) + ",";
//                 }
//             }
//             curr_path.set = set;
//             path[to_idx * (num_of_city_in_subset - 1) + k] = curr_path;
//             k++;
//         }
//     }
// }

// std::string get_other_cities(const std::vector<size_t> &subset_parallel, size_t to_idx, size_t num_of_city_in_subset) {
//     // convert number to string
//     std::string res = "";
//     size_t to_city = subset_parallel[to_idx * num_of_city_in_subset + to_idx];
//     for (size_t i = 0; i < num_of_city_in_subset; i++) {
//         size_t from_city = subset_parallel[to_idx * num_of_city_in_subset + i];
//         if (to_city != from_city) {
//             res += std::to_string(from_city) + ",";
//         }
//     }
//     return res;
// }

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

// std::string get_last_other_cities(size_t curr_city, size_t num_of_city) {
//     std::string res = "";
//     for (size_t i = 2; i <= num_of_city; i++) {
//         if (i != curr_city) {
//             res += std::to_string(i) + ",";
//         }
//     }
//     return res;
// }

void write_output(const std::vector<candidate_t> &path, const std::string &filename, const int dist) {
    std::ofstream f(filename);
    f << path.size() << std::endl;
    f << dist << std::endl;
    f << 1 << " -> " << std::endl;
    for (size_t i = 0; i < path.size(); i++) {
        f << path[i].from_city  << " -> " << std::endl;
    }
    f << 1 << std::endl;
    f << std::endl;
}