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
    
    int error = 0;

    if (input_filename == NULL) {
        printf("Error: You need to specify -f.\n");
        error = 1;
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

    /**
     * graph: city_id: num_of_set: set (stored as string, cities are separated by ","), candidate_t
     * E.g. graph[5][3]['2,3,4,']={city:2,distance=40}: to go to city 5, only only go through city 2,3,4 from city1, the best path
     * to 1->{3,4}->2->5, and the distance from city1 to city 5 is 40
     */
    std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<std::string, candidate_t>>> graph(num_of_city);
    // record distance between each city, outer map city id is smaller than inner map city id
    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> distances(num_of_city);
    update_distances(distances, cities);
    std::vector<city_t> first_city_parallel;
    for (size_t i = 0; i < num_of_city - 1; i++) {
        first_city_parallel.emplace_back(cities[0]);
    }
    
    auto compute_start = Clock::now();
    double compute_time = 0;

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
            for (size_t i = 0; i < num_of_city - 1; i++) {
                candidate_t best_path;
                city_t first_city = first_city_parallel[i];
                std::string set = std::to_string(i+2) + ",";
                best_path.city = i + 2;
                best_path.distance = get_euclidian_distance(first_city, cities[i+1]);
                graph[i+2][1][set] = best_path;
            }
    }

    printf("%d %d\n", graph[2][1]["2,"].city, graph[2][1]["2,"].distance);
    printf("%d %d\n", graph[3][1]["3,"].city, graph[3][1]["3,"].distance);
    printf("%d %d\n", graph[4][1]["4,"].city, graph[4][1]["4,"].distance);

    std::vector<candidate_t> last_candidates(num_of_city-1);


    /* ===========debug done here, will continue========== */
    
    
    #pragma omp parallel
    {
        // use different size of candidate path set start from 2 cities to n-1 cities
        for (size_t s = 2; s <= num_of_city - 1; s++) {
            // for each subset, the number of city = s
            std::vector<std::vector<size_t>> subsets = get_combination(num_of_city, s);
            size_t num_of_subset = subset.size();
            for (size_t i = 0; i < num_of_subset; i++) {
                std::vector<size_t> subset = subsets[i];
                // num_of_city_in_subset is s, will delete var later
                size_t num_of_city_in_subset = subset.size();
                size_t num_of_combination_parallel = num_of_city_in_subset * (num_of_city_in_subset - 1);
                // suppose subset={2,3,4}, subset_parallel={2,3,4,2,3,4}
                std::vector<size_t> subset_parallel = padding_subset(subset);
                #pragma omp barrier
                std::vector<path_t> path(num_of_combination_parallel);
                // step1: generate candidate path from one city i to city j across city {2,...,i-1,i+1,...,j-1,j+1,...n}, where n = s
                #pragma omp for schedule(static)
                    for (size_t j = 0; j < num_of_city_in_subset; j++) {
                        update_path(path, subset_parallel, j, num_of_city_in_subset);
                    }
                std::unsorted_map<size_t, std::vector<candidate_t>> dist_candidate(num_of_city_in_subset, std::vector<candidate_t>(num_of_city_in_subset-1));
                // step2: calculate total distance for each candidate path
                #pragma omp for schedule(static)
                    for (size_t j = 0; j < num_of_combination_parallel; j++) {
                        size_t from_city = path[j].from;
                        size_t to_city = path[j].to;
                        std::string set = path[j].set;
                        size_t curr_dist = from_city < to_city ? distances[from_city][to_city] : distances[to_city][from_city];
                        size_t prev_dist = graph[from_city][s-2][set].distance;
                        candidate_t candidate;
                        candidate.city = from_city;
                        candidate.distance = curr_dist + prev_dist;
                        dist_candidate[to_city][from_city] = candidate;
                    }
                // step3: find the minimum distance from different city to city i across city {2,...,i-1,i+1,...,j-1,j+1,...n}, where n = s
                #pragma omp for schedule(static)
                    for (size_t j = 0; j < num_of_city_in_subset; j++) {
                        size_t to_city = subset_parallel[j * num_of_city_in_subset + j];
                        std::string other_cities = get_other_cities(subset_parallel, j, num_of_city_in_subset);
                        candidate_t best_path = find_best_path(dist_candidate[to_city]);
                        graph[to_city][s-1][other_cities] = best_path;
                    }
            }
        }

        // step4: update for the last path from city i to city 1
        size_t rest_city = num_of_city - 1;
        #pragma omp for schedule(static)
            for (size_t i = 0; i < num_of_city - 1;; i++) {
                size_t from_city = i + 2; // 2/.../n
                size_t to_city = 1;
                std::string other_cities = get_last_other_cities(curr_city, num_of_city);
                size_t curr_dist = distances[to_city][from_city];
                size_t prev_dist = grap[from_city][num_of_city-2][other_cities].distance;
                candidate_t candidate;
                candidate.city = from_city;
                candidate.distance = curr_dist + prev_dist;
                last_candidates[i] = candidate;
            }
    }

    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: %lf.\n", compute_time);

    //go through graph can get the best path
    candidate_t last_best_path = find_best_path(last_candidates);
    std::vector<candidate_t> best_path;
    best_path.emplace_back(last_best_path);
    size_t total_distance = last_best_path.distance;
    for (size_t s = num_of_city - 1; s > 0; s--) {
        size_t to_city = best_path.back().city;
        size_t city = 0;
        size_t distance = std::numeric_limits<int>::max();
        for (auto iter = graph[to_city][s]; iter != graph[to_city][s].end(); iter++) {
            candidate_t curr = iter.second;
            if (curr.distance < distance) {
                distance = curr.distance;
                city = curr.city;
            }
        }
        candidate_t candidate;
        candidate.city = city;
        candidate.distance = distance;
        best_path.emplace_back(candidate);
    }

    //write to output
    //best_path
    //total_distance
    // const char *output_filename = std::
    // FILE *output = fopen()

    // int res = get_euclidian_distance(cities[0], cities[1]);
    printf("here %d\n", g[1][1]["1"]);
    printf("here %d\n", g[2][1]["2"]);
    printf("here %d\n", g.size());
}

size_t get_euclidian_distance(const city_t &city1, const city_t &city2) {
    size_t x1 = city1.x;
    size_t y1 = city1.y;
    size_t x2 = city2.x;
    size_t y2 = city2.y;
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

void update_distances(std::unordered_map<size_t, std::unordered_map<size_t, size_t>> &distances, const std::vector<city_t> &cities) {
    // !let outer key is smaller than inner key
    for (size_t i = 0; i < cities.size(); i++) {
        for (size_t j = i + 1; j < cities.size(); j++) {
            distances[i][j] = get_euclidian_distance(cities[i], cities[j]);
        }
    }
}

std::vector<std::vector<size_t>> get_combination(size_t n, size_t k) {
    std::vector<std::vector<size_t>> res;
    std::vector<size_t> curr;
    helper(2, k, n, curr, res);
    return res;
}

void helper(size_t idx, size_t k, size_t n, std::vector<size_t> &curr, std::vector<std::vector<size_t>> &res) {
    if (curr.size() == k) {
        res.emplace_back(curr);
        return;
    }
    for (size_t i = idx; i < n + 1; i++) {
        curr.emplace_back(i);
        helper(idx + 1, k, n, curr, res);
        curr.pop_back();
    }
}

std::vector<size_t> padding_subset(const std::vector<size_t> &subset) {
    std::vector<size_t> res;
    for (size_t i = 0; i < subset.size(); i++) {
        for (size_t j = 0; j < subset.size(); j++) {
            res.emplace_back(subset[j]);
        }
    }
    return res;
}

void update_path(std::vector<path_t> &path, const std::vector<size_t> &subset_parallel, size_t to_idx, size_t num_of_city_in_subset) {
    size_t to_city = subset_parallel[to_idx * num_of_city_in_subset + to_idx];
    size_t j = 0;
    for (size_t i = 0; i < num_of_city_in_subset; i++) {
        size_t from_city = subset_parallel[to_idx * num_of_city_in_subset + i];
        if (to_city != from_city) {
            path_t curr_path;
            curr_path.from = from_city;
            curr_path.to = to_city;
            std::string set;
            for (size_t j = 0; j < num_of_city_in_subset; j++) {
                size_t city = subset_parallel[to_idx * num_of_city_in_subset + i];
                set += std::to_string(city) + ",";
            }
            curr_path.set = set;
            path[to_idx * num_of_city_in_subset + j] = curr_path;
        }
    }
}

std::string get_other_cities(const std::vector<size_t> &subset_parallel, size_t to_idx, size_t num_of_city_in_subset) {
    // convert number to string
    std::string res;
    size_t to_city = subset_parallel[to_idx * num_of_city_in_subset + to_idx];
    for (size_t i = 0; i < num_of_city_in_subset; i++) {
        size_t from_city = subset_parallel[to_idx * num_of_city_in_subset + i];
        if (to_city != from_city) {
            res += std::to_string(from_city) + ",";
        }
    }
    return res;
}

candidate_t find_best_path(std::vector<candidate_t> candidates) {
    size_t city = 0;
    int distance = std::numeric_limits<int>::max();
    for (size_t i = 0; i < candidates.size(); i++) {
        candidate_t curr_candidate = candidates[i];
        if (curr_candidate.distance < distance) {
            city = curr_candidate.city;
            distance = curr_candidate.distance;
        }
    }
    candidate_t res;
    res.city = city;
    res.distance = distance;
    return res;
}

std::string get_last_other_cities(size_t curr_city, size_t num_of_city) {
    std::string res;
    for (size_t i = 2; i <= num_of_city; i++) {
        if (i != curr_city) {
            res += std::to_string(i) + ",";
        }
    }
    return res;
}