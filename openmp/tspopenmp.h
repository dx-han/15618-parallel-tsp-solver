#ifndef __TSPOPT_H__
#define __TSPOPT_H__

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
#include <omp.h>


typedef struct {
    int x;
    int y;
} city_t;

typedef struct {
    int from;
    int to;
    // int distance;
    std::string set;
} path_t;

// typedef struct {
//     int distance;
//     int city;
// } best_path_t;

typedef struct {
    int city;
    int distance;
} candidate_t;

size_t get_euclidian_distance(const city_t &city1, const city_t &city2);
void update_distances(std::unordered_map<size_t, std::unordered_map<size_t, size_t>> &distances, const std::vector<city_t> &cities);
std::vector<std::vector<size_t>> get_combination(size_t n, size_t k);
void helper(size_t idx, size_t k, size_t n, std::vector<size_t> &curr, std::vector<std::vector<size_t>> &res);
std::vector<size_t> padding_subset(const std::vector<size_t> &subset);
void update_path(std::vector<path_t> &path, const std::vector<size_t> &subset_parallel, size_t to_idx, size_t num_of_city_in_subset);
std::string get_other_cities(const std::vector<size_t> &subset_parallel, size_t to_idx, size_t num_of_city_in_subset);
candidate_t find_best_path(std::vector<candidate_t> candidates);
std::string get_last_other_cities(size_t curr_city, size_t num_of_city);
#endif