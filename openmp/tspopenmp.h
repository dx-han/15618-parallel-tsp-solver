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
    int from_city;
    int dist;
} candidate_t;

int get_euclidian_distance(const city_t &city1, const city_t &city2);
void update_distances(std::unordered_map<int, std::unordered_map<int, int>> &distances, const std::vector<city_t> &cities);
std::vector<std::vector<int>> get_combination(int n, int k);
void helper(int idx, int k, int n, std::vector<int> &curr, std::vector<std::vector<int>> &res);
// std::vector<size_t> padding_subset(const std::vector<size_t> &subset);
void update_graph(const std::vector<int> &subset, std::unordered_map<int, std::unordered_map<int, int>> &distances, std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<std::string, candidate_t>>> &graph);
// void update_path(std::vector<path_t> &path, const std::vector<size_t> &subset_parallel, size_t to_idx, size_t num_of_city_in_subset);
// std::string get_other_cities(const std::vector<size_t> &subset_parallel, size_t to_idx, size_t num_of_city_in_subset);
candidate_t find_best_path(const std::vector<candidate_t> &candidates);
// std::string get_last_other_cities(size_t curr_city, size_t num_of_city);
void write_output(const std::vector<candidate_t> &path, const std::string &filename, const int dist);
#endif