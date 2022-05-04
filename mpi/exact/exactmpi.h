#ifndef __TSPOPT_H__
#define __TSPOPT_H__

#include "mpi.h"
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

typedef struct {
    int x;
    int y;
} city_t;

typedef struct {
    int from_city;
    int dist;
} candidate_t;



void compute(int procID, int nproc, char *inputFilename, double *startTime, double *endTime);
int get_euclidian_distance(const city_t &city1, const city_t &city2);
void update_distances(
    std::unordered_map<int, std::unordered_map<int, int>> &distances, 
    const std::vector<city_t> &cities);
std::vector<std::vector<int>> get_combination(int n, int k);
void helper(int idx, int k, int n, std::vector<int> &curr, std::vector<std::vector<int>> &res);
void initiate_graph(std::unordered_map<int, std::vector<std::vector<int>>> &combinations, std::unordered_map<int, std::unordered_map<int, int>> &distances, std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<std::string, candidate_t>>> &graph);
void update_graph(
    const std::vector<int> &subset, 
    std::unordered_map<int, std::unordered_map<int, int>> &distances, 
    std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_map<std::string, candidate_t>>> &graph,
    std::vector<size_t> &graph_lv1_city,
    std::vector<size_t> &graph_lv2_subset,
    std::vector<size_t> &graph_lv3_str,
    std::vector<candidate_t> &graph_lv4_candidate);
candidate_t find_best_path(const std::vector<candidate_t> &candidates);
void write_output(const std::vector<candidate_t> &path, const std::string &filename, const int dist);

#endif