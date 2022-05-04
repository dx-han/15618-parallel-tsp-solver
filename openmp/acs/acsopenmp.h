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
#include <math.h>

typedef double pheromone_t;


typedef struct {
    int x;
    int y;
} city_t;

double get_euclidian_distance(const city_t &city1, const city_t &city2);
void update_distances(std::unordered_map<int, std::unordered_map<int, double>> &distances, const std::vector<city_t> &cities);
pheromone_t calculate_pheronome0(std::unordered_map<int, std::unordered_map<int, double>>&distances, int num_of_city);
void write_output(const std::vector<int> &path, const std::string &filename, int dist);
#endif