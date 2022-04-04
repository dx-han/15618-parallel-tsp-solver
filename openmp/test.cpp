#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <unordered_map>
using namespace std;

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

std::vector<std::vector<size_t>> get_combination(size_t n, size_t k) {
    std::vector<std::vector<size_t>> res;
    std::vector<size_t> curr;
    helper(2, k, n, curr, res);
    return res;
}



int main()
{
  // std::vector<std::vector<size_t>> res = get_combination(4, 2);
  // for (size_t i = 0; i < res.size(); i++) {
  //   for (size_t j = 0; j < res[i].size(); j++) {
  //     printf("%d ", (int)res[i][j]);
  //   }
  //   printf("\n");
  // }
//   std::unordered_map<int, std::unordered_map<int, int>> distances;
//   distances[1][2] = 12;
//   distances[11][22] = 1122;
//   printf("%d\n", distances[1][2]);
//   printf("%d\n", distances[11][22]);
//   printf("%d\n", distances[11][23]);
//   size_t i = 100;
//   std::string res;
//   res += std::to_string(i) + ",";
//   cout << res << endl;
std::vector<int> res;
res.emplace_back(10);
res.emplace_back(100);
printf("%d\n", res.back());
printf("%d\n", res.back());
}
//g++ -m64 -std=c++11 -I. -O3 -Wall -Wno-unknown-pragmas -o test test.cpp && ./test
//g++ -m64 -std=c++11 -I. -O3 -Wall -fopenmp -Wno-unknown-pragmas -o test test.cpp && ./test