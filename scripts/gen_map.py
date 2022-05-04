import random
import sys

map_size = 10000

if len(sys.argv) < 3:
    print("Usage: python3 gen_map.py [number of cities] [target_foler]")

N = int(sys.argv[1])
prefix = sys.argv[2]
filename = prefix + "/TSP_input_" + str(N) + ".txt"

with open(filename, "w") as f:
    print(N, file=f)
    print(map_size, map_size, file=f)
    for i in range(N):
        print(random.randint(0, map_size - 1), random.randint(0, map_size - 1), file=f)

