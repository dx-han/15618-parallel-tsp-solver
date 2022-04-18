import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
    print("Usage: python3 visualize.py.py [input file] [output file]")

input = sys.argv[1]
output = sys.argv[2]
cities = []
cities_x = []
cities_y = []

with open(input, "r") as f:
    lines = f.readlines()
    N = int(lines[0])
    [maxX, maxY] = lines[1].split()
    maxX = int(maxX)
    maxY = int(maxY)
    points = lines[2:]
    for line in points:
        [x, y] = line.split()
        cities.append((int(x), int(y)))
        cities_x.append(int(x))
        cities_y.append(int(y))

with open(output, "r") as f:
    lines = f.readlines()
    order = [int(t) for t in lines[0].split()]
    order.append(order[0])
    cost = float(lines[1])

order_x = [cities_x[i] for i in order]
order_y = [cities_y[i] for i in order]

plt.scatter(cities_x, cities_y)
plt.plot(order_x, order_y)
plt.savefig("1.png")