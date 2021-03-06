import random
import sys

number_of_city = int(sys.argv[1])
dim_x = 1000
dim_y = dim_x
filename = 'input_{}_{}x{}'.format(number_of_city, dim_x, dim_y)

res = random.sample(range(2, dim_x * dim_y), number_of_city)
f = open('../../inputs/{}.txt'.format(filename), 'w')
f.write('{}\n'.format(number_of_city))
f.write('{} {}\n'.format(dim_x, dim_y))
for num in res:
    row = (num + dim_x - 1) // dim_x
    col = num % dim_x
    f.write('{} {}\n'.format(row, col))
f.close()