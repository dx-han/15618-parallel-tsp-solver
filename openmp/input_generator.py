import random

number_of_city = 29
dim_x = 10
dim_y = dim_x
filename = 'input_{}_{}x{}'.format(number_of_city, dim_x, dim_y)

res = random.sample(range(1, dim_x * dim_y), number_of_city)
f = open('../inputs/{}.txt'.format(filename), 'w')
f.write('{}\n'.format(number_of_city))
f.write('{} {}\n'.format(dim_x, dim_y))
for num in res:
    row = num // dim_x
    col = num % dim_x
    f.write('{} {}\n'.format(row, col))
f.close()