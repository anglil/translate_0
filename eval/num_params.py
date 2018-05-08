import sys

dim_hidden = sys.argv[1]
dim_in = sys.argv[2]
dim_out = sys.argv[3]
num_layer = sys.argv[4]

n = int(dim_hidden)
m = int(dim_in)
v = int(dim_out)
i = int(num_layer)

num_params = 4*(n*m+n**2)*i + 4*(n*v+n**2)*i + n**2
print(num_params)
