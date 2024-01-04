import torch

a, b = torch.utils.data.random_split(range(10), [3, 7])

A = []
for item in a:
    A.append(int(item))

B = []
for item in b:
    B.append(item)

print('{},{}'.format(A,B))