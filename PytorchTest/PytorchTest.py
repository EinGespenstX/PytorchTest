from __future__ import print_function
import numpy
import torch

print(torch.cuda.device_count())
print(torch.cuda.is_available())


#dtype = torch.float
device = torch.device("cuda")
#device = torch.device("cpu")

x = torch.ones(5, 5, device=device, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y
print(z)

out=z.mean()
print(out)
