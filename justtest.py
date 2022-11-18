import numpy as np
import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
torch.set_printoptions(threshold=np.inf)
logger.addHandler(logging.FileHandler(f'./crf.log', 'a',encoding = "UTF-8"))
print = logger.info

#=======================================================================================================================
sigma=2
limit=2
nclasses=20
distance_matrix=torch.zeros((2*limit+1,2*limit+1,nclasses))
output=torch.randn((288,512,20)) # width, height, nclasses
print('#==============================================================================================#')
for i in range(-limit, limit+1):
    for j in range(-limit, limit+1):
        for k in range(nclasses):
            distance_matrix[limit+i,limit+j,k]=i**2+j**2

print(distance_matrix[:,:,0])
print(distance_matrix.size())

weight_matrix = (torch.exp(-(distance_matrix**2)/(2*sigma**2)))/(sigma*np.sqrt(2*np.pi))

print(weight_matrix)
# print(output)#288x512x20
soft = nn.Softmax(dim=2)

prob = soft(output) #288x512x20
# print(prob)
# print(prob[:,0,0])
sum_ = torch.sum(prob[0,0,:])
print(sum_)
nclasses = output.shape[2]
width = output.shape[0]
height = output.shape[1]

weight_flat = torch.flatten(weight_matrix)
def prob_conv(x,y):
    prob_diff = torch.zeros((2 * limit + 1, 2 * limit + 1, nclasses))
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            prob_diff[limit + i, limit + j] = (prob[x + i, y + j] - prob[x, y]) ** 2
    print(f'x:{x},y:{y}')
    prob_diff = torch.flatten(prob_diff)
    # print(prob_diff.shape)
    # print(weight_flat.shape)
    dotprod = torch.dot(prob_diff, weight_flat)
    return dotprod

output_change = torch.zeros((width, height, nclasses))
for x in range(limit, width-limit):
    for y in range(limit, height-limit):
        output_change[x, y] = prob_conv(x, y)

# print(output_change[:,:,0])
output = output-output_change
