from numpy import load
import os
import numpy as np

filepath = 'batch_001.npz'

file = load(filepath)
lst = file.files
print(lst)

data = file["data"]
labels = file["labels"]
det = file["det"]
ID = file["ID"]



print(np.shape(data))
# print(labels)
# print(det)
# print(ID)


print(data[0,0,1,:])

# print(det[0,1,0,:])

# print(det[0,2,0,:])

# print(det[0,3,0,:])
# print(data[:,:,0,:])


# for item in data[:,:,1,:]:
#     print(item)

# for item in lst:
#     x = data["data"]
#     print(np.shape(x))
#     # print(x)
#     # print(x[1,1,2,:])