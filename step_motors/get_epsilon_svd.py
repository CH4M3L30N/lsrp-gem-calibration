#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import gaussian_filter1d

file_name = "data/data"
full_time = 53
measure_time = 1
time_size = full_time // measure_time
data = []
center = []

# reading data
for num in range(full_time):
    file = open(file_name + str(num) + ".txt", "r")
    data.append([])

    for line in file:
        ln = line.strip()
        ln_splitted = ln.split(" ")
        
        for i in ln_splitted:
            data[num].append(float(i))
    
    file.close()

# using SVD
# creating matrix
data_reverse = []
for i in range(len(data[0])):
    data_reverse.append([])

    for j in range(len(data)):
        data_reverse[i].append(data[j][i])

D = np.array(data_reverse)
# using SVD
#print(D)
U, s, Vt = np.linalg.svd(D)
epsilon = []

for i in range(len(U)):
    epsilon.append(U[i][0])

epsilon_max = max(epsilon)
for i in range(len(epsilon)):
    epsilon[i] /= epsilon_max

print(epsilon)
with open("epsilon_svd.txt", "w+") as f:
    tmp_string = ""
    
    for j in range(len(epsilon)):
        tmp_string = str(j) + " " + str(epsilon[j])
            
        f.write(tmp_string + "\n")
#with open("epsilon2.txt", "w+") as f:
#    tmp_string = ""
#    
#    for j in range(len(U)):
#        if j == len(U) - 1:
#            tmp_string += str(U[j])
#        else:
#            tmp_string += str(U[j]) + " "
#            
#    f.write(tmp_string + "\n")
