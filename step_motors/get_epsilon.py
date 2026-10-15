#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.stats
from iminuit import Minuit
from iminuit.cost import LeastSquares

file_name = "data/data"
place_x = 32
place_y = 20
cx = -10
cy = -5
file_max_num = (place_x - cx * 2) * (place_y - cy * 2)
measure_time = 1
time_size = file_max_num // measure_time
data = []
center = []
eps_errors = []
pads_x = 32
pads_y = 20

# reading data
for num in range(time_size):
    file = open(file_name + str(num) + ".txt", "r")
    data.append([])
    data_tmp = []

    for line in file:
        ln = line.strip()
        ln_splitted = ln.split(" ")
        
        if(int(ln_splitted[0]) == pads_x - 1):
            data_tmp.append(float(ln_splitted[2]))
            data[num].append(data_tmp)
            data_tmp = []
        else:
            data_tmp.append(float(ln_splitted[2]))
        
    file.close()

# reading centers
cntr_file = open("data/centers.txt", "r")

for line in cntr_file:
    ln = line.strip()
    ln_splitted = ln.split(" ")
    center_tmp = []
    
    for i in ln_splitted:
        center_tmp.append(int(i))
           
    center.append(center_tmp)

cntr_file.close()

avg_data = []
epsilon = []

# epsilon to 1
for i in range(len(data[0])):
    tmp_epsilon = []
    tmp_data_avg = []
    tmp_eps_errors = []

    for j in range(len(data[0][0])):
        tmp_epsilon.append(1.)
        tmp_data_avg.append(0.)
        tmp_eps_errors.append(0.)
        
    epsilon.append(tmp_epsilon)
    avg_data.append(tmp_data_avg)
    eps_errors.append(tmp_eps_errors)

cycles = 5
cy = pads_y // 2
cx = pads_x // 2
epsilon_mean = 0.
avg_mean = 0
#print(epsilon)

# iteration
for n in range(cycles):
    # average profile in time
    for t in range(time_size):
        for i in range(-cy, cy):    
            for j in range(-cx, cx):
                #print("c", center[t][1] + i, center[t][0] + j)
                if center[t][1] + i >= 0 and center[t][1] + i < pads_y and center[t][0] + j >= 0 and center[t][0] + j < pads_x:
                    #print(i, j, len(data[t]), len(data[t][int(center[t][1] + i)]))
                    avg_data[cy + i][cx + j] += data[t][int(center[t][1] + i)][int(center[t][0] + j)] / epsilon[int(center[t][1] + i)][int(center[t][0] + j)]
                    avg_mean += 1
        #print(avg_data)
    avg_mean /= (pads_x * pads_y)
    
    for i in range(len(avg_data)):    
        for j in range(len(avg_data[0])):
            avg_data[i][j] = avg_data[i][j] / avg_mean
    #print(avg_data)
    #print(min(avg_data))
    
    # filtration
    filtered_data = gaussian_filter(avg_data, sigma=1.0)
    avg_data = list(filtered_data)
    #print(avg_data)

    epsilon_mean = 0.
        
    # counting epsilon
    for i in range(len(data[0])):    
        for j in range(len(data[0][0])):
            summ1 = 0.
            summ2 = 0.
            epsilon_xy = 0.
            for t in range(time_size):
                if int(cy + i - center[t][1]) < pads_y and int(cx + j - center[t][0]) < pads_x:
                    #print(i, j, summ1, summ2, data[t][i][j])
                    if data[t][i][j] > 0.01:
                        summ1 += data[t][i][j] * avg_data[int(cy + i - center[t][1])][int(cx + j - center[t][0])]
                        summ2 += avg_data[int(cy + i - center[t][1])][int(cx + j - center[t][0])] ** 2
                        #print(i, j, summ1, summ2)
            
            #print(i, j, summ1, summ2)
            #if(summ1 > 0.000000001 and summ2 > 0.000000001):
            #    epsilon_xy += summ1 / summ2
            #else:
            #    epsilon_xy += 1
            epsilon_xy += summ1 / summ2
            epsilon_mean += epsilon_xy
            epsilon[i][j] = epsilon_xy
                   
    # normalize epsilon
    epsilon_mean /= len(epsilon) * len(epsilon[0])
    #print(epsilon_mean)
    
    for i in range(len(epsilon)):
        for j in range(len(epsilon[0])):
            sigma_eps = (epsilon[i][j] - epsilon_mean) ** 2
        
    sigma_eps /= len(epsilon)
    #print(epsilon_mean, sigma_eps ** (0.5))
    
    for i in range(len(epsilon)):
        for j in range(len(epsilon[0])):
            epsilon[i][j] /= epsilon_mean
            
# counting errors
error1 = 0.
error2 = 0.
for j in range(len(data[0])):
    error1 = 0.
    error2 = 0.
    
    for i in range(len(data[0])):    
        for j in range(len(data[0][0])):
            error1 = 0.
            error2 = 0.
            epsilon_xy = 0.
            for t in range(time_size):
                if int(cy + i - center[t][1]) < pads_y and int(cx + j - center[t][0]) < pads_x:
                    if data[t][i][j] > 0.01:
                        error1 += data[t][i][j] * avg_data[int(cy + i - center[t][1])][int(cx + j - center[t][0])] ** 2
                        error2 += avg_data[int(cy + i - center[t][1])][int(cx + j - center[t][0])] ** 2
            if (epsilon_mean * error2) > 0.000001:
                eps_errors[i][j] = (error1) ** (0.5) / (epsilon_mean * error2)
            else:
                eps_errors[i][j] = 0.00001

# write epsilon
#print(epsilon)
with open("epsilon.txt", "w+") as f:
    epsilon_max = max(epsilon)

    for i in range(len(epsilon)):
        tmp_string = ""
        
        for j in range(len(epsilon[0])):
            tmp_string = str(j) + " " + str(i) + " " + str(epsilon[i][j]) + " " + str(eps_errors[i][j])
                
            f.write(tmp_string + "\n")
