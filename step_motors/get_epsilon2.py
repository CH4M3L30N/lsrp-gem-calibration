#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.stats
from iminuit import Minuit
from iminuit.cost import LeastSquares

file_name = "data/data"
full_time = 53
measure_time = 1
time_size = full_time // measure_time
data = []
center = []
eps_errors = []
pads_x = 32

# reading data
for num in range(full_time):
    file = open(file_name + str(num) + ".txt", "r")
    data.append([])

    for line in file:
        ln = line.strip()
        ln_splitted = ln.split(" ")
        
        data[num].append(float(ln_splitted[1]))
    
    file.close()

# reading centers
cntr_file = open("data/centers.txt", "r")

for line in cntr_file:
    ln = line.strip()
    center.append(int(ln))

cntr_file.close()

avg_data = []
epsilon = []

# epsilon to 1
for i in range(len(data[0])):
    epsilon.append(1.)
    avg_data.append(0.)
    eps_errors.append(0.)

cycles = 150
cx = len(avg_data) // 2 - 1
epsilon_mean = 0.
#print(center)

# iteration
for n in range(cycles):
    sigma_eps = 0

    # average profile in time
    for t in range(full_time):
        for j in range(-10, 10):
            #print(center[t][1] + i)
            if center[t] + j >= 0 and center[t] + j < len(data[t]):
                avg_data[cx + j] += data[t][int(center[t] + j)] / epsilon[int(center[t] + j)]
                
                
        #print(avg_data)
    
    for j in range(len(avg_data)):
        avg_data[j] = avg_data[j] / pads_x
    #print(avg_data)
    
    # filtration
    filtered_data = gaussian_filter1d(avg_data, sigma=1.0)
    avg_data = list(filtered_data)

    epsilon_mean = 0.
        
    # counting epsilon
    for j in range(len(data[0])):
        summ1 = 0.
        summ2 = 0.
        epsilon_xy = 0.
        for t in range(full_time):
            if int(cx + j - center[t]) < pads_x:
                if data[t][j] > 0.01:
                    summ1 += data[t][j] * avg_data[int(cx + j - center[t])]
                    summ2 += avg_data[int(cx + j - center[t])] * avg_data[int(cx + j - center[t])]
                    #print(summ1, summ2)
            
            #print(summ1, summ2)
        epsilon_xy += summ1 / summ2
        epsilon_mean += epsilon_xy
        epsilon[j] = epsilon_xy
            
    # normalize epsilon
    epsilon_mean /= len(epsilon)
    #print(epsilon_mean, sigma_eps)
    
    for i in range(len(epsilon)):
        sigma_eps = (epsilon[i] - epsilon_mean) ** 2
        
    sigma_eps /= len(epsilon)
    #print(epsilon_mean, sigma_eps ** (0.5))
    
    #epsilon_max = max(epsilon)   
    for i in range(len(epsilon)):
        epsilon[i] /= epsilon_mean
            
# counting errors
error1 = 0.
error2 = 0.
for j in range(len(data[0])):
    error1 = 0.
    error2 = 0.
    for t in range(full_time):
        if int(cx + j - center[t]) < pads_x:
            if data[t][j] > 0.01:
                error1 += data[t][j] * avg_data[int(cx + j - center[t])] ** 2
                error2 += avg_data[int(cx + j - center[t])] ** 2
    eps_errors[j] = (error1) ** (0.5) / (epsilon_mean * error2)
            
# chi-square
file = open("epsilon_real.txt", "r")
real_eps = []

for line in file:
   ln = line.strip()
   ln_splitted = ln.split(" ")
     
   real_eps.append(float(ln_splitted[1]))
    
file.close()

diff = (np.array(epsilon) - np.array(real_eps))
chi2 = np.sum((diff / np.array(eps_errors)) ** 2)
    
ndf = len(epsilon) - 1
prob = scipy.stats.chi2.sf(chi2, ndf)
eps_mean = np.mean(diff)
bin_nums = np.linspace(0, len(epsilon), len(epsilon))

def linear_model(x, a, b):
    return a * x + b

least_squares = LeastSquares(bin_nums, diff, eps_errors, linear_model)
m = Minuit(least_squares, a = 0., b = 0.)
m.fixed["a"] = True
m.migrad()
m.hesse()
print("chi2: ", chi2, f"ndf = {ndf}", f"chi2/ndf = {chi2/ndf}", f"prob = {prob}", f"eps_mean = {eps_mean}")
print(m)
print(m.values["b"])

# write epsilon
#print(epsilon)
with open("epsilon.txt", "w+") as f:
    tmp_string = ""
    
    for j in range(len(epsilon)):
        if j == len(epsilon) - 1:
            tmp_string += str(epsilon[j])
        else:
            tmp_string += str(epsilon[j]) + " "
            
    f.write(tmp_string + "\n")
    
with open("epsilon_iteration.txt", "w+") as f:
    tmp_string = ""
    epsilon_max = max(epsilon)
    
    for j in range(len(epsilon)):
        tmp_string = str(j) + " " + str(epsilon[j]) + " " + str(eps_errors[j])
            
        f.write(tmp_string + "\n")
