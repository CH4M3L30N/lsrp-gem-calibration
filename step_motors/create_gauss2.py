import numpy as np
import random

#chi2 for count_num, sigma, epsilon_real, correlation
place_x = 32

def create_2d_gaussian(size_x, count_num, sigma=1.0):
    # Create a coordinate grid from -1 to 1
    x = np.linspace(-1, 1, size_x)
    
    # Calculate squared distance from center
    d_sq = x**2
    
    # Apply Gaussian formula
    gauss = count_num * np.exp(-(d_sq / (2.0 * sigma**2))) / ((2 * np.pi) ** (0.5) * sigma) * 2 / size_x
    
    # Convert to standard Python 2D list if needed
    return gauss.tolist()
    
def gaussian(x, mean, sigma, size_x, count_num):
    # Calculate squared distance from center
    d_sq = (x - mean)**2
    
    # Apply Gaussian formula
    gauss = count_num * np.exp(-(d_sq / (2.0 * sigma**2))) / ((2 * np.pi) ** (0.5) * sigma) * 2 / size_x
    
    # Convert to standard Python 2D list if needed
    return gauss

# Generate a 20x10 Gaussian grid
gauss_list = create_2d_gaussian(size_x = 22, count_num = 1000000, sigma = 5.0)
signal_list = []
signal_sizex = 20
for i in range (signal_sizex):
    signal_list.append(0)

num = random.randint(1, 5)

if num % 2 == 0:
    for i in range(num):
        if i % 2 == 0:
            for j in range(signal_sizex):
                signal_list[j] = gauss_list[j]
        else:
            for j in range(signal_sizex):
                signal_list[j] = gauss_list[j + 2]
else:
    for i in range(num):
        if i < num - 1:
            if i % 2 == 0:
                for j in range(signal_sizex):
                    signal_list[j] = gauss_list[j]
            else:
                for j in range(signal_sizex):
                    signal_list[j] = gauss_list[j + 2]
        else:
            for j in range(signal_sizex):
                    signal_list[j] = gauss_list[j + 1]

file_max_num = 53
cx = -10
centers = []
stepx = 1
epsilon = []
epsilon_mean = 0

for i in range(32):
    random_eps = random.uniform(0.5, 1.0)
    epsilon.append(random_eps)
    epsilon_mean += random_eps
    #epsilon.append(1.0)

epsilon_mean /= len(epsilon)

for i in range(32):
    epsilon[i] /= epsilon_mean
    #epsilon.append(1.0)

print(epsilon)

with open("epsilon_real.txt", "w+") as f:
    tmp_string = ""
    
    for j in range(len(epsilon)):
        tmp_string = str(j) + " " + str(epsilon[j])
            
        f.write(tmp_string + "\n")

with open("data/centers.txt", "w+") as f:
    a = 0

for n in range(file_max_num):
    # create empty data
    data = []

    for i in range(place_x):
        data.append(0)

    # add gaussian on data
    #for i in range(len(data)):
    #    if i >= cx and i < cx + len(gauss_list) // 2:
    #        data[i] = int(gauss_list[len(gauss_list) // 2 + i - cx] * epsilon[i])
    #        #tmp_diff = data[i]**(0.5)
    #        tmp_diff = np.random.poisson(data[i])#, data[i])
    #        #print(tmp_diff)
    #        data[i] += int(tmp_diff)
    #    elif i < cx and i > cx - len(gauss_list) // 2 - 1:
    #        data[i] = int(gauss_list[10 - cx + i] * epsilon[i])
    #        #tmp_diff = data[i]**(0.5)
    #        tmp_diff = np.random.poisson(data[i])#, data[i])
    #        #print(tmp_diff)
    #        data[i] += int(tmp_diff)
            
    for i in range(len(data)):
        mu = gaussian(i, cx, 5.0, place_x, 1000000) * epsilon[i]
        data[i] = np.random.poisson(mu)
           
    
    #tmp_data = data
    #for j in range(cx):
    #    data[j] = tmp_data[cx - j - 1]           
    
    with open("data/data" + str(n) + ".txt", "w+") as f:
        tmp_string = ""
        
        for j in range(len(data)):
            tmp_string = str(j) + " " + str(data[j])
            
            f.write(tmp_string + "\n")
    
    #print(cx)
    centers.append([cx])
    
    cx += stepx
        
#print(gauss_list)
with open("data/centers.txt", "w+") as f:
    for i in range(len(centers)):
        f.write(str(centers[i][0]) + "\n")
