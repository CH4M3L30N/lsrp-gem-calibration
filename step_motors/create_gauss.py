import numpy as np
import random

place_x = 32
place_y = 20

def create_2d_gaussian(size_x, size_y, sigma=1.0):
    # Create a coordinate grid from -1 to 1
    x, y = np.meshgrid(np.linspace(-1, 1, size_x), np.linspace(-1, 1, size_y))
    
    # Calculate squared distance from center
    d_sq = x**2 + y**2
    
    # Apply Gaussian formula
    gauss = np.exp(-(d_sq / (2.0 * sigma**2)))
    
    # Convert to standard Python 2D list if needed
    return gauss.tolist()
    
def gaussian(x, y, mean_x, mean_y, sigma_x, sigma_y, size_x, size_y, count_num):
    # Calculate squared distance from center
    d_sqx = (x - mean_x)**2
    d_sqy = (y - mean_y)**2
    
    # Apply Gaussian formula
    gauss = count_num * np.exp(-(d_sqx/(2.0 * sigma_x**2) + d_sqy/(2.0 * sigma_y**2))) / ((2 * np.pi) ** (0.5) * sigma_x * sigma_y)# * 4 / (size_x * size_y)
    
    # Convert to standard Python 2D list if needed
    return gauss

# Generate a 20x10 Gaussian grid
#gauss_list = create_2d_gaussian(ray_x, ray_y)

cx = -10
cy = -5
file_max_num = (place_x - cx * 2) * (place_y - cy * 2)
centers = []
stepx = 1
epsilon = []
epsilon_mean = 0

for i in range(place_y):
    epsilon.append([])
    for j in range(place_x):
        random_eps = random.uniform(0.5, 1.0)
        epsilon[i].append(random_eps)
        epsilon_mean += random_eps
        #epsilon.append(1.0)

epsilon_mean /= len(epsilon) * len(epsilon[0])

for i in range(place_y):
    for j in range(place_x):
        epsilon[i][j] /= epsilon_mean

print(epsilon)

with open("epsilon_real.txt", "w+") as f:
    tmp_string = ""
    
    for i in range(len(epsilon)):
        for j in range(len(epsilon[0])):
            tmp_string = str(j) + " " + str(i) + " " + str(epsilon[i][j])
                
            f.write(tmp_string + "\n")

with open("data/centers.txt", "w+") as f:
    a = 0

for n in range(file_max_num):
    # create empty data
    data = []

    for i in range(place_y):
        tmp_data = []

        for j in range(place_x):
            tmp_data.append(0)
    
        data.append(tmp_data)

    # add gaussian on data
    
    for i in range(place_y):
        for j in range(place_x):
            mu = gaussian(j, i, cx, cy, 5.0, 1.0, place_x, place_y, 1000000)
            data[i][j] = np.random.poisson(mu) * epsilon[i][j]
    #print(data[0])
           
    with open("data/data" + str(n) + ".txt", "w+") as f:
        tmp_string = ""
        
        for i in range(place_y):
            for j in range(place_x):
                tmp_string = str(j) + " " + str(i) + " " + str(data[i][j])
                
                f.write(tmp_string + "\n")
    
    #print(cx, cy)
    centers.append([cx,cy])
    
    cx += stepx
    if cx > 41:
        cx = -10
        cy += 1
    if cy > 25:
        break
        
#print(gauss_list)
with open("data/centers.txt", "w+") as f:
    for i in range(len(centers)):
        f.write(str(centers[i][0]) + " " + str(centers[i][1]) + "\n")
