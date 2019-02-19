import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy import linalg
import numpy.random
import random
from secant_functions import get_secants, SAP, get_kappa_profile
from kappa_rare_category_detection import kappa_rare_cat_detect
from mpl_toolkits.mplot3d import Axes3D

# Choose the dimension of this particular example
amb_dim = 6

# Choose number of majority points
# The total number of majority points should be divisible by amb_dim
numb_maj_unknown = 120
numb_min_unknown = 15
numb_maj_known = 400
numb_min_known = 15

# Total numbers for classes
numb_maj = numb_maj_known + numb_maj_unknown
numb_min = numb_min_known + numb_min_unknown

# Initialize array to hold majority datasets
majority = np.zeros((amb_dim,numb_maj))

# Loop through to create Gaussian distribution of points in each dimension
for i in range(numb_maj):
    rand_int = random.randint(0,amb_dim-1)
    for j in range(amb_dim):
        if (j == rand_int):
            majority[j,i] = np.random.normal(0,1)
        else:
            majority[j,i] = np.random.normal(0,.2)

# Now shuffle the points
np.random.shuffle(np.transpose(majority))

# Initial array for minority class
minority = np.zeros((amb_dim,numb_min))

# Initial some angles
for i in range(numb_min):
    theta = np.random.uniform(0,2*math.pi)
    for j in range(amb_dim):
        if (j % 2 == 0):
            minority[j,i] = .25*math.cos((j//2)*theta)
        else:
            minority[j,i] = .25*math.sin((j//2)*theta)

# Separate out points, some to be labeled, so not
known_maj = majority[:,0:numb_maj_known]
unknown_maj = majority[:,numb_maj_known:numb_maj]
known_min = minority[:,0:numb_min_known]
unknown_min = minority[:,numb_min_known:numb_min]


# Set parameters for kappa profile calculuations
lowest_dimension = 2
highest_dimension = amb_dim
iterations = 500
step_size = .03
trials = 50

# Form data matrix of all points for SVD
all_points = np.concatenate((known_maj,known_min,unknown_maj,unknown_min),axis=1)

# Perform SVD
U,S, Vh = linalg.svd(all_points)

# Get first three columns
U = U[:,1:4]

# Get projections
projection_maj = np.matmul(np.transpose(U),known_maj)
projection_min = np.matmul(np.transpose(U),known_min)

fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(projection_maj[0,1:200],projection_maj[1,1:200], projection_maj[2,1:200],c='b',marker='o',s=10)
ax.scatter(projection_min[0,:],projection_min[1,:], projection_min[2,:],c='r',marker='^',s=50)
pyplot.show()

# Run the rare category detection algorithm and get diff values
kappa_disturb = kappa_rare_cat_detect(known_maj,known_min,unknown_maj,unknown_min,1.5,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length=0.0)

# Get size of kappa diff values
kappa_disturb_dim = np.shape(kappa_disturb)

# Keep track of the number of majority minority points recorded
count_maj = 0
count_min = 0

# Loop through and get elements we need
for i in range(kappa_disturb_dim[1]):
    if (kappa_disturb[0,i] == 0):
        if (count_maj == 0):
            maj_values = [kappa_disturb[1,i]]
            count_maj = 1
        else:
            maj_values.append(kappa_disturb[1,i])
            count_maj = count_maj + 1
    else:
        print(1)
        if (count_min == 0):
            min_values = [kappa_disturb[1,i]]
            count_min = 1
        else:
            min_values.append(kappa_disturb[1,i])
            count_min = count_min + 1


# Select number of bins for historgram
bins = numpy.linspace(0, .3, 50)

pyplot.hist(maj_values, bins, alpha=0.5, label='majority class')
pyplot.hist(min_values, bins, alpha=0.5, label='rare class')
pyplot.legend(loc='upper right')
pyplot.xlabel('Shift in kappa values (L2 distance)')
pyplot.title('Amount by which majority and rare class points \n shift the kappa-profile of the rare class')
pyplot.show()



