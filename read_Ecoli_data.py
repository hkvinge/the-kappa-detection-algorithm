import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy import linalg
from secant_functions import get_secants, SAP, get_kappa_profile
from kappa_rare_category_detection import kappa_rare_cat_detect
from mpl_toolkits.mplot3d import Axes3D

##### Read in data ###########

# Get number of lines in data
numb_points = sum(1 for line in open('datasets/Ecoli/ecoli.csv'))

# Import dataset
ecoli_file = open('datasets/Ecoli/ecoli.csv','r')

# Set numpy array to store points
data_points = np.zeros((numb_points,7))
# Set array for labels
labels = list();

# Set variable to count the number of lines
line_count = 0

# Iterate through and grab data and labels
for i in ecoli_file:
    x = i.split()
    for j in range(1,8):
        data_points[line_count,j-1] = float(x[j])
    labels.append(x[8])
    line_count = line_count+1

# Transpose data matrix so that columns correspond
# to data points
data_points = np.transpose(data_points)

##### Get the kappa-profile for the points ######

# Set parameters
lowest_dimension = 2
highest_dimension = 7
iterations = 500
step_size = .03
trials = 10
filter_length = 0

# Call kappa_profile function
kappa_profile = get_kappa_profile(data_points,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length,plot=False)

##### Separate points into different classes ####

# Get set of distinct labels
labels_set = set(labels)
labels_set = sorted(labels_set)

# Record
count = 0

# List to store indices for different classes
indices = list()

# Dictionary to store collections of points for each class
dict = {}

for x in labels_set:
    indices.append([i for i in range(numb_points) if labels[i] == x])
    dict[x] = data_points[:,indices[count]]
    count = count + 1

###### Calculate kappa-profile for each class ########

# A dictionary to store the kappa-profiles
dict_kappa = {}
count = 0

for x in labels_set:
    kappa_profile = get_kappa_profile(dict[x],lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length,plot=False)
    print("The class " + x + " has " + str(len(indices[count])))
    dict_kappa[x] = kappa_profile
    count = count + 1

# Plot these kappa profiles

# Dimension ranges
t = range(lowest_dimension,highest_dimension+1)

# Initialize counter
count = 0

# Loop through and plot kappa profiles
for x in labels_set:
    plt.plot(t,dict_kappa[x],label=x + " " + str(len(indices[count])))
    count = count + 1

# Add legend and show plot
plt.legend()
plt.show()

############### Separate data out into majority and rare classes ###########

# Set number of labeled majority points
numb_maj_known = 200
# Set number of labeled minority points
numb_min_known = 10

# Initialize counter
count = 0

# Construct set of majority points
for x in labels_set:
    if (x != 'om'):
        if (count ==0):
            majority = dict[x]
            count = 1
        else:
            majority = np.concatenate((majority,dict[x]), axis=1)
            count = count + 1
    else:
        rare_class = dict[x]

# Shuffle points to get different orderings.
np.random.shuffle(np.transpose(majority))
np.random.shuffle(np.transpose(rare_class))

# Separate into known and unknown points
known_maj = majority[:,1:numb_maj_known]
unknown_maj = majority[:,numb_maj_known:]
known_rare = rare_class[:,1:numb_min_known]
unknown_rare = rare_class[:,numb_min_known:]

# Form data matrix of all points for SVD
all_points = np.concatenate((majority,rare_class),axis=1)

# Perform SVD
U,S, Vh = linalg.svd(all_points)

# Get first three columns
U = U[:,1:4]

# Get projections
projection_maj = np.matmul(np.transpose(U),majority)
projection_rare = np.matmul(np.transpose(U),rare_class)

fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(projection_maj[0,:],projection_maj[1,:], projection_maj[2,:],c='b',marker='o',s=10)
ax.scatter(projection_rare[0,:],projection_rare[1,:], projection_rare[2,:],c='r',marker='^',s=50)
pyplot.show()

# Choose a new filter length for this this calculation
filter_length = .4

# Apply the kappa-detection algorithm
kappa_disturb = kappa_rare_cat_detect(known_maj,known_rare,unknown_maj,unknown_rare,2,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length)

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
        if (count_min == 0):
            min_values = [kappa_disturb[1,i]]
            count_min = 1
        else:
            min_values.append(kappa_disturb[1,i])
            count_min = count_min + 1


# Select number of bins for historgram
bins = np.linspace(0, .3, 50)

pyplot.hist(maj_values, bins, alpha=0.5, label='majority class')
pyplot.hist(min_values, bins, alpha=0.5, label='rare class')
pyplot.legend(loc='upper right')
pyplot.xlabel('Shift in kappa values (L2 distance)')
pyplot.show()
