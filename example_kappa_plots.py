import math
import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import random
from secant_functions import get_secants, SAP, get_kappa_profile



numb_points = 200

#----------------------------------------------------------
# Create points for torus
#----------------------------------------------------------

random_points_2D = np.random.rand(2,numb_points)
random_points_2D = 2*math.pi*random_points_2D
points_torus = np.zeros((3,numb_points))
embedd_torus = np.zeros((10,numb_points))

# Major radius of torus
R = 2
# Minor radius of torus
r = .2
for i in range(numb_points):
    a = random_points_2D[0,i]
    b = random_points_2D[1,i]
    v = [(R + r*math.cos(a))*math.cos(b),(R + r*math.cos(a))*math.sin(b),r*math.sin(a)]
    v = np.unique(v)
    points_torus[:,i] = v
    a = v[0]
    b = v[1]
    c = v[2]
    w = [a,b,c,a**2,b**2,c**2,a*b*c,b*c,a**2-b**2,math.cos(b*c)]
    w = np.unique(w)
    embedd_torus[:,i] = w.reshape(10)


#----------------------------------------------------------
# Create points for real projective space
#----------------------------------------------------------

random_points_3D = np.random.rand(3,numb_points)
points_proj_space = np.zeros((4,numb_points))
embedd_proj_space = np.zeros((10,numb_points))

# Normalize points to put them on 3-sphere
for i in range(numb_points):
    # Calculate norm of columns
    norm = np.linalg.norm(random_points_3D[:,i])
    # Normalize each column
    random_points_3D[:,i] = (1/norm)*random_points_3D[:,i]
    # Embedding into 4-D space
    a = random_points_3D[0,i]
    b = random_points_3D[1,i]
    c = random_points_3D[2,i]
    # 4-space embedding
    v = [a*b,a*c,b**2-c**2,2*b*c]
    a = v[0]
    b = v[1]
    c = v[2]
    d = v[3]
    w = [a,b,c,d,a**2,b**2,c**2,d**2,a*b*c,c*d]
    w = np.unique(w)
    embedd_proj_space[:,i] = w.reshape(10)

#---------------------------------------------------------
# Create points for 3-sphere
#---------------------------------------------------------

random_points_3D = np.random.rand(4,numb_points)
points_3_sphere = np.zeros((4,numb_points))
embedd_3_sphere = np.zeros((10,numb_points))

# Normalize points to put them on 3-sphere
for i in range(numb_points):
    # Calculate norm of columns
    norm = np.linalg.norm(random_points_3D[:,i])
    # Normalize each column
    points_3_sphere[:,i] = (1/norm)*random_points_3D[:,i]
    # Embedding into 10-D space
    a = points_3_sphere[0,i]
    b = points_3_sphere[1,i]
    c = points_3_sphere[2,i]
    d = points_3_sphere[3,i]
    v = [a,b,c,d,a**2,b**2,c**2,d**2,a*b*c,c*d]
    v = np.unique(v)
    embedd_3_sphere[:,i] = v.reshape(10)

#----------------------------------------------------------
# Create points from Gaussian
#----------------------------------------------------------

# Draw normal distributed points from R^{10}
random_gaussian = np.random.randn(10,numb_points)

#----------------------------------------------------------
# Calculate kappa profiles
#----------------------------------------------------------

# Set parameters for kappa profile calculuations
lowest_dimension = 2
highest_dimension = 10
iterations = 500
step_size = .03
trials = 10

# Calculate kappa-profiles
kappa_torus = get_kappa_profile(embedd_torus,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length=0.0,plot=False)
kappa_proj_space = get_kappa_profile(embedd_proj_space,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length=0.0,plot=False)
kappa_sphere = get_kappa_profile(embedd_3_sphere,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length=0.0,plot=False)
kappa_gaussian = get_kappa_profile(random_gaussian,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length=0.0,plot=False)

# Set x-values for kappa profile plot
x = range(2,11)

# plot kappa-profile
plt.plot(x,kappa_torus,label='torus')
plt.plot(x,kappa_proj_space,label='real projective space')
plt.plot(x,kappa_sphere,label='3-sphere')
plt.plot(x,kappa_gaussian,label='multivariate Gaussian')
plt.xlabel("Projection dimension")
plt.ylabel("kappa")
plt.legend()
plt.show()
