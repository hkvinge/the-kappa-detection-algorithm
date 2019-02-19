import math
import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import numpy.linalg
from secant_functions import get_secants, SAP, get_kappa_profile, calculate_kappa_diff


def kappa_rare_cat_detect(known_maj,known_minority,unknown_maj,unknown_minority,multiple,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length):
    
    """A function which calculates the kappa-profile (for a specificed range of dimensions)
        both with and without an additional point. It then calculates the difference in these
        kappa-profiles
        
    Args:
    known_maj (NumPy float array): A NumPy float array containing
        the labeled majority points as the columns of the array.
    known_minority (NumPy float array): A NumPy float array containing
        the labeled rare class points as the columns of the array.
    known_maj (NumPy float array): A NumPy float array containing
        the unlabeled majority points as the columns of the array.
    known_minority (NumPy float array): A NumPy float array containing
        the unlabeled rare class points as the columns of the array.
    new_point (NumPy float array): A single new point stored as a NumPy
        float array with a single column and the same number of rows as
        data_points.
    multiple (float): A float that controls the size of the hyperball
        from which we take test points when radius = 0.0 (see end of
        description).
    lowest_dimension (NumPy int): The smallest dimension for which the
        corresponding kappa value will be calculated.
    highest_dimension (NumPy int): The largest dimension for which the
        corresponding kappa value will be calculated.
        iterations (int): The number of iterations that the algorithm will
        run.
    step_size (float): The step size for each shift of the projection
        subspace (usually between .01-.1).
    trials (int): Number of times to calculate and average over each
        kappa-profile calculation.
    filter_length (float): All secants under this length
        will not be used. If filter_length = 0, then no
        filtering will be done.
    radius (float): The radius of the hyperball which will be used
        to choose the points that the algorithm will be applied to.
        If radius = 0.0 then the radius is equal is equal to
        
        multiple*max(distance from labeled poin to centroid)
        
        
    Yields:
    (float): The L2 norm of the difference of the kappa-profile of
        data_points with and without new_point.
        
    """
    
    # Get dimensions of points
    unknown_maj_dim = np.shape(unknown_maj)
    unknown_min_dim = np.shape(unknown_minority)

    # Mix two known classes
    unknown_classes = np.concatenate((unknown_maj,unknown_minority), axis = 1)
    # Keep key to record which point is which
    maj_key = np.zeros(unknown_maj_dim[1])
    min_key = np.ones(unknown_min_dim[1])
    key = np.concatenate((maj_key,min_key))
    
    # Get numbers of labeled classes
    dim_maj_known = np.shape(known_maj)
    dim_min_known = np.shape(known_minority)
    
    # Get the ambient dimension
    amb_dim = dim_maj_known[0]
    
    # Get numbers of labeled classes
    dim_maj_unknown = np.shape(unknown_maj)
    dim_min_unknown = np.shape(unknown_minority)

    # Get number of unknown classes
    dim_unknown = np.shape(unknown_classes)

    # Calculate the centroid of minority class
    centroid = (1/dim_min_known[1])*np.sum(known_minority,axis=1)
    
    # Max distance of known minority classes from centroid
    max_distance = 0
    
    # Calculate radius of ball containing minority points
    for i in range(dim_min_known[1]):
        if (np.linalg.norm(centroid - known_minority[:,i]) > max_distance):
            max_distance = np.linalg.norm(centroid - known_minority[:,i])

    numb_points_check = 0

    max_distance = .3

    # Find points to test
    for i in range(dim_unknown[1]):
        # If distance is close
        if (np.linalg.norm(unknown_classes[:,i]-centroid) < max_distance*multiple):
            # Initialize and reshape the point to be considered
            unknown_point = unknown_classes[:,i]
            unknown_point = unknown_point.reshape(amb_dim,1)
            if (numb_points_check == 0):
                unknowns_to_check = unknown_point
                unknowns_to_check_key = [key[i]]
                numb_points_check = numb_points_check + 1
            else:
                unknowns_to_check = np.concatenate((unknowns_to_check,unknown_point),axis=1)
                unknowns_to_check_key = np.concatenate((unknowns_to_check_key,[key[i]]))
                numb_points_check = numb_points_check + 1

    # Create array to store disturbance in kappa-profiles
    kappa_disturb = np.zeros((2,numb_points_check))

    # Move through unknown points that need to be checked calculate the extent to
    # which they perturb the minority kappa-profile
    for i in range(numb_points_check):
        a = calculate_kappa_diff(known_minority,unknowns_to_check[:,i],lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length)
        if (unknowns_to_check_key[i] == 0):
            print("Majority point " + str(i) + ": " + str(a))
        else:
            print("Minority point " + str(i) + ": " + str(a))
        kappa_disturb[0,i] = unknowns_to_check_key[i]
        kappa_disturb[1,i] = a

    return kappa_disturb




