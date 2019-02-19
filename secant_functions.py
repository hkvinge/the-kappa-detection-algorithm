import math
import numpy as np
import numpy.linalg
import numpy.random
import matplotlib.pyplot as plt
import scipy

def calculate_kappa_diff(data_points,new_point,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length):
    
    """A function which calculates the kappa-profile (for a specificed range of dimensions)
        both with and without an additional point. It then calculates the difference in these
        kappa-profiles
    
    Args:
    data_points (NumPy float array): A NumPy float array containing
        the data points as the columns of the array.
    new_point (NumPy float array): A single new point stored as a NumPy
        float array with a single column and the same number of rows as
        data_points.
    lowest_dimension (NumPy int): The smallest dimension for which the
        corresponding kappa value will be calculated.
    highest_dimension (NumPy int): The largest dimension for which the
        corresponding kappa value will be calculated.
    iterations (int): The number of iterations that the algorithm will
        run.
    step_size (float): The step size for each shift of the projection
        subspace (usually between .01-.1).
    
    Yields:
    (float): The L2 norm of the difference of the kappa-profile of
        data_points with and without new_point.
    
    """
    
    # Get ambient dimension
    dims = np.shape(data_points)
    amb_dim = dims[0]

    # Add extra point to datapoints
    new_point = new_point.reshape(amb_dim,1)
    total_points = np.concatenate((data_points,new_point),axis=1)

    # Get kappa-profiles
    kappa_class = get_kappa_profile(data_points,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length)
    kappa_total = get_kappa_profile(total_points,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length)

    # Get norm of diffance between kappa-profiles
    norm_val = np.linalg.norm(kappa_class - kappa_total)

    return norm_val


def get_kappa_profile(data_points,lowest_dimension,highest_dimension,iterations,step_size,trials,filter_length,plot=False):
    
    """A function which calculates the kappa-profile (for a specified range of dimensions) for a data set
    
    Args:
    data_points (NumPy float array): A NumPy float array containing
        the data points as the columns of the array.
    lowest_dimension (NumPy int): The smallest dimension for which the
        corresponding kappa value will be calculated.
    highest_dimension (NumPy int): The largest dimension for which the
        corresponding kappa value will be calculated.
    iterations (int): The number of iterations that the algorithm will
        run.
    step_size (float): The step size for each shift of the projection
        subspace (usually between .01-.1).
    trials (int): The number of times we calculate each kappa-profile to
        average over
    plot (True or False): If True, the kappa-profile will be plotted,
        if False, it will not. The default is False.
    
    Yields:
    (NumPy float array): A 1-dimensional NumPy float array which is the
        kappa-profile (its length is (highest_dimension - lowest_dimension).
    
    """
    

    # Initialize array to hold kappa profile values
    kappa_profile = np.zeros((highest_dimension - lowest_dimension+1,1))
    # Initialize array to hold temporary kappa profile values
    kappa_profile_temp = np.zeros((highest_dimension - lowest_dimension+1,1))
    
    # Calculate the dimension of the data array
    dims = np.shape(data_points)
    # Calculate the input dimension
    input_dim = dims[0]
    # Calculate the number of points
    numb_points = dims[1]

    for i in range(trials):

        # Get secant set for data points
        secants = get_secants(data_points,filter_length)
        
        #print(secants)

        # Generate initial projection
        proj = np.random.rand(input_dim,input_dim)
    
        # Repeatedly run SAP to generate kappa-profile
        for i in range(lowest_dimension,highest_dimension+1):

            # Truncate existing projection to desired dimension
            proj_current = proj[:,0:i]
            # Orthonormalize projection
            proj_current, r = np.linalg.qr(proj_current)

            # Run SAP for current projection dimension
            kappa_value = SAP(secants,proj_current,iterations,step_size,get_proj = False)

            # Grab kappa value
            kappa_profile_temp[i-lowest_dimension] = kappa_value

        kappa_profile = kappa_profile + kappa_profile_temp

        # If 'plot' = True, plot the kappa-profile
        if (plot == True):
            t = range(lowest_dimension,highest_dimension+1)
            plt.plot(t,kappa_profile)
            plt.show()

    kappa_profile = (1/trials)*kappa_profile

    return kappa_profile


def get_secants(data_points,filter_length):
    """Calculate the normalized secant set for an array
    of data point.
    
    Args:
        data_points (NumPy float array): A NumPy float
            array. The data points are assumed to be
            given by the columns of the array.
        filter_length (float): All secants under this length
            will not be used. If filter_length = 0, then no
            filtering will be done.
            
    Yields:
        NumPy array: The normalized secants set of data_points
            stored as a NumPy array with the secants taking the form
            of columns.
            
    """
    
    # Calculate dimension of data array
    dims = np.shape(data_points)
    # Ambient dimension of the data
    input_dim = dims[0]
    # Number of data points
    numb_points = dims[1]

    # Initialize array to hold secants
    secants = np.zeros((input_dim,int((numb_points-1)*numb_points/2)))

    # Initialize counter to count secants
    count = 0
    
    if filter_length == 0:
        for i in range(numb_points):
            for j in range(numb_points):
                if i < j:
                    secant = data_points[:,i] - data_points[:,j]
                    norm = np.linalg.norm(secant)
                    secant = (1/norm)*secant
                    secants[:,count] = secant
                    count = count + 1
    else:
        for i in range(numb_points):
            for j in range(numb_points):
                if i < j:
                    secant = data_points[:,i] - data_points[:,j]
                    norm = np.linalg.norm(secant)
                    if (norm > filter_length):
                        secant = (1/norm)*secant
                        secants[:,count] = secant
                        count = count + 1
        # Only use nonzero rows of the secant matrix
        secants = secants[:,0:count]

    # Return array of secants
    return secants

def SAP(secants,proj,iterations,step_size,get_proj = True):
    """Runs the SAP algorithm on a data set
        
    Args:
        secants (NumPy float array): A NumPy float array containing
            the secants of the data set as columns.
        proj (NumPy float array): A NumPy float array whose columns are
            a set of orthonormal vectors that span the projection subspace.
        iterations (int): The number of iterations that the algorithm will
            run.
        step_size (float): The step size for each shift of the projection
            subspace (usually between .01-.1).
        
    Yields:
        (NumPy float array): A NumPy float array whose columns give a set
            of orthonormal vectors that span the projection subspace.
        
    """
    
    # Initialize array to record shortest secants for kappa-profile
    worst_proj_secants_record = np.zeros((iterations,1))

    # Run the SAP algorithm for some number of iterations
    for i in range(iterations):

        # The projection of each secant
        secant_projections = np.matmul(np.transpose(proj),secants)

        # Calculate norm of each column
        secant_norms = np.linalg.norm(secant_projections,axis=0)

        # Calculate minimum index
        index_min = np.argmin(secant_norms)
        
        if (get_proj == True):
            # Print smallest projected secant norm
            print("The smallest projected secant norm is:  " + str(secant_norms[index_min]))

        worst_proj_secants_record[i] = secant_norms[index_min]
    
        # Grab secant that is most diminished under projection
        most_diminished_secant = secants[:,index_min]

        # Calculate projection of most diminished
        proj_most_diminished_secant = np.matmul(proj,secant_projections[:,index_min])

        # Find largest coefficient for projected secant
        largest_coefficient = np.argmax(np.absolute(secant_projections[:,index_min]))
    
        # Switch columns
        proj[:,largest_coefficient] = proj[:,0]
        proj[:,0] = proj_most_diminished_secant

        # Apply modified Graham-Schmidt QR decomposition
        proj, r = np.linalg.qr(proj)

        # Shift projection
        proj[:,0] = (1-step_size)*proj_most_diminished_secant + step_size*(most_diminished_secant - proj_most_diminished_secant)
 
        # Normalize shift
        norm = np.linalg.norm(proj[:,0])
        if (norm == 0):
            print(norm)
            print(proj_most_diminished_secant)
            print(most_diminished_secant)
        proj[:,0] = (1/norm)*proj[:,0]

    # X-values for plotting of kappa-profile
    t = range(iterations)
    
    if (get_proj == True):
        # Plot length of shortest projected secant as a function of iteration (to check for convergence).
        plt.plot(t,worst_proj_secants_record)
        plt.show()

    # If get_proj is equal to true then return the actual projection
    if get_proj == True:

        return proj

    # Otherwise, return the kappa value for this dimension
    else:

        return worst_proj_secants_record[iterations-1]


        
