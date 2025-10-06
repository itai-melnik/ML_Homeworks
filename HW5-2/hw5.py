import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should randomly sample `k` different pixels from the input
    image as the initial centroids for the K-means algorithm.
    The selected `k` pixels should be sampled uniformly from all sets
    of `k` pixels in the image.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    num_pixels = X.shape[0]
    # Randomly choose k unique indices
    indices = np.random.choice(num_pixels, size=k, replace=False)
    # Select the corresponding pixels
    centroids = X[indices]
    return centroids
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################    
    

def l_p_dist_from_centroids(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids of shape (k, 3)
    The parameter p for the L_p norm distance measure.

    Output: numpy array of shape `(k, num_pixels)`,
    in which entry [j,i] holds the distance of the i-th pixel from the j-th centroid.
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
        # Expand centroids and X to broadcast differences
    diffs = np.abs(centroids[:, np.newaxis, :] - X[np.newaxis, :, :])
    powered = diffs ** p
    summed = np.sum(powered, axis=2)
    distances = summed ** (1/p)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100, epsilon=1e-8):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the L_p distance measure.
    - max_iter: the maximum number of iterations to perform.
    - epsilon: the threshold for convergence.

    Outputs:
    - The final centroids as a numpy array.
    - The final assignment of all pixels to the closest centroids as a numpy array.
    - The final WCS as a float.
    """
    # So this is the main k-means loop! I start with random centroids, then for each iteration, I assign every pixel to the closest centroid, update the centroids to be the mean of their assigned pixels, and check if they've stopped moving (converged). If a cluster ends up empty, I just pick a random pixel for its centroid. I also calculate the WCS (within-cluster sum) at the end. If the centroids barely move, we stop early!
    cluster_assignments = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for iteration in range(max_iter):
        # Compute distances from centroids to all points
        distances = l_p_dist_from_centroids(X, centroids, p)
        # Assign each pixel to closest centroid
        cluster_assignments = np.argmin(distances, axis=0)

        # Compute new centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            assigned_points = X[cluster_assignments == j]
            if len(assigned_points) > 0:
                new_centroids[j] = assigned_points.mean(axis=0)
            else:
                # If a cluster is empty, reinitialize to a random point
                new_centroids[j] = X[np.random.choice(X.shape[0])]

        # Check convergence
        centroid_shifts = np.linalg.norm(new_centroids - centroids, axis=1)
        if np.max(centroid_shifts) < epsilon:
            break

        centroids = new_centroids

    # Compute final WCS
    final_distances = l_p_dist_from_centroids(X, centroids, p)
    distances_to_assigned = final_distances[cluster_assignments, np.arange(X.shape[0])]
    WCS = np.sum(distances_to_assigned ** p)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, cluster_assignments, WCS

def kmeans_pp(X, k, p ,max_iter=100, epsilon=1e-8):
    """
    The kmeans algorithm with alternative centroid initalization.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the L_p distance measure.
    - max_iter: the maximum number of iterations to perform.
    - epsilon: the threshold for convergence.

    Outputs:
    - The final centroids as a numpy array.
    - The final assignment of all pixels to the closest centroids as a numpy array.
    - The final WCS as a float.
     """
    cluster_assignments = None
    centroids = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # This is k-means++! Instead of picking all centroids randomly, I pick the first one randomly, then each next one is picked with probability proportional to how far it is from the closest already-picked centroid (so we spread them out). Then I do the usual k-means loop. If a cluster is empty, I just pick a random pixel for its centroid. WCS is calculated at the end too!
    num_pixels = X.shape[0]
    centroids = []
    first_idx = np.random.choice(num_pixels)
    centroids.append(X[first_idx])
    for _ in range(1, k):
        # Compute distances from all points to the existing centroids
        current_centroids = np.vstack(centroids)
        distances = l_p_dist_from_centroids(X, current_centroids, p)
        min_distances = np.min(distances, axis=0)
        probs = min_distances**2
        probs_sum = np.sum(probs)
        if probs_sum == 0:
            # If all distances are zero, pick random
            next_idx = np.random.choice(num_pixels)
        else:
            probs /= probs_sum
            next_idx = np.random.choice(num_pixels, p=probs)
        centroids.append(X[next_idx])
    centroids = np.vstack(centroids)
    for iteration in range(max_iter):
        distances = l_p_dist_from_centroids(X, centroids, p)
        cluster_assignments = np.argmin(distances, axis=0)
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            assigned_points = X[cluster_assignments == j]
            if len(assigned_points) > 0:
                new_centroids[j] = assigned_points.mean(axis=0)
            else:
                new_centroids[j] = X[np.random.choice(num_pixels)]
        centroid_shifts = np.linalg.norm(new_centroids - centroids, axis=1)
        if np.max(centroid_shifts) < epsilon:
            break
        centroids = new_centroids
    final_distances = l_p_dist_from_centroids(X, centroids, p)
    distances_to_assigned = final_distances[cluster_assignments, np.arange(num_pixels)]
    WCS = np.sum(distances_to_assigned ** p)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, cluster_assignments, WCS
