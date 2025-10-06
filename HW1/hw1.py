###### Your ID ######
# ID1: 213226269
# ID2: 807633
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X, y):
    """
    Perform Standardization on the features and true labels.
    
    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    
    Returns:
    - X: The Standardized input data.
    - y: The Standardized true labels.
    """
    # Get number of instances
    # For X: standardize along axis=0 (features)
   
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    standardized_X = (X - X_mean) / X_std
    
    # For y: standardize all values
    y_mean = np.mean(y)
    y_std = np.std(y)
    standardized_y = (y - y_mean) / y_std


    return standardized_X, standardized_y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    return np.column_stack((np.ones(X.shape[0]), X))

def compute_loss(X, y, theta): #this is different
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the loss associated with the current set of parameters (single number).
    """
        
    # Compute predictions using dot product
    y_hat =  X @ theta  # equivalent to np.dot(X, theta) 
    
    # Compute MSE loss
    J = (1/2) * np.mean((y_hat - y)**2)
    
    return J

def gradient_descent(X, y, theta, eta, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    J_history = [] # Use a python list to save the loss value in every iteration
    
    X_T = X.T
    SCALING_FACTOR = eta / X.shape[0]
    theta = theta.copy()

    for t in range(num_iters): 

        error = (X @ theta) - y

        gradient = X_T @ error 
        
        theta = theta - (SCALING_FACTOR) * (gradient)

        loss = compute_loss(X, y, theta)

        J_history.append(loss) # Save the loss value for this iteration


    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []

    # Compute X^T (transpose of X)
    X_transpose = X.T
    
    # Compute X^T * X
    X_transpose_X = X_transpose @ X
    
    # Compute (X^T * X)^(-1) using np.linalg.inv
    X_transpose_X_inv = np.linalg.inv(X_transpose_X)
    
    # Compute (X^T * X)^(-1) * X^T
    pinv = X_transpose_X_inv @ X_transpose
    
    # Compute the optimal parameters: θ = (X^T * X)^(-1) * X^T * y
    pinv_theta = pinv @ y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def gradient_descent_stop_condition(X, y, theta, eta, max_iter, epsilon=1e-8):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than epsilon. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - max_iter: The maximum number of iterations.
    - epsilon: The threshold for the improvement of the loss value.
    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    J_history = []
    X_T = X.T
    SCALING_FACTOR = eta / X.shape[0]
    theta = theta.copy()
    
    for t in range(max_iter):

        error = (X @ theta) - y

        gradient = X_T @ error 
        
        theta = theta - (SCALING_FACTOR) * (gradient)

        loss = compute_loss(X, y, theta)

        if J_history and (J_history[-1] - loss) < epsilon:
            J_history.append(loss)
            break

        J_history.append(loss) # Save the loss value for this iteration

        
        # Check convergence
        if t > 0 and abs(J_history[t-1] - J_history[t]) < epsilon:
            break
    
    return theta, J_history

def find_best_learning_rate(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of eta and train a model using 
    the training dataset. Maintain a python dictionary with eta as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - eta_dict: A python dictionary - {eta_value : validation_loss}
    """
    
    etas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    eta_dict = {} # {eta_value: validation_loss}
    
    # Initial parameters (starting with zeros)
    np.random.seed(42)
    initial_theta = np.random.random(X_train.shape[1])  # initialize n random theta values in [0,1) 
    #  initial_theta = np.zeros(X_train.shape[1])  # Use zeros instead of random values might nead to differnt results


    
    # For each eta value
    for eta in etas:
        try:
            # Train the model using gradient descent
            theta, _ = gradient_descent(
                X_train, 
                y_train, 
                initial_theta, 
                eta, 
                iterations)

            # Check for NaNs in theta
            if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
                # Skip this eta value
                eta_dict[eta] = float('inf')
                continue
                    
            # Compute validation loss using the trained model
            validation_loss = compute_loss(X_val, y_val, theta)
            
            # Check if the loss is valid
            if np.isnan(validation_loss) or np.isinf(validation_loss):
                eta_dict[eta] = float('inf')
            else:
                # Store in dictionary
                eta_dict[eta] = validation_loss

        except Exception as e:
            # If any errors occur, skip this eta value
            print(f"Error with eta={eta}: {e}")
            eta_dict[eta] = float('inf')
    
    return eta_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_eta: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    n = X_train.shape[1]
    np.random.seed(42)
    theta_rand = np.random.random(6)  # 1 bias theta and 5 feature thetas
    
    
    while len(selected_features) < 5 and len(selected_features) < n:

        feature_loss_dict = {}
        temp_selected_features = selected_features.copy()  # copying updated best features list for edit safety

        for i in range(n):
            if i not in temp_selected_features:

                temp_selected_features.append(i)
                curr_theta = theta_rand[:len(temp_selected_features) + 1]  # current num of selected features + bias

                # create a sub matrix of selected features columns, append bias column
                X_train_curr = apply_bias_trick(X_train[:, temp_selected_features])
                X_val_curr = apply_bias_trick(X_val[:, temp_selected_features])

                best_theta, _ = gradient_descent_stop_condition(
                    X_train_curr,
                    y_train,
                    curr_theta,
                    best_eta,
                    iterations
                )

                loss = compute_loss(X_val_curr, y_val, best_theta)
                feature_loss_dict[i] = loss
                temp_selected_features.remove(i)
                
                

        min_loss = min(feature_loss_dict, key=feature_loss_dict.get)
        selected_features.append(min_loss)
         

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (n instances over p features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    cols = df_poly.columns

    squared_features = pd.DataFrame({f"{col}^2": df_poly[col] ** 2 for col in cols},index=df.index)
    intersection_features = pd.DataFrame({f"{col1}*{col2}": df_poly[col1]*df_poly[col2] 
                                 for i, col1 in enumerate(cols) for col2 in cols[i+1:]}, index=df.index)

    df_poly = pd.concat([df_poly, squared_features, intersection_features], axis=1)


    return df_poly