import numpy as np

def add_bias_term(X):
    """
    Add a bias term to each sample of the input data.
    """

    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################
    
    return np.column_stack((np.ones(X.shape[0]), X))

class LogisticRegressionGD():
    """
    Logistic Regression Classifier.

    Fields:
    -------
    w_ : array-like, shape = [n_features]
      Weights vector, where n_features is the number of features.
    eta : float
      Learning rate (between 0.0 and 1.0)
    max_iter : int
      Maximum number of iterations for gradient descent
    eps : float
      minimum change in the BCE loss to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """
    
    def __init__(self, learning_rate=0.0001, max_iter=10000, eps=0.000001, random_state=1):
       
        # Initialize the weights vector with small random values
        self.random_state = random_state
        self.w_ = np.nan # This is an array of weights 
        self.learning_rate = learning_rate #Basically the eta
        self.max_iter = max_iter
        self.eps = eps
        self.class_names = None

    def _sigmoid(self, z):
        """Sigmoid activation function"""

        z = np.clip(z, -500, 500) 

        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        """
        Return the predicted probabilities of the instances for the positive class (class 1)

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors, where n_samples is the number of samples and
          n_features is the number of features.

        Returns
        -------
        y_pred_prob : array-like, shape = [n_examples]
          Predicted probabilities (for class 1) for all the instances
        """
        class_1_prob = np.nan * np.ones(X.shape[0])

        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################

        # X = add_bias_term(X) As The bias is already applied before
        class_1_prob = self._sigmoid(X.dot(self.w_))
          
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return class_1_prob
        

    def predict(self, X, threshold=0.5):
        """
        Return the predicted class label according to the threshold

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors, where n_samples is the number of samples and
          n_features is the number of features.
        threshold : float, optional
          Threshold for the predicted class label.
          Predict class 1 if the probability is greater than or equal to the threshold and 0 otherwise.
          Default is 0.5. 
        """
  
    
        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################
        probabilities = self.predict_proba(X)
        binary_pred = (probabilities >= threshold).astype(int) # Makes an array of 1 and 0 
        
        # Convert 0/1 predictions back to original class names
        y_pred = np.where(binary_pred == 0, self.class_names[0], self.class_names[1]) # Here turns it back into the actual classnames
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return y_pred

    
    def BCE_loss(self, X, y):
        """
        Calculate the BCE loss (not needed for training)

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Class labels. 

        Returns
        -------
        BCE_loss : float
          The BCE loss.
          Make sure to normalize the BCE loss by the number of samples.
        """
        
        y_01 = np.where(y == self.class_names[0], 0, 1) # represents the class 0/1 labels
        loss = None
        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################
        # X_with_bias = add_bias_term(X)
        probs = self._sigmoid(X.dot(self.w_))

        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps) #Prevents log(0) from occuring

        loss = -np.mean(y_01*np.log(probs) + (1 - y_01)*np.log(1-probs))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return loss

    def fit(self, X, y):
        """ 
        Fit training data by minimizing the BCE loss using gradient descent.
        Updates the weight vector (field of the object) in each iteration using gradient descent.
        The gradient should correspond to the BCE loss normalized by the number of samples.
        Stop the function when the difference between the previous BCE loss and the current is less than eps
        or when you reach max_iter.
        Collect the BCE loss in each iteration in the loss variable.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Class labels.

        """
       
        # make sure to use 0/1 labels:
        self.class_names = np.unique(y)
        y_01 = np.where(y == self.class_names[0], 0, 1)

        # Initialize the random Weights 
        np.random.seed(self.random_state)
        self.w_ = 1e-6 * np.random.randn(X.shape[1])
        
        SCALING_FACTOR = self.learning_rate / X.shape[0]
        X_T = X.T

        
        prev_gradient_norm = float('inf')
        
        for iteration in range(self.max_iter):
            
            # Calculate predictions (needed for gradient)
            predictions = self._sigmoid(X @ self.w_)

            # Calculate gradient of the BCE loss (this is all we need!)
            gradient = X_T @ (predictions - y_01)

            # Update weights using gradient
            self.w_ -= SCALING_FACTOR * gradient

            # Convergence check based on gradient magnitude
            current_gradient_norm = np.linalg.norm(gradient) #"you don't have to calculate the loss in each , iteration. You just need to calculate the gradient of the loss." 
            
            if abs(current_gradient_norm - prev_gradient_norm) < self.eps:
                break
                
            prev_gradient_norm = current_gradient_norm


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        

def select_learning_rate(X_train, y_train, learning_rates, max_iter):
    """
    Select the learning rate attaining the minimal BCE after max_iter GD iterations

    Parameters
    ----------
    X_train : {array-like}, shape = [n_samples, n_features]
      Training vectors, where n_samples is the number of samples and
      n_features is the number of features.
    y_train : array-like, shape = [n_samples]
      Class labels.
    learning_rates : list
      The list of learning rates to test.
    max_iter : int
      The maximum number of iterations for the gradient descent.

    Returns
    -------
    selected_learning_rate : float
      The learning rate attaining the minimal BCE after max_iter GD iterations.
    """
    # Initialize variables to keep track of the minimum BCE and the corresponding learning rate
    min_bce = float('inf')
    selected_learning_rate = None
    
    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################
    
    model = LogisticRegressionGD(max_iter = max_iter)

    for eta in learning_rates:
        try:
        
          model.learning_rate = eta

          model.fit(X_train , y_train)

          if np.any(np.isnan(model.w_)) or np.any(np.isinf(model.w_)):
              # Skip this eta value
              continue
          
          bce_loss = model.BCE_loss(X_train , y_train)

          if np.isnan(bce_loss) or np.isinf(bce_loss):
              continue
          
          elif bce_loss <= min_bce:
              min_bce = bce_loss
              selected_learning_rate = eta
              

        except Exception as e:
          # If any errors occur, skip this eta value
          print(f"Error with eta={eta}: {e}")
          
          
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_learning_rate


def cv_accuracy_and_bce_error(X, y, n_folds):
    """
    Calculate the accuracy and BCE error of the model using cross-validation.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]
      Training samples, where n_samples is the number of samples and
      n_features is the number of features.
    y : array-like, shape = [n_samples]
      Target values.
    n_folds : int
      The number of folds for cross-validation.
    Returns 
    -------
    The function returns two lists: accuracies and BCE_losses.
    Each list contains the results for each of the n_folds of the cross-validation.
    """

    # Split the data into n_folds and initialize the lists for accuracies and BCE losses
    X_splits = np.array_split(X, n_folds)
    y_splits = np.array_split(y, n_folds)
    accuracies = []
    BCE_losses = []

    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################

    model = LogisticRegressionGD()

    for i in range(len(X_splits)):
        # Use fold i as test data
        X_test = X_splits[i]
        y_test = y_splits[i]
        
        # Combine all other folds as training data
        X_train_list = [X_splits[j] for j in range(len(X_splits)) if j != i]
        y_train_list = [y_splits[j] for j in range(len(y_splits)) if j != i]
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # Create and fit the model on training data
  
        model.fit(X_train, y_train)
        
        # Predict on test data (not training data!)
        y_pred = model.predict(X_test)
        
        # Calculate accuracy on test data
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        
        # Calculate BCE loss on test data
        bce_loss = model.BCE_loss(X_test, y_test)
        BCE_losses.append(bce_loss)
        

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return accuracies, BCE_losses


def calc_and_print_metrics(y_true, y_pred, positive_class):
    """
    Calculate and print the metrics for the LogisticRegression classifier.
    """
    # Calculate the metrics
    tp, fp, tn, fn = None, None, None, None
    tpr, fpr, tnr, fnr = None, None, None, None
    accuracy, precision, recall = None, None, None
    risk = None
    f1 = None

    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ##########################################################################
    # Convert to numpy arrays and boolean masks
    y_pred_bin = np.array(y_pred == positive_class)
    y_true_bin = np.array(y_true == positive_class)
    
    # Calculate basic counts
    tp = np.sum((y_pred_bin) & (y_true_bin))
    tn = np.sum((~y_pred_bin) & (~y_true_bin))
    fp = np.sum((y_pred_bin) & (~y_true_bin))
    fn = np.sum((~y_pred_bin) & (y_true_bin))

    # Calculate metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr
    risk = 1 - accuracy
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
      
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # Print the metrics    
    print(f"#TP: {tp}, #FP: {fp}, #TN: {tn}, #FN: {fn}")
    print(f"#TPR: {tpr}, #FPR: {fpr}, #TNR: {tnr}, #FNR: {fnr}")
    print(f"Accuracy: {accuracy}, Risk: {risk}, Precision: {precision}, Recall: {recall}")
    print(f"F1: {f1}")




def fpr_tpr_per_threshold(y_true, positive_class_probs, positive_class="9"):
    """
    Calculate FPR and TPR of a given classifier for different thresholds

    Parameters
    ----------
    y_true : array-like, shape = [n]
      True class labels for the n samples
    positive_class_probs : array-like, shape = [n]
      Predicted probabilities for the positive class for the n samples
    positive_class : str, optional
      The label of the class to be considered as the positive class
    """
    fpr , tpr = [] , []
    
    # consider thresholds from 0 to 1 with step 0.01
    prob_thresholds = np.arange(0, 1, 0.01)

    y_true_bin = (y_true == positive_class)
    total_pos = np.sum(y_true_bin)
    total_neg = len(y_true_bin) - total_pos
    # More efficient 
    # y_true_bin = np.where(y_true == positive_class, 1, 0)
    
    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################
    for threshold in prob_thresholds:
        y_pred_bin = positive_class_probs >= threshold
        
        tp = np.sum((y_pred_bin) & (y_true_bin))
        tn = np.sum((~y_pred_bin) & (~y_true_bin))
        fp = np.sum((y_pred_bin) & (~y_true_bin))
        fn = np.sum((~y_pred_bin) & (y_true_bin))

        # Calculate metrics
        cur_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  
        cur_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    
        fpr.append(cur_fpr)
        tpr.append(cur_tpr)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return fpr, tpr



