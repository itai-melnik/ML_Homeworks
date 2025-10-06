import numpy as np
import pandas as pd
import copy

class LinearSVM(object):
    """
    Linear SVM Classifier. Use hinge loss minimization with gradient descent.

    Parameters
    ----------
    C : float, optional (default=1.0)
        The slackness parameter.
    learning_rate : float, optional (default=0.001)
        The learning rate for the gradient descent.
    max_iter : int, optional (default=10000)
        The maximum number of iterations for the gradient descent.
    random_state : int, optional (default=1)
        Random number generator seed for random weight
        initialization.
    eps : float, optional (default=0.000001)
        The minimum change in the loss to declare convergence.
    """


    def __init__(self, learning_rate=0.005, C=1.0, max_iter=100000, random_state=42, eps=0.000001):
        self.learning_rate = learning_rate
        self.C = C  # regularization parameter
        self.max_iter = max_iter
        self.random_state = random_state
        self.loss_history = [np.inf]
        self.eps = eps

        self.w = None
        self.w0 = None

    def predict_raw(self, X):
        """
        Compute raw predictions (before thresholding): <w,x> + w0
        """
        return X @ self.w + self.w0

    def predict(self, X):
        """
        Return the predicted class labels (0 or 1)
        """
        raw_predictions = self.predict_raw(X)
        return np.where(raw_predictions >= 0, 1, 0)

    def fit(self, X, y, verbose=False):
        """
        Fit training data (the learning phase).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples, where n_examples is the number of examples and
            n_features is the number of features.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Class labels
        verbose : bool, optional (default=False)
            If True, print the loss every 100 iterations.
        """
        # set random seed
        np.random.seed(self.random_state)

        # get shape data
        n_examples = X.shape[0]
        n_features = X.shape[1]

        # guess random w and w0
        self.w = np.random.random((n_features, 1))
        self.w0 = np.random.random()

        # ensure y is in {-1, 1} format
        y = np.array(y).reshape(-1, 1)
        y = np.where(y == 0, -1, y)

        for iteration in range(self.max_iter):
            # compute raw predictions (<w,x> + w0)
            raw_predictions = self.predict_raw(X)
            
            # compute hinge loss and gradients
            loss, dw, dw0 = self.compute_hinge_loss_gradients(X, y, raw_predictions)
            
            # print loss every 100 iterations
            if (iteration + 1) % 100 == 0 and verbose:
                print(f"Iteration {iteration + 1}: Loss = {loss:.6f}")
            
            # update parameters using gradient descent
            self.w  -= self.learning_rate * dw
            self.w0 -= self.learning_rate * dw0

            # check if loss is has converged:
            if self.loss_history[-1] - loss < self.eps:
                break
            
            self.loss_history.append(loss)
            

    def compute_hinge_loss_gradients(self, X, y, raw_predictions):
        """
        Compute hinge loss and its gradients
        Hinge loss:  0.5*||w||^2  + C/n * max(0, 1 - y * (<w,x> + w0))
        Inputs:
        - X: training data (n_examples, n_features)
        - y: class labels (n_examples, 1) in {-1, 1} format
        - raw_predictions: raw predictions (<w,x> + w0) (n_examples, 1)
        Outputs:
        - loss: scalar, the hinge loss
        - dw: gradient of the loss with respect to w (n_features, 1)
        - dw0: gradient of the loss with respect to w0 (scalar)
        """
        loss = 0
        dw = np.zeros_like(self.w)
        dw0 = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        
        n = X.shape[0]

        #compute margins
        margins = y * raw_predictions                       

        #hinge loss values for each sample
        hinge = np.maximum(0, 1 - margins)                  

        #computes the loss
        loss = 0.5 * np.sum(self.w ** 2)                   
        loss += (self.C / n) * np.sum(hinge)     

        #computes the gradient
        #get indixes of samples that violate the margin. when hinge > 0)
        active = (hinge > 0).flatten()                     

        if np.any(active):
            X_active = X[active]                           
            y_active = y[active]                            

            #the gradient for w_i
            dw = self.w - (self.C / n) * (X_active.T @ y_active)

            #the gradient for w_0 
            dw0 = -(self.C / n) * np.sum(y_active)
        else:
            #the gradient for regularisation term (C)
            dw = self.w.copy()
            dw0 = 0.0
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################    
        return loss, dw, dw0


def cross_validation(X, y, n_folds, classifier, random_state=42):
    """
    n-fold cross validation. Split the data randomly to n_folds roughly equal subset. 
    Repeat for i=1,...,n_folds iterations: set aside subset i, train a classifier
    using the remaining n_folds-1 subsets, and evaluate it on subset i.
    Return the average accuracy across all n_folds iterations.

    Parameters:
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,) or (n_samples, 1)
        Target values.
    n_folds : int
        Number of folds for cross-validation.
    classifier : object
        Classifier object that implements fit() and predict() methods.
    random_state : int
        Random seed for reproducibility.

    """

    # set random seed
    np.random.seed(random_state)

    # copy \ reshape data so we will not shuffle the original data
    X = X.copy()
    y = np.reshape(y.copy(), (-1, 1))

    # shuffle data
    data_Xy = np.hstack((X, y.reshape(-1, 1)))
    np.random.shuffle(data_Xy)
    X = data_Xy[:, :-1]
    y = data_Xy[:, -1].reshape(-1,1)

    # build folds
    X_folds = np.array_split(X, n_folds)
    y_folds = np.array_split(y, n_folds)

    accuracy = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    #accuracies across the different folds
    accuracies = []

    for i in range(n_folds):
        #validation set
        X_val = X_folds[i]
        y_val = y_folds[i]

        #training set
        X_train = np.vstack([X_folds[j] for j in range(n_folds) if j != i])
        y_train = np.vstack([y_folds[j] for j in range(n_folds) if j != i])

        #deep copy of classifier
        classifier_copy = copy.deepcopy(classifier)

        #train
        classifier_copy.fit(X_train, y_train)
        #prediction of validation data
        y_pred = classifier_copy.predict(X_val).reshape(-1, 1)

        # compute accuracy for the current fold
        fold_acc = np.mean(y_pred == y_val)
        accuracies.append(fold_acc)

    # average accuracy across folds
    accuracy = float(np.mean(accuracies))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################    
    return accuracy

def norm_pdf(x, mu, sigma):
    """
    Normal desnity function.
    Inputs:
    - x: a real value or a vector of real values
    - mu: mean of the normal distribution
    - sigma: standard deviation of the normal distribution
    Outputs:
    - prob: the probability densities of the normal distribution at x
    """
    x = np.reshape(x, (-1, 1))  # ensure x is a column vector
    prob = np.zeros_like(x)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
 
    prob = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) ** 2) / (sigma ** 2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################    
    return prob

def gmm_pdf(x, weights, mus, sigmas):
    """
    Probability density function of a univariate Gaussian mixture model.
    Inputs:
    - x: a real value or a vector of real values
    - weights: a vector of weights for each Gaussian component
    - mus: a vector of means for each Gaussian component
    - sigmas: a vector of standard deviations for each Gaussian component
    Outputs:
    - prob: the probability densities of the GMM at x
    """
    x = np.reshape(x, (-1, 1))
    prob = np.zeros_like(x)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    #weighted sum of Gaussian PDFs
    prob = np.zeros_like(x)
    for w, mu, sigma in zip(weights, mus, sigmas):
        prob += w * norm_pdf(x, mu, sigma)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################    
    return prob

class GMM(object):
    """
    Fit a Gaussian Mixture Model (EM) to the data.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, max_iter=1000, eps=0.000001, random_state=42):
        # parameters defining the GMM
        self.k = k
        self.weights = None
        self.mus = None
        self.sigmas = None

        # parameters for the EM algorithm
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        np.random.seed(self.random_state)

        # attributes for the EM algorithm
        # these will be updated during the EM process
        self.responsibilities = None
        self.losses = None

    def get_dist_params(self):
        """
        Return the distribution parameters of the GMM.
        Outputs:
        - params: a dictionary with keys 'weights', 'mus', 'sigmas'
          containing the GMM parameters.
          ALREADY IMPLEMENTED. DO NOT MODIFY IT.
        """
        return {'weights': self.weights, 'mus': self.mus, 'sigmas': self.sigmas}
    
    def init_params(self, X):
        """
        Initialize GMM parameters (weights, mus, sigmas).
        Used in the beginning of the EM algorithm.
        Inputs:
        - X: training data (n_examples, n_features)
        THIS FUNCTION IS ALREADY IMPLEMENTED. DO NOT MODIFY IT.
        """
        self.losses = []

        self.weights = np.array( [1 / self.k] * self.k ) # unitform 

        if self.k == 1:
            # if k is 1, single gaussian - a good (best) guess will be the empirical mean and std
            self.mus = np.mean(X)
            self.sigmas = np.std(X)
        else:
            self.mus = np.random.random(self.k)
            self.sigmas = np.random.random(self.k)

            # so we will not start with sigmas that are too small
            self.sigmas[self.sigmas < 0.25] += 0.25

    def fit(self, X, verbose=False):
        """
        Fit GMM to data using the EM algorithm.        
        Use init_params to initialize all model parameters
        and then apply the EM algorithm (by invoking the expectation and maximization function).
        Store the params in attributes of the EM object and the losses in self.losses.
        Function halts when the difference between current and previous loss is less than eps
        or when you reach max_iter.
        Inputs:
        - X: training data (n_examples, n_features=1)
        - verbose: if True, print initial parameters in the begninning and the loss every 5 iterations
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        # initialize parameters
        self.init_params(X)
        if verbose:
            print(f"Initial params -> weights: {self.weights}, mus: {self.mus}, sigmas: {self.sigmas}")

        prev_loss = np.inf
        for iteration in range(self.max_iter):
            
            #E step
            self.expectation(X)
            
            #M step
            self.maximization(X)
            
            #computing loss
            cur_loss = self.loss(X)
            self.losses.append(cur_loss)

            if verbose and (iteration % 5 == 0):
                print(f"Iter {iteration}: loss = {cur_loss:.6f}")

            if prev_loss - cur_loss < self.eps:
                break
            
            prev_loss = cur_loss
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


    def expectation(self, X):
        """
        Implements the E step of the EM algorithm.
        Calculate the responsibilities (posterior probabilities) of each Gaussian component for each data point.
        Update the self.responsibilities attribute.
        Inputs:
        - X: training data (n_examples, n_features=1)
        """
        
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
       

        #the responsibilities
        respons = self.weights * norm_pdf(X, self.mus, self.sigmas) 
        
        #normalize
        respons /= respons.sum(axis=1, keepdims=True) + 1e-300

        self.responsibilities = respons
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, X):
        """
        Implements the M step of the EM algorithm.
        Update the GMM parameters (weights, mus, sigmas) based on the current responsibilities.
        Inputs:
        - X: training data (n_examples, n_features=1)
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        X = np.asarray(X).reshape(-1, 1)
        n = X.shape[0]
        r = self.responsibilities 

        Nk = r.sum(axis=0)
        
        #avoid divisions by zero
        Nk_safe = Nk + 1e-300

        #update the weights
        self.weights = Nk / n

        #update the mus
        self.mus = (r.T @ X).flatten() / Nk_safe

        #updating the sigmas
        diff = X - self.mus                      
        sigmas_sq = (r * diff ** 2).sum(axis=0) / Nk_safe
        
        
        #using sqrt(bigSigma r *(x-mu)**2) rather than biased sigma_hat to avoid numerical issues
        sigma_eps = 1e-12
        self.sigmas = np.sqrt(np.maximum(sigmas_sq, sigma_eps))
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def loss(self, X):
        """
        Calculate the loss function for the GMM.
        The loss is the negative log likelihood of the data given the GMM parameters.
        Inputs:
        - X: training data (n_examples, n_features=1)
        Outputs:
        - c: the loss value (scalar)
        """
        #very small sigmas to avoid numerical issues
        sigma_eps = 1e-12
        self.sigmas[self.sigmas < sigma_eps] = sigma_eps
        c = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        #compute mixture pdf
        pdf_components = self.weights * norm_pdf(X, self.mus, self.sigmas) 
        mix_pdf = pdf_components.sum(axis=1).flatten()

        #the negative log likelihood
        c = -np.sum(np.log(mix_pdf + 1e-300))
        
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################    

        return c

    def pdf(self, x):
        """
        Return the probability density function of the GMM at point x.
        Inputs:
        - x: a real value or a vector of real values
        Outputs:
        - prob: the probability densities of the GMM at x
        """
        prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        prob = gmm_pdf(x, self.weights, self.mus, self.sigmas)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prob
        


class NaiveBayesGMM(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=42):
        self.k = k
        self.random_state = random_state

        self.prior = None
        self.gmm_dict = None

    def fit(self, X, y, verbose=False):
        """
        Fit class conditional distributions and prior probabilities.
        A GMM is fitted for each feature of each class using the EM algorithm.
        The fitted GMM objects are stored in a dictionary, where the keys are class labels and feature indices
        The prior probabilities are stored in the self.prior dictionary
        Inputs:
        - X: training data (n_examples, n_features)
        - y: class labels (n_examples, 1)
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        


        n_samples, n_features = X.shape

        #Initialise containers
        self.prior = {}
        self.gmm_dict = {}

        classes = np.unique(y)

        for cls in classes:
            #Indices of samples belonging to this class
            idx = np.where(y == cls)[0]

            #Prior probability P(y = cls)
            self.prior[cls] = len(idx) / n_samples

            #Fit a univariate GMM to each feature for this class
            self.gmm_dict[cls] = {}
            for feat in range(n_features):
                gmm = GMM(k=self.k, max_iter=1000, eps=1e-6, random_state=self.random_state)
                gmm.fit(X[idx, feat].reshape(-1, 1), verbose=False)
                self.gmm_dict[cls][feat] = gmm

                if verbose:
                    print(f"Fitted GMM for class {cls}, feature {feat} "
                          f"(weights={gmm.weights}, mus={gmm.mus}, sigmas={gmm.sigmas})")

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels
        Inputs:
        - X: test data (n_examples, n_features)
        Outputs:
        - class_predictions: predicted class labels (n_examples, 1)
        """
        class_predictions = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
    
        n_samples, n_features = X.shape

        classes = list(self.prior.keys())
        n_classes = len(classes)

        #the matrix of log-posterior
        log_scores = np.zeros((n_samples, n_classes), dtype=float)

        #because of log(0)
        eps = 1e-300  

        for j, cls in enumerate(classes):
            log_prob = np.log(self.prior[cls] + eps)

            #add log likelihoods from each feature GMM
            for feat in range(n_features):
                gmm = self.gmm_dict[cls][feat]
                pdf_vals = gmm.pdf(X[:, feat].reshape(-1, 1)).flatten()
                log_prob += np.log(pdf_vals + eps)

            log_scores[:, j] = log_prob

        #Pick the class with max posterior
        best_idx = np.argmax(log_scores, axis=1)
        class_predictions = np.array([classes[i] for i in best_idx]).reshape(-1, 1)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return class_predictions
