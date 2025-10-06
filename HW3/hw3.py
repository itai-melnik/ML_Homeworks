import numpy as np
from typing import Union
import math

def poisson_log_pmf(k: Union[int, np.ndarray], rate: float) -> Union[float, np.ndarray]:
    """
    k: A discrete instance or an array of discrete instances
    rate: poisson rate parameter (lambda)

    return the log pmf value for instances k given the rate
    """

    log_p = None
    ###########################################################################
    #                                          #
    ###########################################################################
    results = []
    if type(k) != int:
        for x in k:
            log_p =  np.log(((np.e)**(-rate))*((rate**x)/math.factorial(x)))  #np.log(x) = ln(x)
            results.append(log_p)
            
        return results
    
    elif type(k) == int:
        log_p =  np.log(((np.e)**(-rate))*((rate**k)/math.factorial(k)))  #np.log(x) = ln(x)
        return log_p
            
        
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    #                                         #
    ###########################################################################
    
    mean = np.mean(samples) #the mean is a MLE 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def possion_confidence_interval(lambda_mle, n, alpha=0.05):
    """
    lambda_mle: an MLE for the rate parameter (lambda) in a Poisson distribution
    n: the number of samples used to estimate lambda_mle
    alpha: the significance level for the confidence interval (typically small value like 0.05)
 
    return: a tuple (lower_bound, upper_bound) representing the confidence interval
    """
    # Use norm.ppf to compute the inverse of the normal CDF
    from scipy.stats import norm
    lower_bound = None
    upper_bound = None
    ###########################################################################
    #                                           #
    ###########################################################################
    lower_bound = lambda_mle - norm.ppf(1-alpha/2)*np.sqrt(lambda_mle/n)
    upper_bound = lambda_mle + norm.ppf(1-alpha/2)*np.sqrt(lambda_mle/n)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return lower_bound, upper_bound

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = []
    ###########################################################################
    #                                         #
    ###########################################################################
    
    for r in rates:
        likelihoods.append((np.sum(poisson_log_pmf(samples, r))))
         
    likelihoods = np.array(likelihoods)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.15,
            (1, 1): 0.55
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0,
            (0, 1): 0.3,
            (1, 0): 0.5,
            (1, 1): 0.2
        }  # P(X=x, C=c)

        self.Y_C = {
            (0, 0): 0.05,
            (0, 1): 0.25,
            (1, 0): 0.45,
            (1, 1): 0.25
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0,
            (0, 0, 1): 0.15,
            (0, 1, 0): 0,
            (0, 1, 1): 0.15,
            (1, 0, 0): 0.05,
            (1, 0, 1): 0.1,
            (1, 1, 0): 0.45,
            (1, 1, 1): 0.1,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for keys, value in X_Y.items():
            x, y = keys
            
            bool = (X[x]*Y[y] == value)
            
            if(not bool):
                return True
            
        return False
            
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        
        for keys, value in X_Y_C.items():
            x, y, c = keys
            
            bool = ((X_C[(x, c)]/C[c])*(Y_C[(y, c)]/C[c]) == value/C[c])
            
            if not bool:
                return False
            
        return True
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    #                                       #
    ###########################################################################
    
    #p = (1/np.sqrt(2*np.pi*(std**2)))*(np.e**(-((x-mean)**2)/2*(std**2))) 
    diff = x - mean
    p = (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * (diff / std) ** 2)
   
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates information on the feature-specific
        class conditional distributions for a given class label.
        Each of these distributions is a univariate normal distribution with
        separate parameters (mean and std).
        These distributions are fit to specified training data.
        
        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class label to calculate the class conditionals for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.dataset = dataset
        self.dataset_class = [dataset[i] for i in range(len(dataset) - 1) if dataset[i][-1] == class_value] #dataset with the class label of class_value
        self.class_value = class_value
        self.mean = np.array([np.mean(np.transpose(self.dataset_class)[i]) for i in range(len(np.transpose(self.dataset_class)) - 1)])
        self.std = np.array([np.std(np.transpose(self.dataset_class)[i]) for i in range(len(np.transpose(self.dataset_class)) - 1)])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

                
        prior = len(self.dataset_class) / len(self.dataset)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the feature-specific classc conditionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        #get pdf for each feature given class label = class_value #it is vectorized
        likelihood = normal_pdf(x,self.mean,self.std)
        
            
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        joint_prob = self.get_prior()*(np.prod(self.get_instance_likelihood(x)))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class holds a ClassDistribution object (either NaiveNormal or MultiNormal)
        for each of the two class labels (0 and 1). 
        Using these objects it predicts class labels for input instances using the MAP rule.
    
        Input
            - ccd0 : A ClassDistribution object for class label 0.
            - ccd1 : A ClassDistribution object for class label 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        p0 = self.ccd0.get_instance_joint_prob(x)
        p1 = self.ccd1.get_instance_joint_prob(x)
  
        # Compare posterior probabilities
        pred = 0 if p0 > p1 else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multivariate normal desnity function under specified mean vector
    and covariance matrix for a given x.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    d = len(mean)
    
    pdf = (1/((2*np.pi) ** (d/2)) * (np.linalg.det(cov) ** 0.5))*(np.exp(-0.5*((x-mean) @ np.linalg.inv(cov) @ (x-mean))))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the multivariate normal distribution
        representing the class conditional distribution for a given class label.
        The mean and cov matrix should be computed from a given training data set
        (You can use the numpy function np.cov to compute the sample covarianve matrix).
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        self.dataset = dataset
        
        #dataset with the class label of class_value and class value dropped (only features)
        self.dataset_class = np.array([dataset[i][:-1] 
            for i in range(len(dataset)) if dataset[i][-1] == class_value])
        
        
        self.class_value = class_value
        self.mean = np.array([np.mean(np.transpose(self.dataset_class)[i]) for i in range(len(np.transpose(self.dataset_class)))])
        
        n = self.dataset_class.shape[0]
        self.cov = ((self.dataset_class - self.mean).T @ (self.dataset_class - self.mean))/n #cov matrix estimator 
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = len(self.dataset_class) / len(self.dataset)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the multivariate classc conditionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = multi_normal_pdf(x=x, mean=self.mean, cov=self.cov)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        joint_prob = self.get_prior()*(np.prod(self.get_instance_likelihood(x)))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob



def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given MAP classifier on a given test set.
    
    Input
        - test_set: The test data (Numpy array) on which to compute the accuracy. The class label is the last column
        - map_classifier : A MAPClassifier object that predicits the class label from a feature vector.
        
    Ouput
        - Accuracy = #Correctly Classified / number of test samples
    """
    acc = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    num_correctly_classified = 0
    for i in range(test_set.shape[0]):
        num_correctly_classified += (test_set[i][-1] == map_classifier.predict(test_set[i, :-1]))
    
    acc = num_correctly_classified / test_set.shape[0]

   
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the probabilites for a discrete naive bayes
        class conditional distribution for a given class label.
        The probabilites of each feature-specific class conditional
        are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the probabilities for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.dataset = np.array(dataset)
        self.dataset_class = np.array([dataset[i] for i in range(len(dataset)) if dataset[i][-1] == class_value]) #dataset with the class label of class_value
        self.class_value = class_value
        self.mean = np.array([np.mean(np.transpose(self.dataset_class)[i]) for i in range(len(np.transpose(self.dataset_class)) - 1)])
        self.std = np.array([np.std(np.transpose(self.dataset_class)[i]) for i in range(len(np.transpose(self.dataset_class)) - 1)])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.dataset_class.shape[0]/ self.dataset.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the product of feature-specific discrete class conidtionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
 
        
        
        X_train = self.dataset_class[:,:-1]
        n_c, d = X_train.shape                            # n_c = #samples of this class
        x = np.atleast_2d(x).astype(X_train.dtype)        # ensure shape (m, d)
        m = x.shape[0]

        # vocabulary size per feature (|{v : ∃ sample with feature_j = v}|)  – needed for Laplace
        V = np.apply_along_axis(lambda col: np.unique(col).size, axis=0, arr=X_train)  # shape (d,)

        # Summing across training samples 
        counts = np.zeros((m, d), dtype=int)
        for j in range(d):
            counts[:, j] = (x[:, [j]] == X_train[:, j]).sum(axis=1)
            
        #laplace smoothing
        probs = (counts + 1) / (n_c + V)                 
        #naive bayes
        
        likelihood = probs.prod(axis=1)          
            
            
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        joint_prob = self.get_prior()*(self.get_instance_likelihood(x))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob
