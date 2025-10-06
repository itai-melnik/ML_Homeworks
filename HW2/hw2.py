import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    class_column = data[:, -1] #just get the class column
    edible_frequency = np.mean(class_column == 'e') # frequency of edible labels #edible/(#edible + #poisonous)
    
    p = np.array((edible_frequency, 1-edible_frequency))
    
    gini = 1 - (p.T @ p)
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    class_column = data[:, -1] #just get the class column
    edible_frequency = np.mean(class_column == 'e') # frequency of edible labels #edible/(#edible + #poisonous)
    
    p = np.array((edible_frequency, 1-edible_frequency))
    
   
    # if p = 0 then log(p) is not defined so we ignore it. 
    non_zero_p = p[p > 0.0]
    entropy = -np.sum(non_zero_p * np.log2(non_zero_p))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the data instances associated with the node
        self.terminal = False # True iff node is a leaf
        self.feature = feature # column index of feature/attribute used for splitting the node
        self.pred = self.calc_node_pred() # the class prediction associated with the node
        self.depth = depth # the depth of the node
        self.children = [] # the children of the node (array of DecisionNode objects)
        self.children_values = [] # the value associated with each child for the feature used for splitting the node
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to use for measuring goodness of a split
        self.gain_ratio = gain_ratio # True iff GainRatio is used to score features
        self.feature_importance = 0
        
    
  
              
            
    def calc_node_pred(self):
        """
        Calculate the node's prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        edible_frequency = np.mean(self.data[:, -1] == 'e')
        pred = 'e' if edible_frequency >= 0.5 else 'p'
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
    
        
        feature_values = np.unique(self.data[:, feature])
        
        #dictionary of values of features as keys and data_subsets where those values appear e.g {'sunny': all rows which have sunny}
        groups = {feature_value: self.data[self.data[:, feature] == feature_value] for feature_value in feature_values } 
        
        
        if(self.gain_ratio): #gainratio
            IG = (self.impurity_func(self.data) - 
                  np.sum([ ((len(data_subset)/len(self.data))*self.impurity_func(data_subset)) for data_subset in groups.values()]))
            SplitInfo = - np.sum( [(len(data_subset)/len(self.data))*np.log2((len(data_subset)/len(self.data))) for data_subset in groups.values()])
            
            goodness = IG / SplitInfo if SplitInfo != 0 else 0
        else: #impurity reduction
            goodness = (self.impurity_func(self.data) - 
                        np.sum( [ ((len(data_subset)/len(self.data))*self.impurity_func(data_subset)) for data_subset in groups.values()] )) ##### 
            
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        gos, _ = self.goodness_of_split(self.feature)
        
        self.feature_importance = (len(self.data)/n_total_sample)*gos # num of samples in node / total samples in training set
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        #if node is pure
        if np.all(self.data[:, -1] == self.data[0, -1]):
            self.terminal = True
            return
       
        
        best_gos = 0
        best_i = 0
        best_groups = {}
        
        for i in range(len(self.data.T) - 1):
            curr_gos, groups = self.goodness_of_split(i)
            if curr_gos > best_gos:
                best_gos, best_groups, best_i = curr_gos, groups, i
                
        
        self.feature = best_i #make node associated with the best feature to split by
        
        #depth pruning
        if self.depth >= self.max_depth:
            self.terminal = True
            return
        
        #for chi pruning
        #if prob is above thershold make this node a leaf
        X_2 = self.chi_squared(best_groups)
        deg_freedom = len(np.unique(self.data[:, self.feature]))- 1 
        if (deg_freedom > 0 and self.chi != 1 and X_2 < chi_table[deg_freedom][self.chi]): 
            self.terminal = True
            return
        
        #if goodness of split is 0
        if best_gos == 0:
            self.terminal = True
            return         
            
        for val, data_subset in best_groups.items():
            child = DecisionNode(data_subset, self.impurity_func, 
                    depth=(self.depth + 1), chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            
            child.pred = child.calc_node_pred()
            
            self.add_child(child, val)
        


    def chi_squared(self, groups) -> int:
      
        total= len(self.data)
        total_e = np.sum(self.data[:, -1] == 'e')
        total_p = total - total_e

        chi2 = 0.0
        for data_subset in groups.values():
            col_total = len(data_subset)
            if col_total == 0:
                continue 

            expected_e = total_e * col_total / total
            expected_p = total_p * col_total / total

            observed_e = np.sum(data_subset[:, -1] == 'e')
            observed_p = col_total - observed_e

            chi2 += (observed_e - expected_e) ** 2 / expected_e
            chi2 += (observed_p - expected_p) ** 2 / expected_p

        return chi2
        
            
    
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the training data used to construct the tree
        self.root = None # the root node of the tree
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.gain_ratio = gain_ratio #
        
    def depth(self):
        return self.root.depth
    
    #fixed the given depth method but did not modify the original in case of automatic tests
    def depth_fixed(self):
        def _max_depth(node):
                    # base
                    if node is None:
                        return -1       

                    if node.terminal or not node.children:
                        return node.depth

                    return max(_max_depth(child) for child in node.children)

        return _max_depth(self.root)
            

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        n_total_sample = len(self.data)
        
        self.root = DecisionNode(self.data, self.impurity_func, gain_ratio=self.gain_ratio, max_depth=self.max_depth, chi=self.chi)
            
        def recursive_builder(node: DecisionNode) -> DecisionNode:
            
           
            node.split() #splits according to best feature
            node.calc_feature_importance(n_total_sample) #calculate the feature importance 
            
            #base if all samples the same labels as leaf   
            if node.terminal: 
                return node
            

            #recursively do the same on all children of the node
            for child in node.children:
                recursive_builder(child)
                
         
        recursive_builder(node=self.root)
        
            
        
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        node = self.root
        
        #while node is not a leaf
        while(not node.terminal): # node.feature = -1 if node is not associated with a feature
            att_val = instance[node.feature] #value of attribute e.g sunny
            
            #we might have not encountered something similar in training so could lead to error but 
            #could be a problem check split() later 
            try: 
                index = node.children_values.index(att_val) 
            except:
                break
                
            node = node.children[index] #continue to the next node according to the value 
            
            
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        total_correct_preds = 0
        
        for instance in dataset:
            pred = self.predict(instance)
            actual = instance[-1]
            
            total_correct_preds += (pred==actual)
            
        accuracy = total_correct_preds/len(dataset)
               
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
    
        
 
        

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
   
        tree_train = DecisionTree(data=X_train, impurity_func=calc_entropy, gain_ratio=True, max_depth=max_depth)
        tree_train.build_tree()
        
        train_accuracy = tree_train.calc_accuracy(X_train)
        train_validation = tree_train.calc_accuracy(X_validation)
        training.append(train_accuracy)
        validation.append(train_validation)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for p_cutoff in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        
        tree_train = DecisionTree(data=X_train, impurity_func=calc_entropy, gain_ratio=True,chi=p_cutoff)
        tree_train.build_tree()
        
        train_accuracy = tree_train.calc_accuracy(X_train)
        train_validation = tree_train.calc_accuracy(X_test)
        chi_training_acc.append(train_accuracy)
        chi_validation_acc.append(train_validation)
        depth.append(tree_train.depth_fixed())
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_validation_acc, depth





def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    n_nodes = 1
    
    if node.terminal:
        return 1
    
    for child in node.children:
        n_nodes += count_nodes(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






