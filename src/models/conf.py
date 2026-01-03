import itertools

def gym(self, model='', max_combinations=10):
    if model == 'LR':
        gyms = {}
        gyms['penalty'] = ['l2', 'none']        # Types of penalty norms
        gyms['C'] = [0.1, 0.5, 1.0, 10.0] 
        gyms['class_weight'] = [None, 'balanced']   
        gyms['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] 
    
    elif model == 'SVM':
        gyms = {}  
        gyms['gamma'] = ["scale", "auto"]
        gyms['coef0'] = [0.0, 0.5, 1.0, 2.0]
        gyms['tol'] = [1e-4, 1e-3, 1e-1]
        gyms['C'] = [0.1, 0.5, 1.0, 10.0]
        gyms['kernel'] = ["linear", "poly", "rbf", "sigmoid"] 
    
    elif model == 'MLP':
        gyms = {}  
        gyms['act_fun'] = ['relu', 'tanh']
        gyms['batch_size'] = [64, 256]
        gyms['max_iter'] = [500, 1000, 2000]
        gyms['solver'] = ['lbfgs', 'sgd', 'adam']
        gyms['hidden_layer_sizes'] =[[20], [100, 20], [100, 50, 20]]  # Sizes of hidden layers
        gyms['alpha'] = [0.0001, 0.01, 0.1]
    
    elif model == 'RF':
        gyms = {}  
        gyms['n_estimators'] = [10, 100, 200]
        gyms['class_weight'] = [None, 'balanced'] 
        gyms['max_depth'] = [None, 10, 50]  
        gyms['min_samples_split'] = [2, 10]           # Minimum samples required to split a node
        gyms['min_samples_leaf'] = [1, 10]           # Minimum samples required at a leaf node
        gyms['max_features'] = ['auto', 'log2', None] # Number of features to consider for best split
        gyms['ccp_alpha'] = [0.0, 0.01, 0.1] 
    
    elif model == 'XGB':
        gyms = {} 
        gyms['learning_rate'] = [0.01, 0.1, 0.2]                  # Impact of each tree on the outcome
        gyms['max_depth'] = [3, 5, 7]                             # Maximum depth of each tree
        gyms['n_estimators'] = [100, 500, 1000] 
        gyms['reg_alpha'] = [0, 0.1, 0.5]                         # L1 regularization (sparsity control)
        gyms['reg_lambda'] = [0.1, 0.5, 1] 
        
    elif model == 'PCA':
        gyms = {}
        gyms['contamination'] = [0.1, 0.25, 0.5]
        gyms['svd_solver'] = ['auto', 'full', 'arpack', 'randomized']
        gyms['n_components'] =[5, 10, 15]
        
    elif model == 'IForest':
        gyms = {} 
        gyms['n_estimators'] = [10, 100, 200]
        gyms['max_samples'] = [50, 100]
        gyms['max_features'] = [0.5, 0.7, 1.0]
        gyms['contamination'] = [0.1, 0.25, 0.5]
    
    elif model == 'OCSVM':
        gyms = {}  
        gyms['nu'] = [0.1, 0.5, 0.9]
        gyms['degree'] = [2, 3, 5]
        gyms['gamma'] = ["scale", "auto"]
        gyms['coef0'] = [0.0, 0.5, 1.0, 2.0]
        gyms['tol'] = [1e-4, 1e-3, 1e-1]
        
    elif model == 'cuKNN':
        gyms = {}
        gyms['n_neighbors'] = [3, 5, 10] 
        gyms['algorithm'] = ['auto', 'brute']
        gyms['metric'] = ['euclidean', 'manhattan', 'minkowski', 'cosine']
        
    elif model == 'LOF':
       gyms = {}
       gyms['n_neighbors'] = [5, 10, 20] 
       gyms['algorithm'] = ['auto', 'ball_tree', 'kd_tree']
       gyms['contamination'] = [0.5]
    #    gyms['contamination'] = [0.1, 0.25, 0.5]
       
    elif model == 'cuLOF':
       gyms = {}
       gyms['k'] = [5, 10, 20] 
    #    gyms['algorithm'] = ['auto', 'ball_tree', 'kd_tree']
    #    gyms['contamination'] = [0.1, 0.25, 0.5]
        
        
    elif model == 'RF':
        gyms = {}
        gyms['class_weight'] = [None, 'balanced'] 
        gyms['max_depth'] = [None, 10, 50]                      # Maximum depth of the tree
        gyms['min_samples_split'] = [2, 10]           # Minimum samples required to split a node
        gyms['min_samples_leaf'] = [1, 10]           # Minimum samples required at a leaf node
        gyms['max_features'] = ['auto', 'log2', None] # Number of features to consider for best split
        gyms['ccp_alpha'] = [0.0, 0.01, 0.1]  
    
    elif model == 'AutoEncoder':
        gyms = {}
        gyms['batch_size'] = [32, 64, 128]
        gyms['hidden_neurons'] =[[20], [100, 20], [100, 50, 20]]
        gyms['weight_decay'] = [1e-5, 1e-4, 1e-3]
        gyms['hidden_activation'] = ['relu', 'tanh', 'leaky_relu']
    
    # elif mode == 'LR':
    
    # elif mode == 'XGB':
    
    elif model == 'MLP':
        gyms = {}
        gyms['act_fun'] = ['relu', 'tanh']
        gyms['batch_size'] = [64, 256]
        gyms['max_iter'] = [500, 1000, 2000]
        gyms['solver'] = ['lbfgs', 'sgd', 'adam']  
        gyms['hidden_layer_sizes'] =[[20], [100, 20], [100, 50, 20]]  # Sizes of hidden layers
         
    else:
        raise NotImplementedError

    # Generate limited combinations using itertools.product
    param_keys = list(gyms.keys())
    param_values = list(gyms.values())
    all_combinations = list(itertools.product(*param_values))

    if max_combinations and len(all_combinations) > max_combinations:
        all_combinations = all_combinations[:max_combinations]

    limited_combinations = [dict(zip(param_keys, combination)) for combination in all_combinations]

    return limited_combinations
