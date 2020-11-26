import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class BinaryClassifier:

    def __init__(self, X: np.array, y: np.array, strategy: str, groups: np.array = None,
                    random_state: int = 0, cross_validation: bool = False, logo_validation: bool = False, scoring = 'accuracy'):
        self.X = X
        self.y = y
        self.strategy = strategy # Stress detection strategy - possible options: mlp, knn, svm, logistic_regression, random_forest
        self.groups = groups
        self.random_state = random_state
        self.cross_validation = cross_validation # Cross-validation approach
        self.logo_validation = logo_validation # Leave one group out approach
        self.scoring = scoring
    

    def __get_hyper_parameters(self, method):
        params = dict()
        if method == 'random_forest':
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 500, stop = 1000, num = 2)]
            # Number of features to consider at every split
            # Minimum number of samples required to split a node
            min_samples_split = [2, 4]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 4]
            # Class Weights
            class_weight = [None, 'balanced']
            # Method of selecting samples for training each tree
            # Create the random grid
            params = {'n_estimators': n_estimators,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'class_weight': class_weight,
            }
        elif method == 'logistic_regression':
            params = {'C': [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 10], 'class_weight': ['balanced', None] }
        elif method == 'svm':
            params = {'C': [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 10], 'class_weight': ['balanced', None],
                'kernel': ['rbf'] }
        elif method == 'mlp':
            params = { 'hidden_layer_sizes': [(64,), (128,), (256,), (512,)] } 
        elif method == 'knn':
            params = { 'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'] }
        return params
    

    def __get_classifier(self, method):
        clf = None 
        if method == 'random_forest':
            clf = RandomForestClassifier()
        elif method == 'logistic_regression':
            clf = LogisticRegression()
        elif method == 'svm':
            clf = SVC()
        elif method == 'mlp':
            clf = MLPClassifier()
        elif method == 'knn':
            clf = KNeighborsClassifier()
        return clf
    

    def __transform_data(self, method, X_train, X_test): # Transform the data using Standard Scaler
        scaled_X_train = X_train
        scaled_X_test = X_test
        if method in ['mlp', 'svm', 'knn']: # Only use for MLP, SVM, and KNN as these methods are sensitive to feature scaling
            std_scaler = StandardScaler()
            std_scaler.fit(X_train)
            scaled_X_train = std_scaler.transform(X_train)
            scaled_X_test = std_scaler.transform(X_test)
        return scaled_X_train, scaled_X_test
    

    def cross_validator(self, method: str):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.286, stratify = self.y, random_state = self.random_state) # Split train test data

        # Validate if the test set and train set have two classes
        num_classes_test = len(np.unique(y_test))
        num_classes_train = len(np.unique(y_train))
        if num_classes_test < 2 or num_classes_train < 2: # If one of them does not have enough classes, then ignore it
            balanced_accuracy = -1
            cv_balanced_accuracy = -1
            return cv_balanced_accuracy, balanced_accuracy

        X_train, X_test = self.__transform_data(method, X_train, X_test) # Feature scaling if possible

        # Get hyperparamters for grid-search and classifier of the corresponding method
        hyper_params = self.__get_hyper_parameters(method)
        clf = self.__get_classifier(method)

        # Define some constant parameters of the grid-search algorithm
        CV_NUM_SPLITS = 3
        VERBOSE = False
        N_JOBS = -1 # Use all CPUs

        # Start training
        self.grid_search_cv = GridSearchCV(estimator = clf, scoring = self.scoring, 
            cv = StratifiedKFold(n_splits = CV_NUM_SPLITS), param_grid = hyper_params, verbose = VERBOSE, n_jobs = N_JOBS).fit(X_train, y_train)
        cv_balanced_accuracy = self.grid_search_cv.best_score_
        
        # Fit the classifier into test set
        y_preds = self.grid_search_cv.predict(X_test)
        # Evaluate the results based on balanced_accuracy_score
        balanced_accuracy = self.evaluate(y_test, y_preds)
        return cv_balanced_accuracy, balanced_accuracy


    def leave_one_group_out_validator(self, method: str) -> Dict[str, list]:
        logo = LeaveOneGroupOut()
        test_groups = []
        balanced_accs = []
        cv_balanced_acc_scores = []

        # Get hyperparameters for grid-search and classifier of the corresponding method
        hyper_params = self.__get_hyper_parameters(method)

        # Define some constant parameters of the grid-search algorithm
        CV_NUM_SPLITS = 3
        VERBOSE = False
        N_JOBS = -1 # Use all CPUs

        for train_index, test_index in tqdm(logo.split(self.X, self.y, self.groups)):
            X_train, y_train, X_test, y_test = self.X[train_index], self.y[train_index], self.X[test_index], self.y[test_index] # Get train and test data
            # Validate if the test set and train set have two classes
            num_classes_test = len(np.unique(y_test))
            num_classes_train = len(np.unique(y_train))
            if num_classes_test < 2 or num_classes_train < 2: # If one of them does not have enough classes, then ignore it
                continue

            X_train, X_test = self.__transform_data(method, X_train, X_test) # Feature scaling if possible

            clf = self.__get_classifier(method) # Re-initialize classifier when iterating a new group

            # Run grid-search cross-validation
            self.grid_search_cv = GridSearchCV(estimator = clf, scoring = self.scoring, cv = StratifiedKFold(n_splits = CV_NUM_SPLITS), 
                        param_grid = hyper_params, verbose = VERBOSE, n_jobs = N_JOBS).fit(X_train, y_train)
            cv_balanced_acc_scores.append(self.grid_search_cv.best_score_) # Save Grid-Search CV best score
            
            # Run prediction on test set
            y_preds = self.grid_search_cv.predict(X_test)

            # Evaluate balanced accuracy on the predicted results of test set
            balanced_accuracy = self.evaluate(y_test, y_preds)
            balanced_accs.append(balanced_accuracy) 

            # Save the corresponding user_id
            test_groups.append(self.groups[test_index][0])
        
        results = { 'groups': test_groups, 'balanced_accurary_score': balanced_accs, 'cv_balanced_accuracy_score': cv_balanced_acc_scores }
        return results
 

    def exec_classifier(self):
        if self.cross_validation is True:
            return self.cross_validator(self.strategy)
        if self.logo_validation is True:
            return self.leave_one_group_out_validator(self.strategy)


    def evaluate(self, y_trues, y_preds):
        balanced_accuracy = balanced_accuracy_score(y_trues, y_preds)
        return balanced_accuracy              