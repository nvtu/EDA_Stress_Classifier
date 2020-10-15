from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, SelectFromModel
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools


class TryMLClassifierStrategy():
    

    def __init__(self, X: np.array, y: np.array, groups: np.array = None, balanced_weight: str = None, random_state: int = 0, multiclass: bool = False):
        self.X = X
        self.y = y
        self.groups = groups
        self.balanced_weight = balanced_weight
        self.random_state = random_state
        self.multiclass = multiclass

    
    def try_cross_validation(self, clf):
        stratetified_kfold = StratifiedKFold(n_splits=3)
        acc_train = []
        acc_test = []
        y_scores = []
        y_trues = []
        for train_index, test_index in stratetified_kfold.split(self.X, self.y):
            X_train, y_train, X_test, y_test = self.X[train_index], self.y[train_index], self.X[test_index], self.y[test_index]
            clf.fit(X_train, y_train)
            acc_test.append(clf.score(X_test, y_test))
            acc_train.append(clf.score(X_train, y_train))
            prob_scores = clf.predict_proba(X_test)
            if self.multiclass is True:
                y_scores.append(prob_scores)
            else:
                y_scores.append([prob[-1] for prob in prob_scores])
            y_trues.append(y_test)
        mean_acc_train = sum(acc_train) / len(acc_train)
        mean_acc_test = sum(acc_test) / len(acc_test)
        if self.multiclass is False:
            roc_scores = [roc_auc_score(y_trues[index], y_scores[index]) for index in range(len(y_trues))]
            print(f"ROC AUC Score: {roc_scores}")
        print(f"Mean accuracy of train set: {mean_acc_train}")
        print(f"Mean accuracy of test set: {mean_acc_test}")
    

    def try_group_kfold(self, clf):
        gkf = GroupKFold(n_splits=4)
        acc_train = []
        acc_test = []
        y_scores = []
        y_trues = []
        test_groups = []
        for train_index, test_index in gkf.split(self.X, self.y, groups=self.groups):
            X_train, y_train, X_test, y_test = self.X[train_index], self.y[train_index], self.X[test_index], self.y[test_index]
            clf.fit(X_train, y_train)
            acc_test.append(clf.score(X_test, y_test))
            acc_train.append(clf.score(X_train, y_train))
            prob_scores = clf.predict_proba(X_test)
            if self.multiclass is True:
                y_scores.append(prob_scores)
            else:
                y_scores.append([prob[-1] for prob in prob_scores])
            test_groups.append(self.groups[test_index])
            y_trues.append(y_test)
        mean_acc_train = sum(acc_train) / len(acc_train)
        mean_acc_test = sum(acc_test) / len(acc_test)
        test_groups = [list(set(group)) for group in test_groups]
        print(f"Groups: {test_groups}")
        if self.multiclass is False:
            roc_scores = [roc_auc_score(y_trues[index], y_scores[index]) for index in range(len(y_trues))]
            print(f"ROC AUC Score: {roc_scores}")
        print(f"Mean accuracy of train set: {mean_acc_train}")
        print(f"Mean accuracy of test set: {mean_acc_test}")
    

    def try_recursive_feature_elimination_pipepline(self, clf, cross_validation: bool = False, group_validation: bool = False):
        if cross_validation is True:
            stratetified_kfold = StratifiedKFold(n_splits=3)
            rfecv = RFECV(estimator=clf, step=1, cv=stratetified_kfold, scoring='accuracy')
            X_selected = rfecv.fit_transform(self.X, self.y)
            print(f"{rfecv.n_features_} features remain")
            print("------After recursive feature elimination------")
            ml_strategy = TryMLClassifierStrategy(X_selected, self.y, balanced_weight=self.balanced_weight, multiclass=self.multiclass)
            ml_strategy.try_cross_validation(clf)
        # if group_validation is True:
        #     group_kfold = GroupKFold(n_splits=4)
        #     rfecv = RFECV(estimator=clf, step=1, cv=group_kfold, scoring='accuracy')
        #     X_selected = rfecv.fit_transform(self.X, self.y)
        #     print(f"{rfecv.n_features_} features remain")
        #     print("------After recursive feature elimination------")
        #     ml_strategy = TryMLClassifierStrategy(X_selected, self.y, groups=self.groups, balanced_weight=self.balanced_weight, multiclass=self.multiclass)
        #     ml_strategy.try_group_kfold(clf)
            
    
    def try_ensemble_feature_selection(self, clf, cross_validation: bool = False, group_validation: bool = False):
        selector = SelectFromModel(estimator=clf)
        X_selected = selector.fit_transform(self.X, self.y)
        _, n_remaining_features = X_selected.shape
        print(f"{n_remaining_features} features remain")
        print("------After ensemble feature selection------")
        if cross_validation is True:
            ml_strategy = TryMLClassifierStrategy(X_selected, self.y, balanced_weight=self.balanced_weight, multiclass=self.multiclass)
            ml_strategy.try_cross_validation(clf)
        if group_validation is True:
            ml_strategy = TryMLClassifierStrategy(X_selected, self.y, groups=self.groups, balanced_weight=self.balanced_weight, multiclass=self.multiclass)
            ml_strategy.try_group_kfold(clf)


    def try_logistic_regression(self, cross_validation: bool = False, group_validation: bool = False):
        clf = LogisticRegression(random_state=self.random_state, class_weight=self.balanced_weight)
        # Try KFold
        if cross_validation is True:
            self.try_cross_validation(clf)
            self.try_recursive_feature_elimination_pipepline(clf, cross_validation = cross_validation)
        if group_validation is True:
            self.try_group_kfold(clf)
            self.try_recursive_feature_elimination_pipepline(clf, group_validation = group_validation)


    def try_random_forests(self, cross_validation: bool = False, group_validation: bool = False):
        clf = RandomForestClassifier(random_state=self.random_state)
        # Try KFold
        if cross_validation is True:
            self.try_cross_validation(clf)
            self.try_ensemble_feature_selection(clf, cross_validation=cross_validation)
        if group_validation is True:
            self.try_group_kfold(clf)
            self.try_ensemble_feature_selection(clf, group_validation = group_validation)
        

    def try_SVM(self, cross_validation: bool = False, group_validation: bool = False):
        clf = SVC(gamma='auto', random_state=self.random_state, probability=True)
        # Try KFold
        if cross_validation is True:
            self.try_cross_validation(clf)
        if group_validation is True:
            self.try_group_kfold(clf)


    def try_extra_trees_classifier(self, cross_validation: bool = False, group_validation: bool = False):
        clf = ExtraTreesClassifier(random_state=self.random_state)
        # Try KFold
        if cross_validation is True:
            self.try_cross_validation(clf)
            self.try_ensemble_feature_selection(clf, cross_validation=cross_validation)
        if group_validation is True:
            self.try_group_kfold(clf)
            self.try_ensemble_feature_selection(clf, group_validation=group_validation)
        

    def try_MLPClassifier(self, cross_validation: bool = False, group_validation: bool = False):
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=self.random_state)
        if cross_validation is True:
            self.try_cross_validation(clf)
        if group_validation is True:
            self.try_group_kfold(clf)

        
    def try_KNeighbors_Classifier(self, cross_validation: bool = False, group_validation: bool = False):
        clf = KNeighborsClassifier(n_neighbors=10, weights='distance')
        if cross_validation is True:
            self.try_cross_validation(clf)
        if group_validation is True:
            self.try_group_kfold(clf)


    def try_desicision_tree(self, cross_validation: bool = False, group_validation: bool = False):
        clf = DecisionTreeClassifier(random_state=self.random_state)
        if cross_validation is True:
            self.try_cross_validation(clf)
        if group_validation is True:
            self.try_group_kfold(clf)


    def try_xgboost_classifier(self, cross_validation: bool = False, group_validation: bool = False):
        clf = XGBClassifier()
        if cross_validation is True:
            self.try_cross_validation(clf)
        if group_validation is True:
            self.try_group_kfold(clf)


    def try_lda_classifier(self, cross_validation: bool = False, group_validation: bool = False):
        clf = LinearDiscriminantAnalysis()
        if cross_validation is True:
            self.try_cross_validation(clf)
        if group_validation is True:
            self.try_group_kfold(clf)


    def try_different_strategies(self, cross_validation: bool = False, group_validation: bool = False):
        # Try LogisticRegression
        print("Try Logistic Regression...")
        self.try_logistic_regression(cross_validation = cross_validation, group_validation = group_validation)
        print('--------------------------------------')
        # Try Random Forests
        print("Try Random Forests...")
        self.try_random_forests(cross_validation = cross_validation, group_validation = group_validation)
        print('--------------------------------------')
        # Try Extra Trees Classifiers
        print("Try Extra Trees Classifier...")
        self.try_extra_trees_classifier(cross_validation = cross_validation, group_validation = group_validation)
        print('--------------------------------------')
        # Try SVM
        print("Try SVM Classifier...")
        self.try_SVM(cross_validation = cross_validation, group_validation = group_validation)
        print('--------------------------------------')
        # Try MLPClassifier
        print("Try MLPClassifier...")
        self.try_MLPClassifier(cross_validation = cross_validation, group_validation = group_validation)
        print('--------------------------------------')
        # Try KNN
        print("Try 10-nearest-neighbors...")
        self.try_KNeighbors_Classifier(cross_validation = cross_validation, group_validation = group_validation)
        print('--------------------------------------')
        # Try Decision Tree alone
        print("Try decision trees...")
        self.try_desicision_tree(cross_validation = cross_validation, group_validation = group_validation)
        print('--------------------------------------')
        # Try XGBoostClassifier
        print("Try XGBoostClassifier...")
        self.try_xgboost_classifier(cross_validation = cross_validation, group_validation = group_validation)
        print('--------------------------------------')
        # Try Linear Discriminate Analysis
        print("Try Linear Discriminate Analysis...")
        self.try_lda_classifier(cross_validation = cross_validation, group_validation = group_validation)