# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import os.path as osp
import numpy as np
import configparser
from classifiers import BinaryClassifier
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from dataset_loader import *
from result_utils import *

# %% [markdown]
# # Load eda statistical features and ground-truth from datasets
# %%
# -- Uncomment the dataset that you wanna load -- #
dataset_name = 'AffectiveROAD'
dataset_name = 'WESAD_CHEST'
dataset_name = 'WESAD_WRIST'
# dataset_name = 'DCU_NVT_EXP1'

# %%
WINDOW_SHIFT = 0.25
WINDOW_SIZE = 120

# %%
ds_loader = DatasetLoader(dataset_name, WINDOW_SHIFT = WINDOW_SHIFT, WINDOW_SIZE = WINDOW_SIZE)
# eda_feature_indices = np.array([0, 2, 3, 5, 7, 9, 11, 12, 13, 15, 17, 23, 24, 26, 27, 29, 30, 32, 33, 34]) + 25
# hrv_feature_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 20, 21, 22, 23, 24])
# feature_indices = np.concatenate([hrv_feature_indices, eda_feature_indices])
# dataset = dataset[:, feature_indices]

# %% [markdown]
# # Define stress detection strategies

# %%
# -- Uncomment the detection strategy that you wanna use to detect -- #
strategies = ['random_forest'] #, 'mlp']
# detection_strategy = 'logistic_regression'
# detection_strategy = 'random_forest'
# detection_strategy = 'svm'
# detection_strategy = 'mlp'
# detection_strategy = 'knn'


# %%
# SCORING = 'accuracy'
SCORING = 'balanced_accuracy'


# %% [markdown]
# # Build General Cross-population Stress Detector

# %%
for detection_strategy in strategies:
    detector_type = 'General'
    print(f'--- RUNNING {detector_type} {detection_strategy} ---')
    # clf = BinaryClassifier(ds_loader.dataset, ds_loader.ground_truth, detection_strategy, logo_validation = True, groups = ds_loader.groups, scoring = SCORING)
    # clf = BinaryClassifier(ds_loader.hrv_dataset, ds_loader.ground_truth, detection_strategy, logo_validation = True, groups = ds_loader.groups, scoring = SCORING)
    clf = BinaryClassifier(ds_loader.eda_dataset, ds_loader.ground_truth, detection_strategy, logo_validation = True, groups = ds_loader.groups, scoring = SCORING)
    results = clf.exec_classifier() # Build classifier and return prediction results
    print('------------------------------------------------------')

    # # %%
    # # Save results
    result_helper = ResultUtils(dataset_name, WINDOW_SHIFT = WINDOW_SHIFT, WINDOW_SIZE = WINDOW_SIZE)
    result_helper.dump_result_to_csv(results, detection_strategy, detector_type, physiological_signal_type = '_eda')

# %% [markdown]
# # Build Person-specific Stress Detector

# %%
for detection_strategy in strategies:
    detector_type = 'Personal'

    print(f'--- RUNNING {detector_type} {detection_strategy} ---')
    # clf = BinaryClassifier(ds_loader.dataset, ds_loader.ground_truth, detection_strategy, cross_validation = True, groups = ds_loader.groups, scoring = SCORING)
    # clf = BinaryClassifier(ds_loader.hrv_dataset, ds_loader.ground_truth, detection_strategy, cross_validation = True, groups = ds_loader.groups, scoring = SCORING)
    clf = BinaryClassifier(ds_loader.eda_dataset, ds_loader.ground_truth, detection_strategy, cross_validation = True, groups = ds_loader.groups, scoring = SCORING)
    results = clf.exec_classifier()
    print('------------------------------------------------------')

    # %%
    # Save results
    result_helper = ResultUtils(dataset_name, WINDOW_SHIFT = WINDOW_SHIFT, WINDOW_SIZE = WINDOW_SIZE)
    result_helper.dump_result_to_csv(results, detection_strategy, detector_type, physiological_signal_type = '_eda')


