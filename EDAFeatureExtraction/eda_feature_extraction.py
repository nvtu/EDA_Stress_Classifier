# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import neurokit2 as nk
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import configparser
import os
import os.path as osp
import pickle
from eda_signal_processing import *
from typing import Dict, List
from tqdm import tqdm
import scipy

# %% [markdown]
# # Load eda signal from datasets
# %% [markdown]
# ## Initialize file paths

# %%
def get_dataset_folder_path(dataset_name: str) -> str:
    # Read dataset path from config.ini file
    config_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'config.ini')
    parser = configparser.ConfigParser()
    parser.read(config_path)
    dataset_folder_path = None
    if dataset_name == 'AffectiveROAD':
        dataset_folder_path = parser['DATA_PATH']['affectiveROAD_dataset_path']
    elif dataset_name in ['WESAD_CHEST', 'WESAD_WRIST']:
        dataset_folder_path = parser['DATA_PATH']['wesad_dataset_path']
    elif dataset_name == 'DCU_NVT_EXP1':
        dataset_folder_path = parser['DATA_PATH']['dcu_nvt_dataset_path']
    return dataset_folder_path


# %%
def load_raw_dataset(dataset_name: str):
    dataset = None
    # Initialize dataset folder paths
    dataset_folder_path = get_dataset_folder_path(dataset_name)
    if dataset_name == 'AffectiveROAD':
        # Initialize dataset paths
        affectiveROAD_dataset_file_path = osp.join(dataset_folder_path, 'affectiveROAD_dataset.pkl')
        dataset = pickle.load(open(affectiveROAD_dataset_file_path, 'rb')) # Load affectiveROAD dataset -> sampling_rate = 4 Hz
    elif dataset_name == 'WESAD_CHEST':
        # Initialize dataset paths
        wesad_chest_file_path = osp.join(dataset_folder_path, 'wesad_chest_dataset.pkl')
        dataset = pickle.load(open(wesad_chest_file_path, 'rb')) # Load WESAD_CHEST dataset -> sampling_rate = 700 Hz
    elif dataset_name == 'WESAD_WRIST':
        # Initialize dataset paths
        wesad_wrist_file_path = osp.join(dataset_folder_path, 'wesad_wrist_dataset.pkl')
        dataset = pickle.load(open(wesad_wrist_file_path, 'rb')) # Load WESAD_WRIST dataset -> sampling_rate = 4 Hz
    elif dataset_name == 'DCU_NVT_EXP1':
        # Initialize dataset paths
        dcu_nvt_file_path = osp.join(dataset_folder_path, 'DCU_NVT_EXP1_dataset.pkl')
        dataset = pickle.load(open(dcu_nvt_file_path, 'rb')) # Load DCU_NVT_EXP1 dataset -> sampling_rate = 5 Hz
    return dataset

# %% [markdown]
# ## Load datasets

# %%
# -- Uncomment the dataset that you wanna load -- # 
# dataset_name = 'AffectiveROAD'
# dataset_name = 'WESAD_CHEST'
dataset_name = 'WESAD_WRIST'
# dataset_name = 'DCU_NVT_EXP1'


# %%
dataset = load_raw_dataset(dataset_name) # Load dataset
eda = dataset['eda'] # Get raw EDA signal
ground_truth = dataset['ground_truth'] # Get its corresponding ground-truth

# %% [markdown]
# # Extract statistical features
# %% [markdown]
# ## Declare functions to process and extract statistical features

# %%
# Extract statistical features from the EDA signal with a current WINDOW_SIZE and WINDOW_SHIFT
def extract_stats_features(eda: Dict[str, Dict[str, List[float]]], window_size: int, window_shift: int, sampling_rate: int) -> np.array:
    """ This function extract stats feature corresponding to left-side of the current eda signal with length equals to window_size """ 
    # window_size: unit -> seconds - the length of signal which is cut to extract statistical feature equals to 60 seconds
    # window_shift: unit -> seconds - the step of the sliding window
    # sampling_rate: unit -> Hz - the number of recorded points per second
    stats_features = []
    for user_id, data in tqdm(eda.items()):
        for task_id, eda_signal in data.items():
            len_eda_signal = len(eda_signal)
            step = int(window_shift * sampling_rate) # The true step to slide along the time axis of the signal
            first_iter = int(window_size * sampling_rate) # The true index of the signal at a time-point 
            for current_iter in range(first_iter, len_eda_signal, step): # current_iter is "second_iter"
                previous_iter = current_iter - first_iter
                signal = eda_signal[previous_iter:current_iter]
                eda_features = extract_eda_features(signal, sampling_rate) # Extract SCR, SCL, Onset, Offset, Peaks, etc.
                eda_stats_features = extract_statistics_eda_features(eda_features) # Extract statistical features from extracted EDA features
                stats_features.append(eda_stats_features)
    stats_features = np.array(stats_features) # Transform to numpy array format
    return stats_features


# %%
def map_ground_truth(ground_truth: Dict[str, Dict[str, List[int]]], window_size: int, window_shift: int, sampling_rate: int) -> np.array: 
    """ This function should be call after the function extract_stats_features is called.
        The iterative order of this function is the same as extract_stats_features function to maintain the integrity of the dataset.
    """
    # window_size: unit -> seconds - the length of signal which is cut to extract statistical feature equals to 60 seconds
    # window_shift: unit -> seconds - the step of the sliding window
    # sampling_rate: unit -> Hz - the number of recorded points per second
    gt = []
    for user_id, data in tqdm(ground_truth.items()):
        for task_id, _ground_truth in data.items():
            len_ground_truth = len(_ground_truth)
            start_index = int(window_size * sampling_rate) # The true index of the signal at a time-point
            step = int(window_shift * sampling_rate)
            gt += [_ground_truth[index] for index in range(start_index, len_ground_truth, step)] # Append the flatten array 
    gt = np.array(gt) # Transform to numpy array format
    return gt


# %%
def generate_data_groups_from_ground_truth(ground_truth: Dict[str, Dict[str, List[int]]], window_size: int, window_shift: int, sampling_rate: int) -> np.array:
    """ This function should be call after the function extract_stats_features is called.
        The iterative order of this function is the same as extract_stats_features and map_ground_truth functions to maintain the integrity of the dataset.
        Generate user_id label for each statistical features using ground-truth
    """
    # window_size: unit -> seconds - the length of signal which is cut to extract statistical feature equals to 60 seconds
    # window_shift: unit -> seconds - the step of the sliding window
    # sampling_rate: unit -> Hz - the number of recorded points per second
    groups = []
    for user_id, data in tqdm(ground_truth.items()):
        for task_id, _ground_truth in data.items():
            len_ground_truth = len(_ground_truth)
            start_index = int(window_size * sampling_rate)
            step = int(window_shift * sampling_rate)
            groups += [user_id for _ in range(start_index, len_ground_truth, step)]
    groups = np.array(groups)
    return groups


# %%
def get_sampling_rate(dataset_name: str) -> int:
    sampling_rate = None
    if dataset_name in ['AffectiveROAD', 'WESAD_WRIST']:
        sampling_rate = 4
    elif dataset_name == 'WESAD_CHEST':
        sampling_rate = 700
    elif dataset_name == 'DCU_NVT_EXP1':
        sampling_rate = 5
    return sampling_rate

# %% [markdown]
# ## Extract statistical features

# %%
WINDOW_SIZE = 60 # the length of signal which is cut to extract statistical feature equals to 60 seconds
WINDOW_SHIFT = 20 # the step of the sliding window 
SAMPLING_RATE = get_sampling_rate(dataset_name)


# %%
# Extract EDA statistical features 
eda_stats_features = extract_stats_features(eda, WINDOW_SIZE, WINDOW_SHIFT, SAMPLING_RATE)


# %%
# Map ground-truth to the features 
mapped_ground_truth = map_ground_truth(ground_truth, WINDOW_SIZE, WINDOW_SHIFT, SAMPLING_RATE)


# %%
# Label the group of the data also
groups = generate_data_groups_from_ground_truth(ground_truth, WINDOW_SIZE, WINDOW_SHIFT, SAMPLING_RATE)

# %% [markdown]
# ## Save extracted features and their corresponding ground-truth

# %%
dataset_folder_path = get_dataset_folder_path(dataset_name)


# %%
# Save the features to files in .npy format
feat_output_file_path = osp.join(dataset_folder_path, f'{dataset_name}_stats_feats.npy')
np.save(feat_output_file_path, eda_stats_features)


# %%
# Save the ground-truth of the corresponding signal at its corresponding time-point
gt_output_file_path = osp.join(dataset_folder_path, f'{dataset_name}_ground_truth.npy')
np.save(gt_output_file_path, mapped_ground_truth)


# %%
# Save the user_id mapping of the statistical features
groups_output_file_path = osp.join(dataset_folder_path, f'{dataset_name}_groups.npy')
np.save(groups_output_file_path, groups) 


# %%



