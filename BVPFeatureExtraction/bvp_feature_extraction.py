# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import neurokit2 as nk
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import configparser
import os
import os.path as osp
import pickle
from bvp_signal_processing import *
from typing import Dict, List
from tqdm import tqdm
import scipy
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Load bvp signal from datasets
# %% [markdown]
# ## Initialize file paths

# %%
def get_dataset_folder_path(dataset_name: str) -> str:
    # Read dataset path from config.ini file
    config_path = osp.join(osp.dirname(os.getcwd()), 'config.ini')
    parser = configparser.ConfigParser()
    parser.read(config_path)
    dataset_folder_path = None
    if dataset_name == 'AffectiveROAD':
        dataset_folder_path = parser['DATA_PATH']['affectiveROAD_dataset_path']
    elif dataset_name in ['WESAD_CHEST', 'WESAD_WRIST', 'RESAMPLED_WESAD_CHEST']:
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
    elif dataset_name == 'RESAMPLED_WESAD_CHEST':
        # Initialize dataset paths
        resampled_wesad_file_path = osp.join(dataset_folder_path, 'wesad_chest_resampling_dataset.pkl')
        dataset = pickle.load(open(resampled_wesad_file_path, 'rb')) # Load RESAMPLED_WESAD_CHEST dataset -> sampling_rate = 4 Hz
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
dataset_name = 'AffectiveROAD'
dataset_name = 'WESAD_CHEST'
dataset_name = 'WESAD_WRIST'
# dataset_name = 'RESAMPLED_WESAD_CHEST'
# dataset_name = 'DCU_NVT_EXP1'


# %%
dataset = load_raw_dataset(dataset_name) # Load dataset
bvp = dataset['bvp'] # Get raw BVP signal
ground_truth = dataset['ground_truth'] # Get its corresponding ground-truth

# %% [markdown]
# # Extract statistical features
# %% [markdown]
# ## Declare functions to process and extract statistical features

# %%
# Extract statistical features from the BVP signal with a current WINDOW_SIZE and WINDOW_SHIFT
def extract_stats_features(bvp: Dict[str, Dict[str, List[float]]], window_size: int, window_shift: int, sampling_rate: int) -> np.array:
    """ This function extract stats feature corresponding to left-side of the current bvp signal with length equals to window_size """ 
    # window_size: unit -> seconds - the length of signal which is cut to extract statistical feature equals to 60 seconds
    # window_shift: unit -> seconds - the step of the sliding window
    # sampling_rate: unit -> Hz - the number of recorded points per second
    stats_features = []
    for user_id, data in tqdm(bvp.items()):
        for task_id, bvp_signal in data.items():
            len_bvp_signal = len(bvp_signal)
            step = int(window_shift * sampling_rate) # The true step to slide along the time axis of the signal
            first_iter = int(window_size * sampling_rate) # The true index of the signal at a time-point 
            for current_iter in range(first_iter, len_bvp_signal, step): # current_iter is "second_iter"
                previous_iter = current_iter - first_iter
                signal = bvp_signal[previous_iter:current_iter]
                bvp_stats_features = extract_bvp_features(signal, sampling_rate) # Extract statistical features from extracted BVP features
                stats_features.append(bvp_stats_features)
    stats_features = np.array(stats_features) # Transform to numpy array format
    return stats_features


# %%
def get_sampling_rate(dataset_name: str) -> int:
    sampling_rate = None
    if dataset_name in ['AffectiveROAD', 'WESAD_WRIST', 'RESAMPLED_WESAD_CHEST']:
        sampling_rate = 64
    elif dataset_name == 'WESAD_CHEST':
        sampling_rate = 700
    elif dataset_name == 'DCU_NVT_EXP1':
        sampling_rate = 5
    return sampling_rate

# %% [markdown]
# ## Extract statistical features

# %%
WINDOW_SIZE = 120 # the length of signal which is cut to extract statistical feature equals to 60 seconds
WINDOW_SHIFT = 0.25 # the step of the sliding window 
SAMPLING_RATE = get_sampling_rate(dataset_name)


# %%
# Extract BVP statistical features 
bvp_stats_features = extract_stats_features(bvp, WINDOW_SIZE, WINDOW_SHIFT, SAMPLING_RATE)

# %% [markdown]
# ## Save extracted features and their corresponding ground-truth

# %%
dataset_folder_path = get_dataset_folder_path(dataset_name)


# %%
# Save the features to files in .npy format
feat_output_file_path = osp.join(dataset_folder_path, f'{dataset_name}_heart_stats_feats_{WINDOW_SHIFT}_{WINDOW_SIZE}.npy')
np.save(feat_output_file_path, bvp_stats_features)


# %%
np.any(np.isnan(bvp_stats_features) == True)


# %%
np.argwhere(np.isnan(bvp_stats_features))


# %%



