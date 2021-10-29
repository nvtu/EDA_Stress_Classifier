import configparser
import os
import os.path as osp
import numpy as np
from numpy.lib.function_base import i0


class DatasetLoader:


    def __init__(self, dataset_name, WINDOW_SIZE = 60, WINDOW_SHIFT = 1):
        self.WINDOW_SIZE = WINDOW_SIZE 
        self.WINDOW_SHIFT = WINDOW_SHIFT
        self.dataset, self.ground_truth, self.groups = self.load_dataset(dataset_name, physiological_signal_type = '') # Load combined stats features
        self.hrv_dataset, _, _ = self.load_dataset(dataset_name, physiological_signal_type = '_heart')
        self.eda_dataset, _, _ = self.load_dataset(dataset_name, physiological_signal_type = '_eda')


    def get_dataset_folder_path(self, dataset_name: str) -> str:
        # Read dataset path from config.ini file
        config_path = osp.join(os.getcwd(), 'config.ini')
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


    def load_dataset(self, dataset_name: str, physiological_signal_type: str = ""):
        dataset = None
        ground_truth = None
        # Initialize dataset folder path
        dataset_folder_path = self.get_dataset_folder_path(dataset_name)
        # Initialize dataset file path
        dataset_file_path = osp.join(dataset_folder_path, f'{dataset_name}{physiological_signal_type}_stats_feats_{self.WINDOW_SHIFT}_{self.WINDOW_SIZE}.npy')
        # Initialize ground-truth file path
        ground_truth_file_path = osp.join(dataset_folder_path, f'{dataset_name}_ground_truth_{self.WINDOW_SHIFT}_{self.WINDOW_SIZE}.npy')
        # Initialize group file path
        group_file_path = osp.join(dataset_folder_path, f'{dataset_name}_groups_{self.WINDOW_SHIFT}_{self.WINDOW_SIZE}.npy')

        # Load dataset, ground-truth, and groups
        dataset = np.load(dataset_file_path) # Load dataset
        ground_truth = np.load(ground_truth_file_path) # Load corresponding ground-truth
        groups = np.load(group_file_path) # Load corresponding user_id labels

        dataset[np.isnan(dataset)] = 0
        # Filtering preprocess if dataset name is AffectiveROAD
        if dataset_name == 'AffectiveROAD':
            # print(set(ground_truth))
            indices = np.where(ground_truth >= 0)[0]
            dataset = dataset[indices]
            groups = groups[indices]
            ground_truth = ground_truth[indices]

        return dataset, ground_truth, groups  