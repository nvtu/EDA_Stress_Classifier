import configparser
import os
import os.path as osp
import pandas as pd
from collections import defaultdict
from typing import Tuple, Dict, List
from .signal_processing import *
from collections import Counter
import pickle


class DatasetLoader:

    def __init__(self):
        grandparent_dir_name = osp.dirname(osp.dirname(osp.abspath(__file__)))
        config_file_path = osp.join(grandparent_dir_name, 'config.ini')
        parser = configparser.ConfigParser()
        parser.read(config_file_path)
        self.collected_gsr_path = parser['DATA_PATH']['gsr_dataset_path']
        self.collected_gsr_groundtruth_path = osp.join(osp.dirname(self.collected_gsr_path), 'Ground-Truth.csv')
        self.wesad_gsr_path = parser['DATA_PATH']['wesad_dataset_path']
        self.wesad_gsr_groundtruth_path = osp.join(osp.dirname(self.wesad_gsr_path), 'WESAD-Ground-Truth.csv')


    def load_wesad_gsr_dataset(self) -> Tuple[ Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame ]:
        gsr_data, groundtruth_df = self.__load_dataset(self.wesad_gsr_path, self.wesad_gsr_groundtruth_path)
        return gsr_data, groundtruth_df


    def load_collected_gsr_dataset(self) -> Tuple[ Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame ]:
        gsr_data, groundtruth_df = self.__load_dataset(self.collected_gsr_path, self.collected_gsr_groundtruth_path)
        return gsr_data, groundtruth_df


    def __load_dataset(self, dataset_path, ground_truth_path) -> Tuple[ Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame ]:
        # Load participants' GSR dataset
        gsr_data = defaultdict(dict)
        participant_ids = sorted(os.listdir(dataset_path))
        participant_data_path = [osp.join(dataset_path, participant_id) for participant_id in participant_ids]
        # Iterate through each participant's data folder
        for index, participant_data in enumerate(participant_data_path):
            task_names = sorted(os.listdir(participant_data))
            gsr_by_tasks = [osp.join(participant_data, task_name) for task_name in task_names]
            participant_id = participant_ids[index]
            # Iterate through each participant's gsr data by task
            for _index, gsr_by_task in enumerate(gsr_by_tasks):
                task_name = task_names[_index]
                df = pd.read_csv(gsr_by_task)
                gsr_data[participant_id][task_name] = df

        # Load ground-truth of the dataset
        groundtruth_df = pd.read_csv(ground_truth_path)
        return gsr_data, groundtruth_df

    
    # This function will not work with the updated code
    def aggregate_gsr_dataset(self, dataset: Dict[str, Dict[str, pd.DataFrame]], selected_columns: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        agg_dataset = defaultdict(dict)
        for participant_id, data in dataset.items():
            for task_id, gsr_data in data.items():
                agg_gsr_data = aggregate_signal_data(gsr_data[selected_columns], 5)
                agg_dataset[participant_id][task_id] = agg_gsr_data
        return agg_dataset


    def divide_into_intervals(self, dataset: Dict[str, Dict[str, np.array]], ground_truth: pd.DataFrame, num_samples: int, sampling_rate = 5) -> \
                            Tuple[ Dict[str, Dict[str, List[np.array]]], Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]]]:
        # num_samples should be in second unit
        interval_dataset = defaultdict(dict)
        interval_ground_truth = defaultdict(dict)
        interval_group = defaultdict(dict)
        ground_truth = ground_truth.set_index("INSTANCE")
        for participant_id, data in dataset.items():
            for task_id, gsr_data in data.items():
                interval_dataset[participant_id][task_id] = []
                interval_ground_truth[participant_id][task_id] = []
                interval_group[participant_id][task_id] = []
                n_data = len(gsr_data)
                index = 0
                while index < n_data:
                    lower_bound = index
                    upper_bound = min(index + num_samples * sampling_rate, n_data) # If the number of remaining data is not enough
                    index += num_samples * sampling_rate
                    data = gsr_data[lower_bound:upper_bound]
                    data_ground_truth = ground_truth.loc[task_id].values
                    interval_dataset[participant_id][task_id].append(data)
                    interval_ground_truth[participant_id][task_id].append(data_ground_truth)
                    interval_group[participant_id][task_id].append(participant_id)
        return interval_dataset, interval_ground_truth, interval_group


    def divide_person_specific_data_into_intervals(self, dataset: np.array, ground_truth: np.array, num_samples: int, sampling_rate = 5) -> Tuple[ np.array, np.array ]:
        # num_samples should be in second unit
        interval_dataset = []
        interval_ground_truth = []
        for data_index, data in enumerate(dataset):
            n_data = len(data)
            index = 0
            while index < n_data:
                low_bound = index
                upper_bound = min(index + num_samples * sampling_rate, n_data) # If the number of remaining data is not enough
                index += num_samples * sampling_rate
                _data = data[low_bound:upper_bound]
                interval_dataset.append(_data)
                interval_ground_truth.append(ground_truth[data_index])
        interval_dataset = np.array(interval_dataset)
        interval_ground_truth = np.array(interval_ground_truth)
        return interval_dataset, interval_ground_truth

    
    def prepare_person_specific_dataset(self, dataset: Dict[str, np.array], ground_truth: pd.DataFrame, selected_columns = None) -> Tuple[ np.array, np.array ]:
        # dataset: [participant_id][task_id] -> pd.DataFrame
        ground_truth = ground_truth.set_index("INSTANCE")
        prepared_dataset = [] 
        prepared_ground_truth = []
        for task_id, gsr_data in dataset.items():
            prepared_dataset.append(gsr_data)
            prepared_ground_truth.append(ground_truth.loc[task_id].values)
        prepared_dataset = np.array(prepared_dataset)
        prepared_ground_truth = np.array(prepared_ground_truth)
        return prepared_dataset, prepared_ground_truth
    

    def flatten(self, dataset: Dict[str, Dict[str, List[object]]]) -> np.array:
        output = []
        for participant_id, data in dataset.items():
            for task_id, _data in data.items():
                output += _data
        output = np.array(output)
        return output


    def class_percentage_analysis(self, ground_truth: np.array):
        total_cnt = len(ground_truth)
        class_cnt = Counter(ground_truth)
        for class_id, cnt in class_cnt.items():
            print(f"{class_id}: {cnt * 1.0 / total_cnt}")