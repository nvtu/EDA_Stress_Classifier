import os
import os.path as osp
import configparser
import pandas as pd


class ResultUtils:

    def __init__(self, dataset_name: str, WINDOW_SHIFT: int = 1, WINDOW_SIZE: int = 60):
        self.dataset_name = dataset_name
        self.WINDOW_SHIFT = WINDOW_SHIFT
        self.WINDOW_SIZE = WINDOW_SIZE


    def get_output_folder_path(self) -> str:
        config_path = osp.join(os.getcwd(), 'config.ini')
        parser = configparser.ConfigParser()
        parser.read(config_path)
        # Get output_folder_path for a specific dataset
        output_folder_path = osp.join(parser['DATA_PATH']['result_path'], self.dataset_name)
        # Create the output folder if it does not exist
        if not osp.exists(output_folder_path):
            os.makedirs(output_folder_path)
        return output_folder_path


    def dump_result_to_csv(self, results, detection_strategy: str, detector_type: str, physiological_signal_type: str = ""):
        output_folder_path = osp.join(self.get_output_folder_path(), detector_type)
        # Create the folder if it does not exist
        if not osp.exists(output_folder_path):
            os.makedirs(output_folder_path)
        # Get output_file_path
        output_file_path = osp.join(output_folder_path, f'{self.dataset_name}-{detection_strategy}{physiological_signal_type}-{self.WINDOW_SHIFT}-{self.WINDOW_SIZE}.csv')
        # Generate DataFrame to save to csv format
        df = pd.DataFrame.from_dict(results)
        df.to_csv(output_file_path, index=False)   

