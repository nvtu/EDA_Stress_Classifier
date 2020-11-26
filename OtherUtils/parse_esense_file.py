import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import platform


if __name__ == '__main__':
    parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    if platform.system() == 'Windows':
        parent_dir = parent_dir.replace('\\', '/')
    raw_data_dir = '/'.join([parent_dir, 'MyDataset', 'raw_eSense_Skin_Response'])
    data_dir = osp.join(parent_dir, 'MyDataset', 'GSR')
    if not osp.exists(data_dir):
        os.makedirs(data_dir)

    file_path_list = [osp.join(raw_data_dir, f) for f in sorted(os.listdir(raw_data_dir)) if f.split('.')[-1] == 'csv']   
    for file_path in tqdm(file_path_list):
        data = []
        # Get necessary labels & metadata
        content = [line.rsplit() for line in open(file_path, 'r').readlines()]
        date = content[3][-2].replace('.', '-').split(';')[-1]
        start_time = content[3][-1].replace(';', '') 
        duration = content[4][-1].split(';')[1]

        # Parse raw data signal
        file_label = None
        start_index = 5
        for i in range(start_index, len(content) - 1):
            if content[i][0].replace(';', '') == 'STATISTICS':
                start_index = i
        start_index += 1
        labels = content[start_index][0].split(';')[:-1]
        start_index += 2
        content = content[start_index:-1]
        for line in content:
            signal_data = line[0].split(';')[:-1]
            data.append(signal_data)

        # Save to file
        output_file_path = file_path.replace(raw_data_dir, data_dir)
        df = pd.DataFrame(data=data, columns=labels) 
        df.to_csv(output_file_path, index=False)