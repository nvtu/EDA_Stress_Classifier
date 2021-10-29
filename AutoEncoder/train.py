from autoencoder import Encoder, Decoder
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
import os
import os.path as osp
import json
import configparser

"""
Train autoencoder. 
You need to divide your dataset into train and test set to evaluate the performance of the encoder & decoder.
The dataset should be divided into 80/20 or 90/10.
"""

# %%
def get_dataset_folder_path(dataset_name: str) -> str:
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


# %%
def load_dataset(dataset_name: str):
    dataset = None
    ground_truth = None
    # Initialize dataset folder path
    dataset_folder_path = get_dataset_folder_path(dataset_name)
    # Initialize dataset file path
    dataset_file_path = osp.join(dataset_folder_path, f'{dataset_name}_stats_feats_1.npy')
    # Initialize ground-truth file path
    ground_truth_file_path = osp.join(dataset_folder_path, f'{dataset_name}_ground_truth_1.npy')
    # Initialize group file path
    group_file_path = osp.join(dataset_folder_path, f'{dataset_name}_groups_1.npy')

    # Load dataset, ground-truth, and groups
    dataset = np.load(dataset_file_path) # Load dataset
    ground_truth = np.load(ground_truth_file_path) # Load corresponding ground-truth
    groups = np.load(group_file_path) # Load corresponding user_id labels

    dataset[np.isnan(dataset)] = 0
    # Filtering preprocess if dataset name is AffectiveROAD
    if dataset_name == 'AffectiveROAD':
        indices = np.where(ground_truth >= 0)[0]
        dataset = dataset[indices]
        groups = groups[indices]
        ground_truth = ground_truth[indices]

    return dataset, ground_truth, groups


def train(train_data, test_data, load_pretrained=False, pretrained_path=None):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    train_data = torch.Tensor(train_data).to(device)
    _test_data = torch.Tensor(test_data).to(device)
    
    num_data, input_dimension = train_data.shape # input dimension = dictionary size ~ 700
    hidden_dimensions = [1024, 728]
    code_dimension = 512

    total_steps = 500
    learning_rate = 1e-4
    iter_per_batch = 10
    batch_size = num_data // iter_per_batch

    config_dict = {
        'input_dimension': input_dimension,
        'hidden_dimensions': hidden_dimensions,
        'code_dimension': code_dimension
    }
    config_file_path = osp.join(output_path, 'config.json')
    with open(config_file_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    encoder = Encoder(input_dimension, hidden_dimensions, code_dimension)
    encoder.to(device)
    decoder = Decoder(input_dimension, hidden_dimensions, code_dimension)
    decoder.to(device)
    if load_pretrained == True:
        if pretrained_path != None:
            encoder_file_path, decoder_file_path = pretrained_path
            encoder.load_state_dict(torch.load(encoder_file_path))
            decoder.load_state_dict(torch.load(decoder_file_path))
        else:
            raise AssertionError("Pretrained path must be provided!!!")

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    best_loss = float("Inf")

    for i in tqdm(range(total_steps)):
        # Get next batch
        batch_index = i % iter_per_batch * batch_size
        X = train_data[batch_index : min(num_data, batch_index + batch_size)]
        out = encoder(X)
        out = decoder(out)
        loss = loss_func(out, X)
        loss.backward()
        optimizer.step()
        encoder.zero_grad()
        decoder.zero_grad()
        
        # Evaluate autoencoder performance on test data
        out = decoder(encoder((_test_data))).detach().cpu().numpy()
        test_data_loss = np.mean(np.sqrt(np.sum((out - test_data)**2, axis=1)))
        print("-- Train loss: {}".format(loss))
        print("-- Test loss: {}".format(test_data_loss))
        
        if test_data_loss < best_loss:
            best_loss = test_data_loss
            encoder_file_path = osp.join(output_path, 'encoder.pt')
            decoder_file_path = osp.join(output_path, 'decoder.pt')
            torch.save(encoder.state_dict(), encoder_file_path)
            torch.save(decoder.state_dict(), decoder_file_path)

if __name__ == '__main__':
    # -- Uncomment the dataset that you wanna load -- #
    # dataset_name = 'AffectiveROAD'
    # dataset_name = 'WESAD_CHEST'
    dataset_name = 'WESAD_WRIST'
    # dataset_name = 'DCU_NVT_EXP1'


    # %%
    dataset, ground_truth, groups = load_dataset(dataset_name) # Load dataset and ground-truths
    train_data, test_data = split_train_test(data)
    encoder_file_path = osp.join(output_path, 'encoder.pt')
    decoder_file_path = osp.join(output_path, 'decoder.pt')
    train(train_data, test_data, load_pretrained=False, pretrained_path=(encoder_file_path, decoder_file_path))


