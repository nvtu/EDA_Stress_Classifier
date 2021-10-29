import torch
import os
import numpy as np
import os.path as osp
# from AUTOENCODER.autoencoder import Encoder
from autoencoder import Encoder
import json


ntcir14_path = '/'.join(osp.abspath(__file__).split('/')[:-3])
data_path = osp.join(ntcir14_path, 'DATA')
encoder_file_path = osp.join(data_path, 'processed_data', 'autoencoder', 'encoder.pt')
encoder_config_path = osp.join(data_path, 'processed_data', 'autoencoder', 'config.json')
bow_matrix_path = osp.join(data_path, 'processed_data', 'bow', 'combined_bow_matrix.npy')
output_path = osp.join(data_path, 'processed_data', 'autoencoder')


"""
Encode the feature vector by loading pretrained models
"""

def load_autoencoder(encoder_file_path, encoder_config_path):
    config = json.load(open(encoder_config_path))
    input_dimension = config['input_dimension']
    hidden_dimensions = config['hidden_dimensions']
    code_dimension = config['code_dimension']
    encoder = Encoder(input_dimension, hidden_dimensions, code_dimension)
    encoder.load_state_dict(torch.load(encoder_file_path))
    encoder.eval()
    return encoder


def encode(feature_matrix, encoder):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    feature_matrix = torch.Tensor(feature_matrix).to(device)
    encoded_feature_matrix = encoder(feature_matrix).detach().cpu().numpy()
    return encoded_feature_matrix


if __name__ == '__main__':
    combined_matrix = np.load(bow_matrix_path)
    encoder = load_autoencoder(encoder_file_path, encoder_config_path)
    encoded_feature_matrix = encode(combined_matrix, encoder) 
    output_file_path = osp.join(output_path, 'combined_encoded_bow_matrix.npy')
    np.save(output_file_path, encoded_feature_matrix)
    