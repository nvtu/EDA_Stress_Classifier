import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict
import neurokit2 as nk
from scipy.stats import kurtosis, skew
import math


def aggregate_signal_data(data: pd.DataFrame, sampling_rate: int) -> pd.DataFrame:
    # data variables should contain two columns: SECOND and MICROSIEMENS
    
    aggregated_data = []
    signal_data = []
    for index, row in enumerate(data.values):
        if (int(index) % sampling_rate) == 0 and int(index) > 0:
            avg_signal_data = np.average(signal_data, axis=0)
            signal_data = []
            aggregated_data.append(avg_signal_data.tolist())
        signal_data.append(row)
    
    df = pd.DataFrame(data=aggregated_data, columns=data.columns)
    return df


def select_single_signal(data: Dict[str, Dict[str, object]], feature_index: int) -> Dict[str, Dict[str, object]]:
    output = defaultdict(dict)
    for participant_id, dataset in data.items():
        for task_id, signal_data in dataset.items():
            selected_signal_data = None
            output[participant_id][task_id] = []
            if type(signal_data) is pd.DataFrame:
                selected_signal_data = signal_data.values[:, feature_index]
            elif type(signal_data) is list:
                selected_signal_data = [feat.values[:, feature_index] for feat in signal_data]
            output[participant_id][task_id] = selected_signal_data
    return output


def extract_gsr_features(microsiemens: List[float], sampling_rate) -> pd.DataFrame:
    eda_decomposed = nk.eda_phasic(microsiemens, sampling_rate=sampling_rate)
    eda_peaks, info = nk.eda_peaks(eda_decomposed['EDA_Phasic'], sampling_rate=sampling_rate)
    signals = pd.DataFrame({"EDA_Raw": microsiemens})
    signals = pd.concat([signals, eda_decomposed, eda_peaks], axis=1)
    return signals


def statistics_gsr_signal_features(gsr_features: pd.DataFrame) -> np.array:
    eda_decomposed_phasic = gsr_features['EDA_Phasic'].values
    eda_peaks = gsr_features['SCR_Peaks'].values
    first_order_grad = np.gradient(eda_decomposed_phasic)
    second_order_grad = np.gradient(first_order_grad)
    mean_scr, max_scr, min_scr, kurtosis_scr, skewness_scr = eda_decomposed_phasic.mean(), eda_decomposed_phasic.max(), \
        eda_decomposed_phasic.min(), kurtosis(eda_decomposed_phasic), skew(eda_decomposed_phasic)
    mean_peak, max_peak, min_peak, std_peak, = eda_peaks.mean(), eda_peaks.max(), eda_peaks.min(), eda_peaks.std()
    mean_first_grad, std_first_grad = first_order_grad.mean(), first_order_grad.std()
    mean_second_grad, std_second_grad = second_order_grad.mean(), second_order_grad.std()
    ALSC = math.sqrt(sum([(eda_decomposed_phasic[i] - eda_decomposed_phasic[i-1]) ** 2 + 1 for i in range(1, len(eda_decomposed_phasic))]))
    INSC = np.sum(np.abs(eda_decomposed_phasic))
    APSC = np.sum(eda_decomposed_phasic * eda_decomposed_phasic) / len(eda_decomposed_phasic)
    RMSC = math.sqrt(APSC)
    statistics_feat = np.array([mean_scr, max_scr, min_scr, kurtosis_scr, skewness_scr, mean_peak, max_peak, min_peak, std_peak, 
        mean_first_grad, std_first_grad, mean_second_grad, std_second_grad, mean_peak, max_peak, min_peak, std_peak, ALSC, INSC, APSC, RMSC])
    return statistics_feat