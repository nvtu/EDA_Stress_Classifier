import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict
import neurokit2 as nk
from scipy.stats import kurtosis, skew, linregress, pearsonr
import math


def resampling_data_signal(data: Dict[str, Dict[str, np.array]], sampling_rate: int, desired_sampling_rate: int, method: str = 'interpolation') -> Dict[str, Dict[str, object]]:
    output = defaultdict(dict)
    for participant_id, dataset in data.items():
        for task_id, signal_data in dataset.items():
            resampled_signal = nk.signal_resample(signal_data, method = method, sampling_rate = sampling_rate, desired_sampling_rate = desired_sampling_rate)
            output[participant_id][task_id] = resampled_signal
    return output


def extract_eda_features(eda: List[float], sampling_rate) -> pd.DataFrame:
    HIGHCUT_FREQUENCY = 5 # defaults as BioSPPy
    nyquist_freq = 2 * HIGHCUT_FREQUENCY / sampling_rate # Normalize frequency to Nyquist Frequency (Fs/2)
    if 0 < nyquist_freq < 1:
        eda = nk.eda_clean(eda, sampling_rate=sampling_rate, method='biosppy')
    eda_decomposed = nk.eda_phasic(eda, sampling_rate=sampling_rate)
    scr_peaks, info = nk.eda_peaks(eda_decomposed['EDA_Phasic'], sampling_rate=sampling_rate)
    signals = pd.DataFrame({"EDA_Raw": eda})
    signals = pd.concat([signals, eda_decomposed, scr_peaks], axis=1)
    return signals


def extract_statistics_eda_features(eda: pd.DataFrame) -> np.array:
    eda_raw = eda['EDA_Raw'].values
    eda_decomposed_phasic = eda['EDA_Phasic'].values
    eda_decomposed_tonic = eda['EDA_Tonic'].values
    scr_peaks = eda['SCR_Peaks'].values
    scr_onsets = eda['SCR_Onsets'].values
    scr_amplitude = eda['SCR_Amplitude'].values
    _time_axis = np.array([_time for _time in range(len(eda_raw))])

    # Choi J, Ahmed B, Gutierrez-Osuna R. Development and evaluation of an ambulatory stress monitor based on wearable sensors. 
    # IEEE Trans Inf Technol Biomed. 2012 Mar;16(2):279-86. doi: 10.1109/TITB.2011.2169804. Epub 2011 Sep 29. PMID: 21965215.
    mean_scl, std_scl = eda_decomposed_tonic.mean(), eda_decomposed_tonic.std()
    std_scr = eda_decomposed_phasic.std()
    corr, _ = pearsonr(_time_axis, eda_decomposed_tonic)

    # J. A. Healey and R. W. Picard, "Detecting stress during real-world driving tasks using physiological sensors," 
    # in IEEE Transactions on Intelligent Transportation Systems, vol. 6, no. 2, pp. 156-166, June 2005, doi: 10.1109/TITS.2005.848368.
    onset_time_points = scr_onsets * _time_axis
    onset_magnitude = abs(scr_amplitude[onset_time_points > 0])
    onset_time_points = onset_time_points[onset_time_points > 0]
    peak_time_points = scr_peaks * _time_axis
    peak_magnitude = abs(scr_amplitude[peak_time_points > 0])
    peak_time_points = peak_time_points[peak_time_points > 0]
    # Balance the size of onset and peak points as we only consider a complete startle response
    if len(peak_time_points) > 0 and len(onset_time_points) > 0:
        if peak_time_points[0] < onset_time_points[0]:
            peak_time_points = peak_time_points[1:]
            peak_magnitude = peak_magnitude[1:]
        if len(peak_time_points) > len(onset_time_points):
            peak_time_points = peak_time_points[:-1]
            peak_magnitude = peak_magnitude[:-1]
        elif len(peak_time_points) < len(onset_time_points):
            onset_time_points = onset_time_points[:-1]
            onset_magnitude = onset_magnitude[:-1]
    
    scr_response_duration = peak_time_points - onset_time_points
    startle_magnitude = peak_magnitude - onset_magnitude

    num_responses = len(scr_response_duration)
    sum_scr_response_duration = scr_response_duration.sum()
    sum_scr_amplitude = startle_magnitude.sum()
    area_of_response_curve = (scr_response_duration * startle_magnitude).sum() / 2.0
    
    # Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. 2018. Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.
    # In Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18). Association for Computing Machinery, New York, NY, USA, 400–408. DOI:https://doi.org/10.1145/3242969.3242985
    # slope_eda, *r = linregress(_time_axis, eda_raw)
    num_scr_peaks = scr_peaks.sum()
    mean_eda, std_eda, min_eda, max_eda = eda_raw.mean(), eda_raw.std(), eda_raw.min(), eda_raw.max()
    eda_dynamic_range = max_eda - min_eda

    # Nkurikiyeyezu, K., Yokokubo, A., & Lopez, G. (2019). The Influence of Person-Specific Biometrics in Improving Generic Stress Predictive Models. 
    # ArXiv, abs/1910.01770.
    first_order_grad = np.gradient(eda_decomposed_phasic)
    second_order_grad = np.gradient(first_order_grad)
    scr_at_peaks = np.array([eda_decomposed_phasic[index] for index in range(len(eda_raw)) if scr_peaks[index] > 0])
    scr_at_onsets = np.array([eda_decomposed_phasic[index] for index in range(len(eda_raw)) if scr_onsets[index] > 0])
    # If the data has no peaks or onsets
    if len(scr_at_peaks) == 0:
        scr_at_peaks = np.array([0])
    if len(scr_at_onsets) == 0:
        scr_at_onsets = np.array([0])

    mean_scr, max_scr, min_scr = eda_decomposed_phasic.mean(), eda_decomposed_phasic.max(), eda_decomposed_phasic.min()
    kurtosis_scr, skewness_scr = kurtosis(eda_decomposed_phasic), skew(eda_decomposed_phasic)
    mean_first_grad, std_first_grad = first_order_grad.mean(), first_order_grad.std()
    mean_second_grad, std_second_grad = second_order_grad.mean(), second_order_grad.std()
    mean_peaks, max_peaks, min_peaks, std_peaks = scr_at_peaks.mean(), scr_at_peaks.max(), scr_at_peaks.min(), scr_at_peaks.std()
    mean_onsets, max_onsets, min_onsets, std_onsets = scr_at_onsets.mean(), scr_at_onsets.max(), scr_at_onsets.min(), scr_at_onsets.std()
    ALSC = math.sqrt(sum([(eda_decomposed_phasic[i] - eda_decomposed_phasic[i-1]) ** 2 + 1 for i in range(1, len(eda_decomposed_phasic))]))
    INSC = np.sum(np.abs(eda_decomposed_phasic))
    APSC = np.sum(eda_decomposed_phasic * eda_decomposed_phasic) / len(eda_decomposed_phasic)
    RMSC = math.sqrt(APSC)
    statistics_feat = np.array([
                mean_scl, std_scl, std_scr, corr, 
                num_responses, sum_scr_response_duration, sum_scr_amplitude, area_of_response_curve,
                num_scr_peaks, mean_eda, std_eda, min_eda, max_eda, eda_dynamic_range,
                mean_scr, max_scr, min_scr, kurtosis_scr, skewness_scr, mean_first_grad, std_first_grad, mean_second_grad, std_second_grad, 
                mean_peaks, max_peaks, min_peaks, std_peaks, mean_onsets, max_onsets, min_onsets, std_onsets,
                ALSC, INSC, APSC, RMSC])
    return statistics_feat