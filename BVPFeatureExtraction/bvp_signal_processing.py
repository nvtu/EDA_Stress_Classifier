import numpy as np
import neurokit2 as nk
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler


# ---------------------- HRV UTILS FROM NEUROKIT2 ----------------------------------

def _hrv_get_rri(peaks = None, sampling_rate = 1000, interpolate = False, **kwargs):
    rri = np.diff(peaks) / sampling_rate * 1000
    if interpolate is False:
        return rri
    else:
        # Minimum sampling rate for interpolation
        if sampling_rate < 10:
            sampling_rate = 10
        # Compute length of interpolated heart period signal at requested sampling rate.
        desired_length = int(np.rint(peaks[-1]))
        rri = nk.signal_interpolate(
            peaks[1:],  # Skip first peak since it has no corresponding element in heart_period
            rri,
            x_new=np.arange(desired_length),
            **kwargs
        )
        return rri, sampling_rate


def _hrv_sanitize_input(peaks=None):

    if isinstance(peaks, tuple):
        peaks = _hrv_sanitize_tuple(peaks)
    elif isinstance(peaks, (dict, pd.DataFrame)):
        peaks = _hrv_sanitize_dict_or_df(peaks)
    else:
        peaks = _hrv_sanitize_peaks(peaks)

    return peaks


# =============================================================================
# Internals
# =============================================================================
def _hrv_sanitize_tuple(peaks):

    # Get sampling rate
    info = [i for i in peaks if isinstance(i, dict)]
    sampling_rate = info[0]['sampling_rate']

    # Get peaks
    if isinstance(peaks[0], (dict, pd.DataFrame)):
        try:
            peaks = _hrv_sanitize_dict_or_df(peaks[0])
        except NameError:
            if isinstance(peaks[1], (dict, pd.DataFrame)):
                try:
                    peaks = _hrv_sanitize_dict_or_df(peaks[1])
                except NameError:
                    peaks = _hrv_sanitize_peaks(peaks[1])
            else:
                peaks = _hrv_sanitize_peaks(peaks[0])

    return peaks, sampling_rate


def _hrv_sanitize_dict_or_df(peaks):

    # Get columns
    if isinstance(peaks, dict):
        cols = np.array(list(peaks.keys()))
        if 'sampling_rate' in cols:
            sampling_rate = peaks['sampling_rate']
        else:
            sampling_rate = None
    elif isinstance(peaks, pd.DataFrame):
        cols = peaks.columns.values
        sampling_rate = None

    cols = cols[["Peak" in s for s in cols]]

    if len(cols) > 1:
        cols = cols[[("ECG" in s) or ("PPG" in s) for s in cols]]

    if len(cols) == 0:
        raise NameError(
            "NeuroKit error: hrv(): Wrong input, ",
            "we couldn't extract R-peak indices. ",
            "You need to provide a list of R-peak indices.",
        )

    peaks = _hrv_sanitize_peaks(peaks[cols[0]])

    if sampling_rate is not None:
        return peaks, sampling_rate
    else:
        return peaks


def _hrv_sanitize_peaks(peaks):

    if isinstance(peaks, pd.Series):
        peaks = peaks.values

    if len(np.unique(peaks)) == 2:
        if np.all(np.unique(peaks) == np.array([0, 1])):
            peaks = np.where(peaks == 1)[0]

    if isinstance(peaks, list):
        peaks = np.array(peaks)

    return peaks

# ---------------------------------------------------------------------------------------------


def extract_bvp_features(bvp_data, sampling_rate):
    # Extract Heart Rate, RR Interval, and Heart Rate Variability features from PPG signals
    # bvp_data = MinMaxScaler().fit_transform(np.array(bvp_data).reshape(-1, 1)).ravel()
    ppg_signals, info = nk.ppg_process(bvp_data, sampling_rate = sampling_rate)
    hr = ppg_signals['PPG_Rate']
    # hr = MinMaxScaler().fit_transform(np.array(hr).reshape(-1, 1)).ravel()
    peaks = info['PPG_Peaks']

    # Sanitize input
    peaks = _hrv_sanitize_input(peaks)
    if isinstance(peaks, tuple):  # Detect actual sampling rate
        peaks, sampling_rate = peaks[0], peaks[1]
    rri = _hrv_get_rri(peaks, sampling_rate = sampling_rate, interpolate = False)
    diff_rri = np.diff(rri)
    hrv_features = nk.hrv(peaks, sampling_rate = sampling_rate) # Ignore NeuroKitWarning: The duration of recording is too short to support a sufficiently long window for high frequency resolution as we used another frequency for hrv_frequency
    hrv_frequency = nk.hrv_frequency(peaks, sampling_rate = sampling_rate, ulf = (0.01, 0.04), lf = (0.04, 0.15), hf = (0.15, 0.4)) # the parameters of ULF, LF, HF follows the original paper of WESAD dataset
    
    # Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. 2018. Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.
    # In Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18). Association for Computing Machinery, New York, NY, USA, 400–408. DOI:https://doi.org/10.1145/3242969.3242985
    # Not including: f_x_HRV of ULF and HLF, rel_f_x, sum f_x_HRV
    mean_HR, std_HR = np.mean(hr), np.std(hr)
    mean_HRV, std_HRV = hrv_features['HRV_MeanNN'], hrv_features['HRV_SDNN']
    HRV_ULF, HRV_LF, HRV_HF, HRV_LFHF, HRV_LFnorm, HRV_HFnorm = hrv_frequency['HRV_ULF'], hrv_frequency['HRV_LF'], hrv_frequency['HRV_HF'], hrv_frequency['HRV_LFHF'], hrv_frequency['HRV_LFn'], hrv_frequency['HRV_HFn']
    rms = np.sqrt(np.nanmean(rri ** 2))
    nn50 = np.sum(np.abs(diff_rri) > 50)
    HRV_TINN, HRV_pNN50, HRV_RMSSD = hrv_features['HRV_TINN'], hrv_features['HRV_pNN50'], hrv_features['HRV_RMSSD']

    # Nkurikiyeyezu, K., Yokokubo, A., & Lopez, G. (2019). The Influence of Person-Specific Biometrics in Improving Generic Stress Predictive Models. 
    # ArXiv, abs/1910.01770.
    kurtosis_HRV, skewness_HRV = kurtosis(rri), skew(rri)
    HRV_VLF = hrv_frequency['HRV_VLF']
    HRV_SD1, HRV_SD2 = hrv_features['HRV_SD1'], hrv_features['HRV_SD2']
    HRV_SDSD = hrv_features['HRV_SDSD']
    HRV_SDSD_RMSSD = HRV_SDSD / HRV_RMSSD
    adj_sum_rri = diff_rri + 2 * rri[:-1]
    HRV_pNN25 = np.sum(np.abs(diff_rri) > 25) / len(rri) * 100
    relative_RRI = 2 * diff_rri / adj_sum_rri
    mean_relativeRRI, median_relativeRRI, std_relativeRRI, RMSSD_relativeRRI, kurtosis_relativeRRI, skew_relativeRRI = np.mean(relative_RRI), np.median(relative_RRI), np.std(relative_RRI), np.sqrt(np.mean(np.diff(relative_RRI) ** 2)), kurtosis(relative_RRI), skew(relative_RRI)

    # Combining the extracted features
    features = [mean_HR, std_HR, mean_HRV, std_HRV, kurtosis_HRV, skewness_HRV, rms, nn50, HRV_pNN50, HRV_pNN25, HRV_TINN, HRV_RMSSD, HRV_LF, HRV_HF, HRV_LFHF, HRV_LFnorm, HRV_HFnorm, HRV_SD1, HRV_SD2, HRV_SDSD, HRV_SDSD_RMSSD, mean_relativeRRI, median_relativeRRI, std_relativeRRI, RMSSD_relativeRRI, kurtosis_relativeRRI, skew_relativeRRI]
    features = np.array(list(map(float, features)))
    return features