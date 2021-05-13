# Stress Detection Using Electrodermal Activity Signal (EDA) and Machine Learning Models

This code is made to determine the approach to build stress detection using Electrodermal Activity Signal (EDA) with machine learning models. It proves the hypotheses that using low sampling-rate signal as data to build stress detector has no statistically difference from the one trained on high sampling-rate data. 
With low sampling-rate data, it is proved that using person-specific stress detector (user-dependent model) is better than general cross-population stress detector (user-independent model).

The experiment was conduct using **WESAD**, **AffectiveROAD**, and **DCU-NVT-EXP1** datasets. You can download it [here](https://drive.google.com/file/d/15BpNyTRY0OsFJ06FaVZcDtRgV-Y4a81P/view?usp=sharing).

**Note**: Change the absolute paths to the dataset in file ```config.ini``` before running the code.

## 1. Format datasets to a pre-defined structure
Run ```DatasetPreprocess/*.ipynb``` to re-format the datasets to a pre-defined structure The structure used for this code is as follows:
```
{
    "eda": {
        "user_id": {
            "task_id": List[float] (dimension: 1 x d)
        }
    }
    "ground_truth": {
        "user_id": {
            "task_id": List[int] (dimension: 1 x d)
        }
    }
}
```
## 2. Process raw EDA signal and extract statistical features
Run ```EDAFeatureExtraction/eda_feature_extraction.ipynb```. 
Run ```EDAFeatureExtraction/eda_feature_extraction.py``` if you want to run code by command-line.
```
python3 EDAFeatureExtraction/eda_feature_extraction.py
```
## 3. Build Stress Detectors and Analyse Results to Prove the Hypotheses
Perform Grid Search Cross-Validation on Logistic Regression, Random Forests, Support Vector Machine, Multi-layer Perceptron, and K-Nearest-Neighbors with target evaluation metric: **balanced accuracy**. 
- Run ```analyse_dataset.ipynb``` to have an overview of the dataset. 
- Run ```run_main_exp1.ipynb``` to train stress detectors and output evaluation scores of detectors on test set.
- Run ```analyse_exp1_results.ipynb``` to analyse the results and prove hypotheses.
