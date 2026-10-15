# Smart Home Monitoring System using the GSHAM Dataset

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Tableau](https://img.shields.io/badge/Tableau-E97627?logo=tableau&logoColor=white)

Predicting indoor temperature from smart home IoT sensor readings, comparing classic regression models with deep learning (LSTM and 1D CNN). Insights were also visualized in Tableau.

## Approach

1. **EDA and cleaning** (`EDA and Data Cleaning.ipynb`): shape, missing values, data types, summary statistics and a correlation heatmap, then cleaning into `cleaned_iot_dataset.csv`
2. **Preprocessing:** Min-Max scaling so every sensor contributes on the same scale, and a train/test split
3. **Models** (`IoT.ipynb`, `LSTM.ipynb`, `CNN.ipynb`): Decision Tree and Random Forest regression, an LSTM, a stacked LSTM for multi-sensor time series forecasting, and a 1D CNN
4. **Evaluation:** MAE, MSE, RMSE and R², plus plots of predicted vs actual values

## Results (target: temperature, scaled 0–1)

| Model | MAE | MSE |
| --- | --- | --- |
| Decision Tree Regressor | 0.00055 | 0.000044 |
| **Random Forest Regressor** (50 trees) | **0.00052** | **0.000023** |
| LSTM (32 units + dense layers) | 0.0725 | 0.0096 |

The tree-based models fit the scaled sensor data best. The LSTM and CNN notebooks explore sequence models for forecasting all 12 sensor values over time.

## Data

The notebooks read `iot_dataset.csv` (raw) and `cleaned_iot_dataset.csv`. They are not in this repo because they are about 100 MB, over GitHub's file size limit. Put the GSHAM smart home dataset CSV in this folder as `iot_dataset.csv`, then run `EDA and Data Cleaning.ipynb` to create the cleaned file.

## Run it

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn jupyter
jupyter notebook
```
