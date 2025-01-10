# UsefulML in python 3.8

> ML Useful package for Time Series Forecasting
> Author: [Jung-Pu-Chang](https://www.linkedin.com/in/jungpu-chang-024859264/)、[容噗玩Data](https://www.youtube.com/channel/UCmWCMqDKCR56pqd10qNkv3Q)    

## Directory

```bash
.
├── README.md
├── LICENSE
├── requirements.txt
└── PreProcessing.py
```

## Example

```bash
from ucimlrepo import fetch_ucirepo
from LGB import LightGBM

default_of_credit_card_clients = fetch_ucirepo(id=350)
train_X = default_of_credit_card_clients.data.features
train_Y = default_of_credit_card_clients.data.targets
train_X.columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
                   'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                   'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                   'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

params = {
          'boosting_type': 'dart', 
          'n_estimators' : 1000,
          'learning_rate': 0.05,
          'n_jobs' : -1, 
          'random_state' : 7,
          'verbose' : 0
          }

scoring = {
           'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
           }

param_grid = {
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [3, 5, 7],
              'num_leaves': [15, 31, 63],
              'n_estimators': [1000, 1500, 2000],
              'boosting_type' : ['gbdt','dart'],
              'random_state' : [7], 
              }   

train_X_fs, feature_name = LightGBM.permutation_selection(train_X, train_Y, 
                                                          params = params,
                                                          imp = 0.005)
model, cv, cv_idx = LightGBM.build_model(train_X_fs, train_Y, 
                                         params = params, scoring = scoring, 
                                         fold_time = 5)

model_grid_tune = LightGBM.grid_tune(train_X_fs, train_Y, 
                                     fold_time = 3, param_grid = param_grid)
model_random_tune = LightGBM.random_tune(train_X_fs, train_Y, 
                                         fold_time = 3, param_grid = param_grid)
```

## Module Description : PreProcessing.py  

### Def Contents
| def  | purpose |
|:------:|:-------:|
| sparse_fill_0 | Detect Sparse & Fill NA | 
| auto_diff | Detect diff lag & diff | 
| auto_transform | BoxCox & Yeojohnson Transform |
| pelt | Change Point Detect |
| decompose | Change Point Detect |
| ad_test | Change Point Detect |

