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
import pandas as pd
from PreProcessing import TS_PreProcess
from prophet import Prophet

df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=365),
    'y': [i + (i * 0.1) for i in range(365)]  # 模擬數據
})

df['date'] = df['ds'].astype(str).str[:10].replace(to_replace ='-', value = '', regex = True).astype(int)
test = TS_PreProcess(df, 'date','y')
df = test.preprocessing('additive',4,4)
```

## Module Description : PreProcessing.py  

### Def Contents
| def  | purpose |
|:------:|:-------:|
| sparse_fill_0 | Detect Sparse & Fill NA | 
| auto_diff | Detect diff lag & diff | 
| auto_transform | BoxCox & Yeojohnson Transform |
| pelt | Change Point Detect |
| decompose | Visualize |
| ad_test | ADF-Test for Stationary |
| preprocessing | Complete all the above functions |


