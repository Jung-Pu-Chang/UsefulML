import sys
sys.path.append(r'D:\Users\Pu_chang\Desktop\資料分析\UsefulML')
from PreProcessing import TS_PreProcess
import datetime
import itertools # grid_search
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet 
from prophet.diagnostics import performance_metrics, cross_validation
import warnings
warnings.filterwarnings('ignore')

'每個產品於每個商店銷量，時間 = d_1 ~ d_1913'
path = 'D:/Users/Pu_chang/Desktop/資料分析/3. 推論分析/時間序列/data/M5/'
validation = pd.read_csv(path + 'sales_train_validation.csv', encoding='utf-8',header=0)
calender = pd.read_csv(path + 'calendar.csv', encoding='utf-8',header=0)
evaluation = pd.read_csv(path + 'sales_train_evaluation.csv', encoding='utf-8',header=0)
greater_than_zero = (validation.iloc[:,6:] > 0).sum(axis=1).reset_index()

'易 : 全 > 0的跑，5859，2011-01-29 ~ 2016-04-24'
def split_train_test(item, df1, df2) : 
    train = df1.loc[item].reset_index().iloc[6:,:]
    train.columns = ['d', 'y']
    train['y'] = train['y'].astype(int)
    train = pd.merge(train, calender, on="d",how='inner')
    train = train.sort_values(by = 'date', ascending=True) 
    
    test = df2.loc[item].reset_index().iloc[6:,:]
    test.columns = ['d', 'y']
    test['y'] = test['y'].astype(int)
    test = pd.merge(test, calender, on="d",how='inner')
    test['d'] = test['d'].replace(to_replace ='d_', value = '', regex = True).astype(int)
    test = test.loc[test['d']>=1914]
    test = test.sort_values(by = 'date', ascending=True) 
    return train, test

train, test = split_train_test(5859, validation, evaluation)

#%%
'前處理'
train['date'] = train['date'].replace(to_replace ='-', value = '', regex = True).astype(int)

preprocess = TS_PreProcess(train, 'date','y')
df = preprocess.preprocessing('additive',4,4) 

'sparse_fill_0 > auto_diff > auto_transform > pelt > decompose > ad_test'

out = preprocess.sparse_fill_0()
diff_data, lag = preprocess.auto_diff(out) # 差分偵測
bc_data, bc_lambda = preprocess.auto_transform(out) # 常態與穩定轉換
result = preprocess.pelt(out,4,4) # 斷點
preprocess.decompose(out, 'additive') # multiplicative 不常用，有0就掛
preprocess.ad_test(out) # 穩定偵測

'''
圖 1 = 整體趨勢可看斷點
圖 2 = 長期趨勢 : 線性 & 指數
圖 3 = 季節性，若隨時間增加用乘 multi
圖 4 = 殘差，一個範圍內穩定跳動
'''
#%%
'Loss = SMAPE [0~200%] : 避免分母 0'
def smape(A, F):
    denominator = np.abs(A) + np.abs(F)
    denominator[denominator == 0] = 1e-10  # 避免分母為零
    smape_value = 100 / len(A) * np.sum(2 * np.abs(F - A) / denominator)
    return f"{smape_value:.1f}%"

#%%
'prophet : y vs trans forecast'
df['ds'] = df['date'].astype(str).str[:4] + ['-'] + df['date'].astype(str).str[4:6] + ['-'] + df['date'].astype(str).str[6:8]
df['ds'] = pd.to_datetime(df['ds'])
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=28, include_history = False)
forecast = m.predict(future)
smape(test['y'].reset_index(drop=True), forecast['yhat'].reset_index(drop=True))
# smape = 16.6%

df['y'] = df['yeojohnson_trans']
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=28, include_history = False)
forecast = m.predict(future)
smape(test['y'].reset_index(drop=True), forecast['yhat'].reset_index(drop=True))
# smape = 66.3%

'yeojohnson inverse'
forecast = np.power((df['lamda'].mean() * forecast[['yhat']]) + 1, 1 / df['lamda'].mean()) - 1
smape(test['y'].reset_index(drop=True), forecast['yhat'].reset_index(drop=True))
# smape = 16.8%

#%%
'prophet tune : 自動斷點 vs 手動斷點'
df = preprocess.preprocessing('additive',4,4) 
df['ds'] = df['date'].astype(str).str[:4] + ['-'] + df['date'].astype(str).str[4:6] + ['-'] + df['date'].astype(str).str[6:8]
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['yeojohnson_trans']
m = Prophet(growth = 'linear', # logistic，長期趨勢 = 線性
            changepoint_prior_scale = 0.05, 
            changepoints = ['2015-04-08'],
            weekly_seasonality = False, # 自定義季節性
            #holidays = holidays # 自定義事件 
            ) 
m.add_seasonality(name='weekly', period=7, # P = 365.25(年) 30.5(月) 7(週) 
                  fourier_order=3, # N(時間段數量) = 10(年) 5(月) 3(週)
                  prior_scale=0.1)
m.add_country_holidays(country_name='US') # 台灣 = TW
m.fit(df)
future = m.make_future_dataframe(periods=28, include_history = False)
forecast = m.predict(future)
forecast = np.power((df['lamda'].mean() * forecast[['yhat']]) + 1, 1 / df['lamda'].mean()) - 1
smape(test['y'].reset_index(drop=True), forecast['yhat'].reset_index(drop=True))
# smape = 16.7% 自動、14.9% 手動

#%%
'multi-prophet : 多變量 vs 單變量'
df = preprocess.preprocessing('additive',4,4) 
df['ds'] = df['date'].astype(str).str[:4] + ['-'] + df['date'].astype(str).str[4:6] + ['-'] + df['date'].astype(str).str[6:8]
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = df['yeojohnson_trans']
df = pd.merge(df, train[["date",'snap_CA', 'snap_TX', 'snap_WI']], on="date",how='inner')

m = Prophet(growth = 'linear', # logistic，長期趨勢 = 線性
            changepoint_prior_scale = 0.05, 
            changepoints = ['2015-04-08'],
            weekly_seasonality = False, # 自定義季節性
            #holidays = holidays # 自定義事件 
            ) 
m.add_seasonality(name='weekly', period=7, # P = 365.25(年) 30.5(月) 7(週) 
                  fourier_order=3, # N(時間段數量) = 10(年) 5(月) 3(週)
                  prior_scale=0.1)
m.add_country_holidays(country_name='US') # 台灣 = TW
m.add_regressor('snap_CA') # 加州 是否有特殊活動或假日
m.add_regressor('snap_TX') # 德州 是否有特殊活動或假日
m.add_regressor('snap_WI') # 威斯康星州 是否有特殊活動或假日
m.fit(df)
future = m.make_future_dataframe(periods=28, include_history = False)
future['date'] = future['ds'].astype(str).str[:10]
future = pd.merge(future, test[["date",'snap_CA', 'snap_TX', 'snap_WI']], on="date",how='inner')
forecast = m.predict(future)
forecast = np.power((df['lamda'].mean() * forecast[['yhat']]) + 1, 1 / df['lamda'].mean()) - 1
smape(test['y'].reset_index(drop=True), forecast['yhat'].reset_index(drop=True))
# smape = 16.7% 自動、14.9% 手動

#%%
'multi-prophet : 多變量 + 外部特徵(氣候等)'



#%%
'難 : 幾乎都是 0的，prophet + LGB + 相似度補'
train = raw.loc[6682].reset_index().iloc[6:,:]
train.columns = ['d', 'y']
train['y'] = train['y'].astype(int)
train = pd.merge(train, calender, on="d",how='inner')
train = train.sort_values(by = 'date', ascending=True) 





