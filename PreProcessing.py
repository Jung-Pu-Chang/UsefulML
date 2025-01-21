import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima # diff
import ruptures as rpt # from R
from scipy import stats # boxcox & yeojohnson
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import LabelEncoder

class TS_PreProcess : 
    def __init__(self, df, date_col, target_col): 
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
        self.df = df
        self.date_col = date_col # 日期
        self.target_col = target_col # 應變量
        self.df = df.groupby([self.date_col])[self.target_col].sum().reset_index()
        
        try :
            self.df[self.date_col] = self.df[self.date_col].astype(int)
        except ValueError: 
            print("Date Format Transform Failed ! Support type : 20250118")
        
        try :
            self.df[self.target_col] = self.df[self.target_col].astype(int)
        except ValueError: 
            print("Target Format Transform Failed ! Support type : 100")
   
    def sparse_fill_0(self) : # 資料清洗 : sparse 補 0
        date_min = str(self.df[self.date_col].min()).zfill(8)
        date_max = str(self.df[self.date_col].max()).zfill(8)
        first = datetime.date(int(date_min[:4]), int(date_min[4:6]), int(date_min[6:8]))
        last = datetime.date(int(date_max[:4]), int(date_max[4:6]), int(date_max[6:8]))
        number_of_days = abs(first - last).days+1
        date_list = [(first + datetime.timedelta(days = day)).isoformat() for day in range(number_of_days)]
        date = pd.DataFrame(date_list)
        date.columns = [self.date_col]
        date[self.date_col] = date[self.date_col].replace(to_replace ='-', value = '', regex = True).astype(int)
        out = pd.merge(date, self.df[[self.date_col, self.target_col]], on=self.date_col,how='left')
        out[self.target_col] = out[self.target_col].fillna(0) 
        return out
    
    def auto_diff(self, out) : # 特徵轉換 : 差分 
        #差分階數，test = adf kpss pp    
        lag = pmdarima.arima.ndiffs(out[self.target_col].dropna(), test='adf') 
        if lag > 0:
            out[[self.target_col]] = out[[self.target_col]].diff(lag).dropna() # 1階差分
            print(f'Diff : {lag} ')
        else :
            lag = 0 # 無差分
            print('Do not need to diff !')
        return out, lag
    
    def auto_transform(self, out) : # 轉常態 & 去除離群
        try : 
            fitted_data, fitted_lambda = stats.boxcox(out[self.target_col].to_numpy()) 
            fitted_data = pd.DataFrame(fitted_data)
            fitted_data = pd.concat([out[[self.date_col]], fitted_data],axis=1)
            fitted_data.columns = [self.date_col, 'boxcox_trans']
            print('Target all positive! \nTransform by boxcox !') # 理論基礎較強，優先使用
        except :
            fitted_data, fitted_lambda = stats.yeojohnson(out[self.target_col].to_numpy()) 
            fitted_data = pd.DataFrame(fitted_data)
            fitted_data = pd.concat([out[[self.date_col]], fitted_data],axis=1)
            fitted_data.columns = [self.date_col, 'yeojohnson_trans']
            print('Target has zero or negative! \nTransform by yeojohnson !') # boxcox微調
        return fitted_data, fitted_lambda

    def pelt(self, out, n_bkps, sigma) : # 以 ML 基礎計算
        use = out[[self.target_col]]
        signal, bkps = rpt.pw_constant(use.shape[0], use.shape[1], n_bkps, noise_std=sigma)
        algo = rpt.Pelt(model="rbf").fit(signal) # Pelt
        result = algo.predict(pen=10)
        rpt.display(signal, bkps, result)
        plt.show()
        return result
    
    def decompose(self, out, model) : # 拆解 additive or multiplicative
        res = seasonal_decompose(out[self.target_col], period=12, model=model) 
        resplot = res.plot()
        resplot.set_size_inches(15,8)
        return res

    def ad_test(self, out) : # 穩定性檢查，< alpha 穩定 
        result = adfuller(out[self.target_col].dropna())
        print('ADF Statistic: %f' % result[0])
        print('p-value (<0.05 = 穩定) : %f' % result[1])  
    
    def preprocessing(self, model, n_bkps, sigma) : 
        
        '''
        1. sparse_fill_0 
        2. auto_diff 
        3. auto_transform 
        4. pelt 
        5. decompose 
        6. ad_test '''
    
        out = self.sparse_fill_0()
        diff_data, lag = self.auto_diff(out)
        diff_data = diff_data.rename(columns={self.target_col: 'diff_trans'})
        bc_data, bc_lambda = self.auto_transform(out)
        result = self.pelt(out, n_bkps, sigma)
        self.decompose(out, model) # multiplicative
        self.ad_test(out)
        out = pd.merge(out, diff_data, on=self.date_col, how='left')
        out = pd.merge(out, bc_data, on=self.date_col, how='left')
        out['diff_lag'] = lag
        out['lamda'] = bc_lambda
        out['is_changepoint'] = out.index.isin(result).astype(int)
        return out

#%%
if __name__=='__main__': 
    path = 'D:/Users/Pu_chang/Desktop/資料分析/3. 推論分析/時間序列/data/M5/'
    validation = pd.read_csv(path + 'sales_train_validation.csv', encoding='utf-8',header=0)
    calender = pd.read_csv(path + 'calendar.csv', encoding='utf-8',header=0)
    train = validation.loc[5889].reset_index().iloc[6:,:]
    train.columns = ['d', 'y']
    train['y'] = train['y'].astype(int)
    train = pd.merge(train, calender, on="d",how='inner')
    train = train.sort_values(by = 'date', ascending=True)
    
    test = TS_PreProcess(df, 'date','y')
    df = test.preprocessing('additive',4,4)
    #out = test.sparse_fill_0()
    #test.decompose(out, 'additive') # multiplicative
    #test.ad_test(out)
    #diff_data, lag = test.auto_diff(out)
    #bc_data, bc_lambda = test.auto_transform(out)
    #result = test.pelt(out,4,4)
    'yeojohnson inverse'
    forecast = np.power((fitted_lambda * forecast[['yhat']]) + 1, 1 / fitted_lambda) - 1


