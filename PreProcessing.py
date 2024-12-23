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

class PreProcessing : 
    def __init__(self, df, date_col, target_col): 
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
        self.df = df
        self.date_col = date_col # 日期
        self.target_col = target_col # 應變量
        self.df = df.groupby([self.date_col])[self.target_col].sum().reset_index()
        
        try :
            self.df[self.date_col] = self.df[self.date_col].astype(int)
        except ValueError: 
            print("Date Format Transform Failed ! ")
        
        try :
            self.df[self.target_col] = self.df[self.target_col].astype(int)
        except ValueError: 
            print("Target Format Transform Failed ! ")
   
    def sparse_fill_0(self) : # 資料清洗 : sparse 補 0
        first = datetime.date(int(self.df[self.date_col].min().astype(str)[:4]), 
                              int(self.df[self.date_col].min().astype(str)[4:6]), 
                              int(self.df[self.date_col].min().astype(str)[5:8]))
        last = datetime.date(int(self.df[self.date_col].max().astype(str)[:4]), 
                             int(self.df[self.date_col].max().astype(str)[4:6]), 
                             int(self.df[self.date_col].max().astype(str)[5:8]))
        number_of_days = abs(first - last).days+1
        date_list = [(first + datetime.timedelta(days = day)).isoformat() for day in range(number_of_days)]
        date = pd.DataFrame(date_list)
        date.columns = [self.date_col]
        date[self.date_col] = date[self.date_col].replace(to_replace ='-', value = '', regex = True).astype(int)
        out = pd.merge(date, self.df[[self.date_col, self.target_col]], on=self.date_col,how='left')
        out[self.target_col] = out[self.target_col].fillna(0) 
        return out

    def decompose(self, out, model) : # 拆解 additive or multiplicative
        res = seasonal_decompose(out[self.target_col], period=12, model=model) 
        resplot = res.plot()
        resplot.set_size_inches(15,8)
        return res

    def ad_test(self, out) : # 穩定性檢查，< alpha 穩定 
        result = adfuller(out[self.target_col].dropna())
        print('ADF Statistic: %f' % result[0])
        print('p-value (<0.05 = 穩定) : %f' % result[1])  
    
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
            print('Target all positive! \nTransform by boxcox !') # 理論基礎較強，優先使用
            fitted_data, fitted_lambda = stats.boxcox(out[self.target_col].to_numpy()) 
            fitted_data = pd.DataFrame(fitted_data)
            fitted_data = pd.concat([out[[self.date_col]], fitted_data],axis=1)
        except :
            print('Target has zero or negative! \nTransform by yeojohnson !') # boxcox微調
            fitted_data, fitted_lambda = stats.yeojohnson(out[self.target_col].to_numpy()) 
            fitted_data = pd.DataFrame(fitted_data)
            fitted_data = pd.concat([out[[self.date_col]], fitted_data],axis=1)
        return fitted_data, fitted_lambda

    def pelt(self, out, n_bkps, sigma) : # 斷點偵測算法
        use = out[[self.target_col]]
        signal, bkps = rpt.pw_constant(use.shape[0], use.shape[1], n_bkps, noise_std=sigma)
        algo = rpt.Pelt(model="rbf").fit(signal) # Pelt
        result = algo.predict(pen=10)
        rpt.display(signal, bkps, result)
        plt.show()
        return result
    
    def preprocessing(self, model, n_bkps, sigma) : 
        out = self.sparse_fill_0()
        diff_data, lag = self.auto_diff(out)
        diff_data = diff_data.rename(columns={self.target_col: 'diff_trans'})
        bc_data, bc_lambda = self.auto_transform(out)
        bc_data = bc_data.rename(columns={self.target_col: 'boxcox_trans'})
        result = self.pelt(out, n_bkps, sigma)
        self.decompose(out, model) # multiplicative
        self.ad_test(out)
        out = pd.merge(out, diff_data, on=self.date_col, how='left')
        out = pd.merge(out, bc_data, on=self.date_col, how='left')
        out['diff_lag'] = lag
        out['lamda'] = bc_lambda
        return out
        
class ExternalData : 
    def add_calendar(cal):
        cal.loc[(cal['是否放假']==2), "holiday"] = '1'
        cal.loc[(cal['備註'].isnull()==False), "holiday"] = '2'
        cal['holiday'] = cal['holiday'].fillna(0).astype(int)
        labelencoder = LabelEncoder()
        cal['星期'] = labelencoder.fit_transform(cal['星期'])
        cal = cal[['西元日期','星期','holiday']]
        print('holiday : 0 = 上班、1 = 周休二日、2 = 節慶')
        return cal
#%%
if __name__=='__main__': 
    path = 'C:/Users/user/Desktop/資料分析/3. 推論分析/時間序列/data/'
    df = pd.read_csv(path + "三商美福.csv", encoding='Big5')
    df = df.loc[(df['料號']=="BLDP30")]
    
    test = PreProcessing(df,'需求日','需求量')
    df = test.preprocessing('additive',4,4)
    #out = test.sparse_fill_0()
    #test.decompose(out, 'additive') # multiplicative
    #test.ad_test(out)
    #diff_data, lag = test.auto_diff(out)
    #bc_data, bc_lambda = test.auto_transform(out)
    #result = test.pelt(out,4,4)
    'yeojohnson inverse'
    forecast = np.power((fitted_lambda * forecast[['yhat']]) + 1, 1 / fitted_lambda) - 1

    df1 = pd.read_csv(path + '外部資料/110中華民國政府行政機關辦公日曆表.csv', encoding='big5',header=0) 
    df2 = pd.read_csv(path + '外部資料/111年中華民國政府行政機關辦公日曆表.csv', encoding='big5',header=0) 
    df3 = pd.read_csv(path + '外部資料/112年中華民國政府行政機關辦公日曆表.csv', encoding='big5',header=0) 
    cal = pd.concat([df1,df2,df3],axis=0) 
    cal = ExternalData.add_calendar(cal)
