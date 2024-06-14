from eli5.sklearn import PermutationImportance 
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold, cross_validate, cross_val_score, GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

class LightGBM:    
    
    def permutation_selection(train_X, train_Y, params, imp, isClassifier=True):
        """
        permutation 特徵篩選 
    
        Args:
        train_X (DataFrame): 訓練數據集的特徵值。
        train_Y (Series or array-like): 訓練數據集的目標值。
        params (dict): 模型參數。
        imp (float): 特徵重要度閥值。
        isClassifier (bool, optional): 是否使用分類器，默認為 True。
    
        Returns:
        DataFrame or None: 選擇後的特徵值，如果出現異常則返回 None。
        array-like or None: 選擇後的特徵名稱，如果出現異常則返回 None。
        """
        try:
            if isClassifier:
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)
            
            model.fit(train_X, train_Y)
            perm = PermutationImportance(model, random_state=1).fit(train_X, train_Y)
            sel = SelectFromModel(perm, prefit=True, threshold=imp)
            X_trans = sel.transform(train_X)
            feature_name = train_X.columns[sel.get_support()]
            train_X = pd.DataFrame(X_trans)
            train_X.columns = feature_name
            return train_X, feature_name
        except Exception as e:
            print('permutation_selection has error : ' + e)
            return None, None
    
    def build_model(train_X, train_Y, params, scoring, fold_time, isClassifier=True) :
        """
        建模 & k-fold，返回模型和交叉驗證結果。
     
        Args:
        train_X (DataFrame): 訓練數據集的特徵值。
        train_Y (Series or array-like): 訓練數據集的目標值。
        params (dict): 模型參數。
        scoring (dict): 評估指標。
        fold_time (int): 交叉驗證次數。
        isClassifier (bool, optional): 是否使用分類器，默認為 True。
     
        Returns:
        object: 訓練好的模型。
        DataFrame: 交叉驗證結果。
        list: 每次驗證資料的 index。
        """
        try:
            if isClassifier :
                model = lgb.LGBMClassifier(**params)
            else : 
                model = lgb.LGBMRegressor(**params)
            
            kf = KFold(fold_time, random_state = 7, shuffle = True)
            cv = pd.DataFrame(cross_validate(model, train_X, train_Y, 
                                             cv = kf, scoring = scoring,))
            cv_idx = [test_index for train_index, test_index in kf.split(train_X)]

            model.fit(train_X, train_Y)
            return model, cv, cv_idx 
        except Exception as e:
            print('build_model has error : ' + e)
            return None, None  
        
    def grid_tune(train_X, train_Y, fold_time, isClassifier=True, param_grid=None):
        """
        使用 GridSearchCV 調參，即每種參數組合都嘗試，返回最佳模型。
    
        Args:
        train_X (DataFrame): 訓練數據集的特徵值。
        train_Y (Series or array-like): 訓練數據集的目標值。
        fold_time (int): 交叉驗證次數。
        isClassifier (bool, optional): 是否使用分類器，默認為 True。
        param_grid (dict or None, optional): 網格搜索的參數網格，默認為 None。
    
        Returns:
        object: 最佳模型。
        """
        try:
            if isClassifier:
                model = lgb.LGBMClassifier()
            else:
                model = lgb.LGBMRegressor()
            
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=param_grid,
                                       cv=3,
                                       n_jobs=-1)
            grid_search.fit(train_X, train_Y)
            best_model = grid_search.best_estimator_
            return best_model
        except Exception as e:
            print('grid_tune has error : ' + e)
            return None 
    
    def random_tune(train_X, train_Y, fold_time, isClassifier=True, param_grid=None) : 
        """
        使用 RandomizedSearchCV 進行隨機調參，返回最佳模型。
        
        Args:
        train_X (DataFrame): 訓練數據集的特徵值。
        train_Y (Series or array-like): 訓練數據集的目標值。
        fold_time (int): 交叉驗證次數。
        isClassifier (bool, optional): 是否使用分類器，默認為 True。
        param_grid (dict or None, optional): 隨機搜索的參數網格，默認為 None。
        
        Returns:
        object: 最佳模型。
        """
        try:
            if isClassifier :
                model = lgb.LGBMClassifier()
            else : 
                model = lgb.LGBMRegressor()
            
            random_search = RandomizedSearchCV(estimator = model,
                                               param_distributions = param_grid,
                                               cv = fold_time,
                                               n_jobs = -1,)
            random_search.fit(train_X, train_Y)
            best_model = random_search.best_estimator_
            return best_model
        except Exception as e:
            print('random_tune has error : ' + e)
            return None

if __name__ == '__main__':
    from ucimlrepo import fetch_ucirepo
    default_of_credit_card_clients = fetch_ucirepo(id=350)
    train_X = default_of_credit_card_clients.data.features
    train_Y = default_of_credit_card_clients.data.targets
    train_X.columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
                       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    params = {
              'boosting_type': 'dart', #生成方式 gbdt, dart, rf 
              'n_estimators' : 1000,
              'learning_rate': 0.05,
              'n_jobs' : -1, #執行所有CPU
              'random_state' : 7,
              'verbose' : 0
              }
    
    '違約預測 (二元分類預測)'
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

    'PAY_AMT6 第 6期付款金額預測 (數值預測)'
    scoring = {
               'r2_score' : make_scorer(r2_score), 
               'mae' : make_scorer(mean_absolute_error),
               'mape' : make_scorer(mean_absolute_percentage_error), 
               }
    
    train_X_fs, feature_name = LightGBM.permutation_selection(train_X.drop(['PAY_AMT6'], axis=1), 
                                                              train_X['PAY_AMT6'], 
                                                              params = params,
                                                              imp = 0.005,
                                                              isClassifier=False)
    model, cv, cv_idx = LightGBM.build_model(train_X_fs, train_X['PAY_AMT6'], 
                                             params = params, scoring = scoring, 
                                             fold_time = 5, isClassifier=False)

    model_grid_tune = LightGBM.grid_tune(train_X_fs, train_X['PAY_AMT6'], 
                                         fold_time = 3, param_grid = param_grid,
                                         isClassifier=False)
    
    model_random_tune = LightGBM.random_tune(train_X_fs, train_X['PAY_AMT6'], 
                                             fold_time = 3, param_grid = param_grid,
                                             isClassifier=False)
    
