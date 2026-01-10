from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, f_classif
from sklearn.impute import KNNImputer

import pandas as pd
import numpy as np

from operator import or_



class DecimalScaler:
    def fit_transform(self, series:pd.Series) -> pd.Series:
        # Find the maximum absolute value in the series
        max_abs = series.abs().max()
        # Calculate the scaling factor
        scale_factor = np.ceil(np.log10(max_abs))
        # Scale the series
        return series / (10 ** scale_factor)





# 1) Handling missing values
def handle_missing_values(df:pd.DataFrame, target:str) -> None:
    x          = df.drop(columns=[target])
    n_rows_tot = x.shape[0]
    n_cols_tot = x.shape[1]
    n_rows_na  = x.isna().any(axis=1).sum()
    n_cols_na  = x.isna().any(axis=0).sum()

    # print(n_rows_na)
    # print(df.isna())

    # If <= 10% of the rows have missing values, drop them
    if (n_rows_na / n_rows_tot) <= 0.1:
        to_drop = x.index[x.isna().any(axis='columns')]
        df.drop(to_drop, axis=0, inplace=True)
    # If <= 10% of the columns have missing values, drop them
    elif (n_cols_na / n_cols_tot) <= 0.1:
        to_drop = x.columns[x.isna().any(axis='index')]
        df.drop(to_drop, axis=1, inplace=True)
    # # Else, impute the missing values using the KNN algorithm
    # else:
    #     imputer = KNNImputer()
    #     new_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    #     for col in df.columns: df[col] = new_df[col]
    # Else, fill the missing values with the mean of the column if numerical, else the mode
    else:
        for col in x.columns:
            if pd.api.types.is_numeric_dtype(df[col]):    to_impute = x[col].mean()
            else:                                         to_impute = x[col].mode()[0]
            df[col].fillna(to_impute, inplace=True)



# 2) Nominal encoding
def nominal_encoding(df:pd.DataFrame, target:str='') -> None:
    # ret = None
    for col in df.columns:
        # if pd.api.types.is_string_dtype(df[col]):
        if (not pd.api.types.is_numeric_dtype(df[col])) or pd.api.types.is_bool_dtype(df[col]):
            # If the column is the target column and we don't want to encode it, skip it
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            # if col == target: ret = {i:label for i,label in enumerate(le.classes_)}
    
    # return ret



# 3) Feature selection
def feature_selection_corr_mi(df:pd.DataFrame, x:pd.DataFrame, y:pd.DataFrame, target:str, corr_threshold:float, mi_threshold:float) -> list[str]:
    # Use mutual information to weed out features which are too uncorrelated with the target
    mi = mutual_info_classif(x, y)
    # print('MI:', mi)
    col_mi = pd.Series(mi, index=x.columns)
    to_drop = set(col_mi[col_mi < mi_threshold].index)
    x.drop(to_drop, axis=1, inplace=True) # Drop rn so that we can check for correlation

    # Use correlation matrix as to weed out features which are highly correlated with each other. Remove the feature with lower MI
    corr_matrix = x.corr().abs()
    # print(corr_matrix)
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2 and corr_matrix[col1][col2] > corr_threshold:
                col1_mi = col_mi[col1]
                col2_mi = col_mi[col2]
                to_drop.add(col1 if col1_mi < col2_mi else col2)
    
    df.drop(to_drop, axis=1, inplace=True)


def feature_selection_chi2(df:pd.DataFrame, x:pd.DataFrame, y:pd.DataFrame, n_features:int) -> None:
    # Ensure non-negative (for chi2)
    x_scaled = MinMaxScaler().fit_transform(x)
    # Compute the chi2 scores
    chi2_scores, _ = chi2(x_scaled, y)
    chi2_series = pd.Series(chi2_scores, index=x.columns).sort_values(ascending=False)
    # Select the top n features
    to_drop = chi2_series.index[n_features:]

    df.drop(to_drop, axis=1, inplace=True)
    

def feature_selection_anova(df:pd.DataFrame, x:pd.DataFrame, y:pd.DataFrame, n_features:int) -> None:
    # Compute the ANOVA F-statistic
    f_scores, _ = f_classif(x, y)
    f_series = pd.Series(f_scores, index=x.columns).sort_values(ascending=False)
    # Select the top n features
    to_drop = f_series.index[n_features:]

    df.drop(to_drop, axis=1, inplace=True)


def feature_selection(df:pd.DataFrame, target:str, corr_threshold:float=0.8, mi_threshold:float=0.1, n_features:int=8, method:str='corr+mi') -> None:
    x = df.drop(target, axis=1)
    y = df[target]
    
    if method == 'corr+mi': feature_selection_corr_mi(df, x, y, target, corr_threshold, mi_threshold)
    elif method == 'chi2':  feature_selection_chi2(df, x, y, n_features)
    elif method == 'anova': feature_selection_anova(df, x, y, n_features)
    else: raise ValueError('Method must be either "corr+mi", "chi2" or "anova"')

    '''
    In all the 3, higher -> better
    '''



# 4) Scaling
def scale(df:pd.DataFrame, target:str, scaler_type:str='minmax') -> None:
    if scaler_type not in ['minmax', 'standard', 'robust', 'decimal']: raise ValueError('Scaler must be either "minmax", "standard", "robust" or "decimal"')
    
    match scaler_type:
        case 'minmax':   scaler = MinMaxScaler()
        case 'standard': scaler = StandardScaler()
        case 'robust':   scaler = RobustScaler()
        case 'decimal':  scaler = DecimalScaler()
    
    for col in df.columns:
        if col == target: continue
        df[col] = scaler.fit_transform( df[[col]] )







def preprocess(df:pd.DataFrame, target:str, corr_threshold:float=0.8, mi_threshold:float=0.1, scaler:str='minmax', n_features:int=8, fs_method:str='corr+mi') -> pd.DataFrame:
    handle_missing_values(df, target)
    nominal_encoding(df)

    df_wo_fs = df.copy()
    feature_selection(df, target, corr_threshold, mi_threshold, n_features, fs_method)

    scale(df,       target, scaler)
    scale(df_wo_fs, target, scaler)

    return df_wo_fs



#df = pd.read_csv('/home/mark/Documents/breast-cancer-wisconsin.csv')
#df.replace('?', np.nan, inplace=True)
#handle_missing_values(df, 'Class')

#df.to_csv('cancer-data.csv', index=False)


