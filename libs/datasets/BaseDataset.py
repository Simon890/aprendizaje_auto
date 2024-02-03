import logging
import pandas as pd
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger()

class BaseDataset:
    def __init__(
            self
    ):
        self.df : DataFrame = DataFrame() 
        self.scaled_df : DataFrame | None = None
        self.linear_model : LinearRegression = LinearRegression()
        self.std_scaler : StandardScaler = StandardScaler()
        self.x_pred = None 
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_pred_scaled = None 
        self.x_train_scaled = None
        self.x_test_scaled = None
        self.y_train_scaled = None
        self.y_test_scaled = None

    def load_csv(self, path:str)->DataFrame:
        try:
            self.df = pd.read_csv(path)
        except:
            logger.error(f"{path} could not be read")

        return self.df 

    def plot_nulls(self, yticklabels: bool=False, cmap:str="viridis", cbar:bool=False , args:dict={})->None:
        sbn.heatmap(self.df.isnull(), yticklabels=yticklabels, cmap=cmap, cbar=cbar, **args)

    def get_amount_of_rows_with_missing_vals(self)->int:
        return self.df.shape[0] - self.df.dropna().shape[0]
    
    def get_percent_of_row_with_missing_vals(self)->float:
        total_null_rows = self.get_amount_of_rows_with_missing_vals()
        total_rows = self.df.shape[0]
        return (total_null_rows * 100) / total_rows

    def sin_transformer(self, period):
        return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

    def cos_transformer(self, period):
        return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    def save(self, path:str, mode: str = "w", header:bool = True )->None:
        self.df.to_csv(path, index=False, mode=mode, header=header)
    
    def plot_interpolated_scatter(self, cols: list[str], method: str="linear", pd_data: pd.DataFrame| None = None, order:any = None, sx :int = None, sy:int = None, fsize : tuple=(12,12)):

        if pd_data is None:
            pd_data = self.df

        sx = int(len(cols)/2) + int(len(cols)%2) if not sx else sx 
        sy = 2 if not sy else sy 

        f, axs = plt.subplots(sx,sy, figsize=fsize, squeeze=False)
        
        i = 0
        for x in range(0, sx):
            for y in range(0,sy):
                if i > len(cols)-1:
                    break
                col = cols[i]
                data_to_plot = pd_data[[col]].copy()
                data_to_plot.loc[:, "is_nan"] = data_to_plot[col].isnull()
                data_to_plot.interpolate(method=method,order=order, inplace=True)
                sbn.scatterplot(data=data_to_plot, x=range(0, data_to_plot.shape[0] ), y=col, hue="is_nan", ax=axs[x][y])
                i+=1
        plt.show()  

    def plot_correlation(self, df:DataFrame|None=None, cols: list[str] = [], method:str = "pearson" , numeric_only=True,
                         yticklabels=True, cbar=False, annot=True, figsize=(50, 50)):

        if not df:
            df = self.df if not self.scaled_df else self.scaled_df

        if not cols:
            cols = df.columns

        heatmap_data = df[cols]

        plt.figure(figsize=figsize)
        sbn.heatmap(heatmap_data.corr(numeric_only=numeric_only, method=method), yticklabels=yticklabels, cbar=cbar, annot=annot)

    def scale_dataset(self, cols: list[str] = []):
        if not cols :
            cols = self.df.columns
        self.scaled_df = self.std_scaler.fit_transform( self.df[cols] )
        return self.scaled_df
    
    def scale_training_data(self):
        self.x_pred_scaled = self.std_scaler.fit_transform( self.x_pred )
        self.x_train = self.std_scaler.fit_transform( self.x_train )
        self.y_train = self.std_scaler.fit_transform( self.y_train )
        self.x_test = self.std_scaler.fit_transform( self.y_test )
        self.y_test = self.std_scaler.fit_transform( self.y_test )
        return self.x_pred_scaled, self.x_train_scaled, self.y_train_scaled, self.x_test_scaled, self.x_test_scaled

    def split_dataset(self, x_train_cols: list[str], y_train_col: str, test_size: float = 0.2 , dropna: bool = True, inplace: bool =True, shuffle_ds: bool = True):

        if shuffle_ds:
            self.df = shuffle(self.df)
        
        x_pred = self.df[ self.df[y_train_col].isnull() ][x_train_cols]
        x_pred.dropna(inplace=True)

        if dropna:
            self.df.dropna(inplace=inplace)

        x_train, x_test, y_train, y_test = train_test_split(self.df[x_train_cols], self.df[y_train_col], test_size=test_size)

        self.x_pred = x_pred
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        return x_pred, x_train, x_test, y_train, y_test

    def linear_regresion(self, x_train=None, x_test=None, y_train=None, y_test=None, calculate_metrics: bool = True):
        
        x_train = x_train if x_train else self.x_train
        x_test = x_test if x_test else self.x_test
        y_train = y_train if y_train else self.y_train
        y_test = y_test if y_test else self.y_test

        
        self.linear_model.fit(x_train, y_train)
        y_pred = self.linear_model.predict(x_test)
        if calculate_metrics:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            return self.linear_model, y_pred, r2, mse
        else :
            return self.linear_model, y_pred, None, None

    def linear_prediction(self, x_pred):
        return self.linear_model.predict(x_pred)

    def plot_pie_classes(self, col : str, title: str = "", figsize: tuple = (12, 12), args: dict = {}):

        tot_no = data["RainTomorrow"].value_counts()[0]
        tot_si = data["RainTomorrow"].value_counts()[1]
        
        porcentaje_si = round(tot_si / (tot_no + tot_si) * 100, 2)
        porcentaje_no = round(tot_no / (tot_no + tot_si) * 100, 2)
        data_cantidad = pd.DataFrame(columns=["RainTomorrow", "%"], index=["Si", "No", "Total"], data=[
            [tot_si, porcentaje_si],
            [tot_no, porcentaje_no],
            [tot_si + tot_no, porcentaje_si + porcentaje_no]
        ])

        plt.figure()
        plt.pie([tot_si, tot_no], labels=["Si", "No"], autopct='%1.1f%%')
        plt.show()
        plt.figure(figsize=figsize)
        self.df[col].value_counts().plot.pie(title=title, **args)