import logging
import pandas as pd
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from pandas import DataFrame
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

logger = logging.getLogger()

class BaseDataset:
    def __init__(
            self
    ):
        self.df : DataFrame = DataFrame() 
        self.scaled_df : DataFrame | None = None
        # self.train_df : DataFrame | None = None
        # self.test_df : DataFrame | None = None
        self.x_pred = None 
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

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

    
    def scale_ds(self, target : Literal["dataframe", "splitted"] = "splitted", scale_target: bool = False):

        std_scaler = StandardScaler()

        if target == "dataframe":
            self.scaled_df = std_scaler.fit_transform( self.df )
            return self.scaled_df
        
        elif target == "splitted":
            std_scaler.fit( self.x_train )
            self.x_train = std_scaler.transform( self.x_train )
            self.x_test = std_scaler.transform( self.x_test )

        if scale_target:
            std_scaler_y = StandardScaler()

            y_train = self.y_train.to_numpy().reshape(-1, 1) 
            y_test = self.y_test.to_numpy().reshape(-1, 1)

            std_scaler_y.fit(y_train)
            self.y_train = std_scaler_y.transform(y_train)
            self.y_test = std_scaler_y.transform(y_test)

        
        return self.x_train, self.y_train, self.x_test, self.y_test

    def split_dataset(self, x_train_cols: list[str], y_train_col: str, test_size: float = 0.2 , shuffle_ds: bool = True, stratify_y: bool = False, args: dict = {}):

        if shuffle_ds:
            self.df = shuffle(self.df)
        
        x_pred = self.df[ self.df[y_train_col].isnull() ][x_train_cols]
        x_pred.dropna(inplace=True)

        cols = list(x_train_cols) + [y_train_col]
        traind_ds = self.df[cols].dropna()
        x = traind_ds[x_train_cols]
        y = traind_ds[y_train_col]

        if stratify_y:
            args = {"stratify": y}
        args["shuffle"] = shuffle_ds
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, **args)

        self.x_pred = x_pred
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        return x_pred, x_train, x_test, y_train, y_test

    def linear_regresion(self, x_train=None, x_test=None, y_train=None, y_test=None, calculate_metrics: bool = True):
        
        linear_model = LinearRegression()

        x_train = x_train if x_train else self.x_train
        x_test = x_test if x_test else self.x_test
        y_train = y_train if y_train else self.y_train
        y_test = y_test if y_test else self.y_test
        
        linear_model.fit(x_train, y_train)
        y_pred = linear_model.predict(x_test)
        if calculate_metrics:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            return linear_model, y_pred, r2, mse
        else :
            return linear_model, y_pred, None, None

    def linear_prediction(self, x_pred):
        linear_model = LinearRegression()
        linear_model.fit(self.x_train, self.y_train)
        return linear_model.predict(x_pred)

    def plot_pie_classes(self, col : str, title: str = "", figsize: tuple = (12, 12), args: dict = {}):

        fig, ax = plt.subplots(figsize=figsize)

        data = self.df[col].value_counts()
        data.plot.pie(y=col, ax=ax, title=title, **args)

        return data
    
    def smote_ds(self, inplace: bool = False, random_state:int|None=42, smote_args: dict = {}, resample_args: dict = {}):

        if random_state:
            smote_args["random_state"] = random_state

        if self.x_train is None or self.y_train is None:
            raise ValueError("scaled data is not available, split the dataset first")
        
        smote = SMOTE(**smote_args)

        x = self.x_train
        y = self.y_train

        xsmote, ysmote = smote.fit_resample(x, y)

        if inplace:
            self.x_train = xsmote
            self.y_train = ysmote
        return xsmote, ysmote

    def plot_distributions(self, y_col: str, cols: list[str] =[], figsize: tuple = (18, 12), max_cols : int=6 , args: dict = {}):

        if not cols:
            cols = list(self.df.columns)
            cols.remove(y_col)

        n_rows = int(len(cols)/max_cols) + (len(cols)%max_cols>0)
        n_cols = max_cols

        plt.figure()
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.subplots_adjust(hspace=0.5)
        
        for i, col in enumerate(cols):
            f = i // n_cols
            c = i % n_cols
            
            sbn.kdeplot(self.df, x=col, hue=y_col, ax=axs[f, c])
            axs[f, c].set_title(col)
            axs[f, c].set_xlabel('')
            axs[f, c].set_ylabel('')
        plt.show()  

    def plot_PCA(self,
                 n_components: int|None =None,
                 figsize= (20,20)
                 ):
        
        tmp_df = self.df.copy()
              
        if not n_components:
            n_components = len(tmp_df.columns)

        tmp_df.dropna(inplace=True)
        pca_pipe = make_pipeline(StandardScaler(), PCA())
        pca_pipe.fit(tmp_df)
        pca_model = pca_pipe.named_steps['pca'] 
        pca_df = pd.DataFrame(
            data    = pca_model.components_,
            columns = tmp_df.columns,
            index   = [f'PC{i}'for i in range(1, pca_model.n_components_ + 1)]
        )

        print(pca_model.explained_variance_ratio_)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

        plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        componentes = pca_model.components_
        plt.imshow(componentes.T, cmap='viridis', aspect='auto')
        plt.yticks(range(len(tmp_df.columns)), tmp_df.columns)
        plt.xticks(range(len(tmp_df.columns)), np.arange(pca_model.n_components_) + 1)
        plt.grid(False)
        plt.colorbar()
        plt.show()



        return pca_model.explained_variance_ratio_

    # def generate_train_test_df(self, x_cols : list[str], y_col:str):

    #     df_x_train = pd.DataFrame(self.x_train_scaled, columns=x_cols)
    #     df_y_tran = pd.DataFrame(self.y_train, columns=[y_col])
    #     self.train_df = pd.concat([df_x_train, df_y_tran], axis=1)
        
    #     df_x_test = pd.DataFrame(self.x_test_scaled, columns=x_cols)
    #     df_y_test = pd.DataFrame(self.y_test, columns=[y_col])
    #     self.test_df = pd.concat([df_x_test, df_y_test], axis=1)

    #     return self.train_df, self.test_df