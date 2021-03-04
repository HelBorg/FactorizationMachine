import numpy as np
import pandas as pd
import scipy.sparse
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


class TransformData:
    col_names = ['Customer_ID', 'Rating', 'Date']
    chunk_size = 10000

    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, header=None,
                              names=self.col_names, usecols=[0, 1, 2])
        self.y = None
        self.X = None

    def transform(self):
        self._add_movie_id()
        self._add_month_column()

        # Add columns corresponding to each customer and each movie
        self._encode_data()

    def save_data(self, data_path):
        self.df.to_csv(data_path, header=True, index=False)

    def load_data(self, data_path):
        self.df = pd.read_csv(data_path, header=0)
        self._encode_data()

    def get_fold(self, n_folds=5):
        kf = KFold(n_splits=n_folds, shuffle=True)
        for train_index, test_index in kf.split(self.df):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            yield X_train, X_test, y_train, y_test

    def _add_movie_id(self):
        ###
        # Add column Movie_ID and remove rows corresponding to Movie_ID
        # self contains field df - DataFrame
        # df = "Movie_ID:\n   Customer_ID Rating Date\n Customer_ID Rating Date\n ..."
        ###
        df_nan = pd.DataFrame(pd.isnull(self.df.Rating))
        df_nan = df_nan[df_nan['Rating']]
        df_nan.reset_index(inplace=True)

        movie_np = []
        movie_id = 1

        # Generate movie_id column
        for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
            movie_np = np.append(movie_np, np.full((1, i - j - 1), movie_id))
            movie_id += 1

        # Account for last record and corresponding length
        last_record = np.full((1, len(self.df) - df_nan.iloc[-1, 0] - 1), movie_id)
        movie_np = np.append(movie_np, last_record)

        # Remove rows corresponding to Movie ID
        self.df = self.df[pd.notnull(self.df['Rating'])]
        movie_np = movie_np.astype(int)
        self.df['Movie_ID'] = movie_np

        self.df.dropna(inplace=True)

    def _add_month_column(self):
        ###
        # Add column Month
        # self.df.Date - date in format "yyyy-mm-dd"
        ###
        date_time = pd.to_datetime(self.df["Date"])
        month = date_time.apply(lambda ts: ts.month)
        month.rename("Month")
        self.df = self.df.join(month.rename("Month")).drop("Date", 1)

    def _encode_data(self):
        encoder = OneHotEncoder(categories='auto')

        user_mat = encoder.fit_transform(np.asarray(self.df["Customer_ID"]).reshape(-1, 1))
        movie_mat = encoder.fit_transform(np.asarray(self.df["Movie_ID"]).reshape(-1, 1))
        month_mat = encoder.fit_transform(np.asarray(self.df["Month"]).reshape(-1, 1))

        self.X = scipy.sparse.hstack([user_mat, movie_mat, month_mat]).tocsr()
        self.y = np.asarray(self.df['Rating']).reshape(-1, 1)
