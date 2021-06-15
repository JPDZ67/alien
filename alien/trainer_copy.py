import joblib
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from termcolor import colored
from alien.utils import compute_rmse
from alien.encoders import TimeFeaturesEncoder
from alien.data import CAT_FEATURES, NUM_FEATURES, BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, STORAGE_LOCATION
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Trainer(object):
    def __init__(self, X, y, **kwargs):
        self.kwargs = kwargs
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get('split', True)
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = self.time_split_by_state(self.X_train, self.y_train, test_size = 0.05)  # train_test_split(self.X_train, self.y_train, test_size=0.2)
        self.pipeline = None

    def time_split(self, X, y, test_size=0.05):
        test_rows = int(len(X) * test_size)
        train_rows = len(X) - test_rows

        print(f"X train size - {len(X[:train_rows])}")
        print(f"X test size - {len(X[train_rows:])}")

        return X[:train_rows], X[train_rows:], y[:train_rows], y[train_rows:]
    
    def time_split_by_state(self, X, y, test_size = 0.05):
        df = pd.concat([X,y], axis=1)
        states_ = list(df.state.unique())
        df_X_train_aux = pd.DataFrame()
        df_X_test_aux = pd.DataFrame()
        df_y_train_aux = pd.DataFrame()
        df_y_test_aux =pd.DataFrame()

        for st_ in states_:
            aux = df.loc[X.state == st_]
            y = aux['sightings_t+1']
            X = aux.drop('sightings_t+1', axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#self.time_split(X, y, test_size=0.05)
            df_X_train_aux = df_X_train_aux.append(X_train)
            df_X_test_aux = df_X_test_aux.append(X_test)
            df_y_train_aux = df_y_train_aux.append(y_train)
            df_y_test_aux = df_y_test_aux.append(y_test)
        
        return df_X_train_aux, df_X_test_aux, df_y_train_aux, df_y_test_aux
    

    def get_estimator(self):
        estimator = self.kwargs.get('estimator')
        if estimator == 'Lasso':
            model = Lasso()
        elif estimator == 'Ridge':
            model = Ridge()
        elif estimator == "Linear":
            model = LinearRegression()
        elif estimator == "GBM":
            model = GradientBoostingRegressor(loss='huber', learning_rate=0.2, n_estimators=250)
        else:
            model = Lasso()

        return model

    def set_pipeline(self):
        datetime_pipe = make_pipeline(TimeFeaturesEncoder(time_column='year_season'),
                                      OneHotEncoder(handle_unknown='ignore'))

        categorical_pipe = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

        population_pipe = make_pipeline(SimpleImputer())

        numerical_pipe = make_pipeline(StandardScaler())

        feateng_blocks = [
            ('datetime', datetime_pipe, ['year_season']),
            ('categorical', categorical_pipe, CAT_FEATURES),
            ('population', population_pipe, ['population']),
            ('numerical', numerical_pipe, NUM_FEATURES)
        ]

        features_encoder = ColumnTransformer(feateng_blocks, remainder='drop')

        self.pipeline = Pipeline(steps=[
            ('features', features_encoder),
            ('model', self.get_estimator())
        ])

    def train(self):
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        print(self.pipeline.get_params())

    def evaluate(self):
        y_pred = self.pipeline.predict(self.X_train)
        abs_error = mean_absolute_error(self.y_train, y_pred)

        print(colored("Absolute error: {}".format(abs_error), "blue"))

    def fine_tune(self):
        self.set_pipeline()
        pipe = self.pipeline
        grid = GridSearchCV(pipe, verbose=3, param_grid={
            'model__learning_rate': [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3],
            'model__loss': ['ls', 'huber', 'lad'],
            #'model__max_depth': [1, 2, 3, 5, 6, 7, 8, 9, 10],
            'model__max_features': ['auto', 'sqrt', 'log2'],
            #'model__min_samples_leaf': [1, 2, 3, 4, 5],
            #'model__min_samples_split': [1.0, 2, 3],
            #'model__min_weight_fraction_leaf': [0.1, 0.5, 0.0],
            'model__n_estimators': [50, 75, 100, 125, 150, 175, 200, 250],
            #'model__subsample': [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        })

        grid.fit(self.X_train, self.y_train)
        print(grid.best_params_)

    def upload_model_to_gcp(self):
        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(STORAGE_LOCATION)

        blob.upload_from_filename('model.joblib')

    def save_model(self, model):
        joblib.dump(model, 'model.joblib')
        print(colored("--- Model Saved Locally", "green"))

        self.upload_model_to_gcp()
        print(colored("--- Model Saved to GCP", "blue"))


if __name__ == "__main__":
    params = dict(split=True,
                  estimator='GBM')

    df = pd.read_csv('/Users/juan/code/Polanket/alien/raw_data/final_df.csv')
    df.drop(columns=['year', 'season'],
            inplace=True,
            errors='ignore')
    y = df['sightings_t+1']
    X = df.drop('sightings_t+1', axis=1)
    del df
    t = Trainer(X=X, y=y, **params)
    del X, y

    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    #print(colored("############  Grid search model ############", "green"))
    #t.fine_tune()
