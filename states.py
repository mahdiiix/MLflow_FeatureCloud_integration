from FeatureCloud.app.engine.app import AppState, app_state
import mlflow
import subprocess
import random
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('next')  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        with mlflow.start_run():
            params = {'n_estimators': 100, 'max_depth': 10}
            #model = RandomForestClassifier(**params)
            mlflow.log_params(params)
            mlflow.log_metric('acc', random.random(), step=1)
            mlflow.log_metric('acc', random.random(), step=2)
            mlflow.log_metric('acc', random.random(), step=3)
            mlflow.log_metric('acc', random.random(), step=4)
            mlflow.log_metric('acc', random.random(), step=5)
            mlflow.log_metric('acc', random.random(), step=6)
            mlflow.log_metric('acc', random.random(), step=7)
            df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
            df.to_csv('result.csv')
            mlflow.log_artifact('result.csv')
            #mlflow.sklearn.log_model(model, 'random_forest', registered_model_name='RF')
        return 'next'  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.

@app_state('next')
class Next(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        while True:
            pass
        return 'terminal'
