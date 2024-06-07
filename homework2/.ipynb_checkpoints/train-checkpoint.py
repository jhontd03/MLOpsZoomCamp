import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment-homework2")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):


    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    mlflow.autolog()

    with mlflow.start_run():

        #tags are optional. They are useful for large teams and organization purposes
        #first param is the key and the second is the value
        mlflow.set_tag("developer", "Jhon")

        #log any param that may be significant for your experiment.
        #We've decided to track the datasets we're using.
        mlflow.log_param("train-data-path", "./output/train.pkl")
        mlflow.log_param("valid-data-path", "./output/test.pkl")


        #we're also logging hyperparams; alpha in this example
        
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        #and we're also logging metrics
        mlflow.log_metric("rmse", rmse)



if __name__ == '__main__':
    run_train()
