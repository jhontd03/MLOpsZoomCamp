import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-homework3")

mlflow.sklearn.autolog(disable=True)

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here

    lr = data[0]
    dv = data[1]

    with mlflow.start_run():
        mlflow.set_tag("developer", "jhon")
        mlflow.set_tag("model", "LinearRegression")

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")            
            
        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")

