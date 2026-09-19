import mlflow


def test_mlflow_connection():

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    experiment_name = "image-classification-mlops"

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name
        )
    else:
        experiment_id = experiment.experiment_id

    assert experiment_id is not None
