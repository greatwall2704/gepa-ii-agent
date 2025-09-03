# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


def initialize_mlflow(mlflow_tracking_uri: str | None = None, mlflow_experiment_name: str | None = None):
    try:
        import mlflow  # type: ignore

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        if mlflow_experiment_name:
            mlflow.set_experiment(mlflow_experiment_name)
    except ImportError:
        raise ImportError("mlflow is not installed. Please install it or set use_mlflow=False.")
    except Exception as e:
        raise RuntimeError(f"Error setting up mlflow: {e}")
