# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any, Literal

LoggingBackend = Literal["wandb", "mlflow", "none"]


class ExperimentTracker:
    """
    Unified experiment tracking that supports both wandb and mlflow.
    """
    
    def __init__(
        self,
        backend: LoggingBackend = "none",
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
    ):
        self.backend = backend
        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs or {}
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        
        self._wandb_run = None
        self._mlflow_run = None
        
    def initialize(self):
        """Initialize the logging backend."""
        if self.backend == "wandb":
            self._initialize_wandb()
        elif self.backend == "mlflow":
            self._initialize_mlflow()
    
    def _initialize_wandb(self):
        """Initialize wandb."""
        try:
            import wandb  # type: ignore
            if self.wandb_api_key:
                wandb.login(key=self.wandb_api_key, verify=True)
            else:
                wandb.login()
        except ImportError:
            raise ImportError("wandb is not installed. Please install it or set backend='mlflow' or 'none'.")
        except Exception as e:
            raise RuntimeError(f"Error logging into wandb: {e}")
        
        self._wandb_run = wandb.init(**self.wandb_init_kwargs)
    
    def _initialize_mlflow(self):
        """Initialize mlflow."""
        try:
            import mlflow  # type: ignore
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            if self.mlflow_experiment_name:
                mlflow.set_experiment(self.mlflow_experiment_name)
        except ImportError:
            raise ImportError("mlflow is not installed. Please install it or set backend='wandb' or 'none'.")
        except Exception as e:
            raise RuntimeError(f"Error setting up mlflow: {e}")
    
    def start_run(self, nested: bool = False):
        """Start a new run."""
        if self.backend == "wandb":
            # wandb doesn't need explicit start_run, it's handled in init
            pass
        elif self.backend == "mlflow":
            import mlflow  # type: ignore
            self._mlflow_run = mlflow.start_run(nested=nested)
    
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """Log metrics to the active backend."""
        if self.backend == "wandb":
            import wandb  # type: ignore
            wandb.log(metrics, step=step)
        elif self.backend == "mlflow":
            import mlflow  # type: ignore
            mlflow.log_metrics(metrics, step=step)
    
    def end_run(self):
        """End the current run."""
        if self.backend == "wandb":
            import wandb  # type: ignore
            if wandb.run is not None:
                wandb.finish()
        elif self.backend == "mlflow":
            import mlflow  # type: ignore
            if mlflow.active_run() is not None:
                mlflow.end_run()
    
    def is_active(self) -> bool:
        """Check if there's an active run."""
        if self.backend == "wandb":
            import wandb  # type: ignore
            return wandb.run is not None
        elif self.backend == "mlflow":
            import mlflow  # type: ignore
            return mlflow.active_run() is not None
        return False


def create_experiment_tracker(
    use_wandb: bool = False,
    use_mlflow: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
) -> ExperimentTracker:
    """
    Create an experiment tracker based on the specified backend.
    
    Args:
        use_wandb: Whether to use wandb
        use_mlflow: Whether to use mlflow
        wandb_api_key: API key for wandb
        wandb_init_kwargs: Additional kwargs for wandb.init()
        mlflow_tracking_uri: Tracking URI for mlflow
        mlflow_experiment_name: Experiment name for mlflow
        
    Returns:
        ExperimentTracker instance
        
    Raises:
        ValueError: If both or neither backend is specified
    """
    if use_wandb and use_mlflow:
        raise ValueError("Cannot use both wandb and mlflow simultaneously. Choose one.")
    elif use_wandb:
        backend = "wandb"
    elif use_mlflow:
        backend = "mlflow"
    else:
        backend = "none"
    
    return ExperimentTracker(
        backend=backend,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
