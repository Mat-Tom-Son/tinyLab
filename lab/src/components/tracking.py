"""MLflow tracking wrapper for local-first experiment logging."""
import mlflow


class MLFlowTracker:
    """Wrapper for MLflow experiment tracking."""

    def __init__(self, experiment, run_name):
        """Initialize tracker with experiment and run name."""
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment)
        self.run = mlflow.start_run(run_name=run_name)

    def log_param(self, key, value):
        """Log a single parameter."""
        mlflow.log_param(key, value)

    def log_params(self, params_dict):
        """Log multiple parameters from a flat dict."""
        mlflow.log_params(params_dict)

    def log_metric(self, key, value, step=None):
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics_dict, step=None):
        """Log multiple metrics."""
        mlflow.log_metrics(metrics_dict, step=step)

    def log_artifact(self, path):
        """Log a file as an artifact."""
        mlflow.log_artifact(path)

    def end_run(self):
        """End the current run."""
        mlflow.end_run()
