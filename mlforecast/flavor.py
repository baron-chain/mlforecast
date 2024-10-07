import os
from typing import Any, Dict, Optional, Union, Iterable

import mlflow
import pandas as pd
import yaml
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

import mlforecast
from mlforecast import MLForecast

FLAVOR_NAME = "mlforecast"
_MODEL_DATA_SUBPATH = "mlforecast-model"

def get_default_pip_requirements() -> list:
    """
    Create list of default pip requirements for MLflow Models.

    Returns:
        list: Default pip requirements for MLflow Models produced by this flavor.
    """
    return [_get_pinned_requirement("mlforecast")]

def get_default_conda_env() -> dict:
    """
    Return default Conda environment for MLflow Models.

    Returns:
        dict: The default Conda environment for MLflow Models produced by calls to
        save_model() and log_model().
    """
    return _mlflow_conda_env(additional_conda_deps=get_default_pip_requirements())

def save_model(
    model: MLForecast,
    path: str,
    conda_env: Union[dict, str] = None,
    code_paths: Optional[Iterable[str]] = None,
    mlflow_model: Optional[Model] = None,
    signature: Optional[mlflow.models.ModelSignature] = None,
    input_example: Optional[Union[pd.DataFrame, dict, list]] = None,
    pip_requirements: Optional[Union[Iterable[str], str]] = None,
    extra_pip_requirements: Optional[Union[Iterable[str], str]] = None,
) -> None:
    """
    Save an MLForecast model to a local path.

    Args:
        model (MLForecast): Fitted MLForecast model object.
        path (str): Local path where the model is to be saved.
        conda_env (Union[dict, str], optional): Either a dictionary representation of a 
            Conda environment or the path to a conda environment yaml file.
        code_paths (Optional[Iterable[str]]): A list of local filesystem paths to Python 
            file dependencies (or directories containing file dependencies).
        mlflow_model (Optional[Model]): MLflow model configuration to which to add the 
            python_function flavor.
        signature (Optional[mlflow.models.ModelSignature]): Model signature describing 
            model input and output Schema.
        input_example (Optional[Union[pd.DataFrame, dict, list]]): Input example provides 
            one or several instances of valid model input.
        pip_requirements (Optional[Union[Iterable[str], str]]): Pip requirements specification.
        extra_pip_requirements (Optional[Union[Iterable[str], str]]): Additional pip requirements.

    Raises:
        MlflowException: If there's an error during the model saving process.
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_path = os.path.join(path, _MODEL_DATA_SUBPATH)
    model.save(model_data_path)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlforecast.flavor",
        model_path=_MODEL_DATA_SUBPATH,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=_MODEL_DATA_SUBPATH,
        mlforecast_version=mlforecast.__version__,
        serialization_format="cloudpickle",
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))

def log_model(
    model: MLForecast,
    artifact_path: str,
    conda_env: Optional[Union[dict, str]] = None,
    code_paths: Optional[Iterable[str]] = None,
    registered_model_name: Optional[str] = None,
    signature: Optional[mlflow.models.ModelSignature] = None,
    input_example: Optional[Union[pd.DataFrame, dict, list]] = None,
    await_registration_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: Optional[Union[Iterable[str], str]] = None,
    extra_pip_requirements: Optional[Union[Iterable[str], str]] = None,
    **kwargs: Any,
) -> mlflow.models.model.ModelInfo:
    """
    Log an MLForecast model as an MLflow artifact for the current run.

    Args:
        model (MLForecast): Fitted MLForecast model object.
        artifact_path (str): Run-relative artifact path to save the model to.
        conda_env (Optional[Union[dict, str]]): Either a dictionary representation of a 
            Conda environment or the path to a conda environment yaml file.
        code_paths (Optional[Iterable[str]]): A list of local filesystem paths to Python 
            file dependencies (or directories containing file dependencies).
        registered_model_name (Optional[str]): If given, create a model version under 
            registered_model_name, also creating a registered model if one with the given 
            name does not exist.
        signature (Optional[mlflow.models.ModelSignature]): Model signature describing 
            model input and output Schema.
        input_example (Optional[Union[pd.DataFrame, dict, list]]): Input example provides 
            one or several instances of valid model input.
        await_registration_for (int): Number of seconds to wait for the model version to 
            finish being created and is in READY status.
        pip_requirements (Optional[Union[Iterable[str], str]]): Pip requirements specification.
        extra_pip_requirements (Optional[Union[Iterable[str], str]]): Additional pip requirements.
        **kwargs: Additional arguments for mlflow.models.model.Model.

    Returns:
        mlflow.models.model.ModelInfo: Metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlforecast.flavor,
        registered_model_name=registered_model_name,
        model=model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )

def load_model(model_uri: str, dst_path: Optional[str] = None) -> MLForecast:
    """
    Load an MLForecast model from a local file or a run.

    Args:
        model_uri (str): The location, in URI format, of the MLflow model.
        dst_path (Optional[str]): The local filesystem path to which to download the model artifact.

    Returns:
        MLForecast: An MLForecast model instance.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    model_file_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    return MLForecast.load(model_file_path)

def _load_pyfunc(path: str) -> '_MLForecastModelWrapper':
    """
    Load PyFunc implementation. Called by pyfunc.load_model.

    Args:
        path (str): Local filesystem path to the MLflow Model with the mlforecast flavor.

    Returns:
        _MLForecastModelWrapper: Wrapped MLForecast model for PyFunc.
    """
    pyfunc_flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=pyfunc.FLAVOR_NAME)
    path = os.path.join(path, pyfunc_flavor_conf["model_path"])
    return _MLForecastModelWrapper(MLForecast.load(path))

class _MLForecastModelWrapper:
    def __init__(self, model: MLForecast):
        self.model = model

    def predict(
        self,
        config_df: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions using the wrapped MLForecast model.

        Args:
            config_df (pd.DataFrame): Configuration DataFrame for prediction.
            params (Optional[Dict[str, Any]]): Additional parameters for prediction.

        Returns:
            pd.DataFrame: Predictions generated by the model.

        Raises:
            MlflowException: If there's an error in the prediction process.
        """
        n_rows = config_df.shape[0]

        if n_rows > 1:
            raise MlflowException(
                f"The provided prediction DataFrame contains {n_rows} rows. "
                "Only 1 row should be supplied.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        attrs = config_df.iloc[0].to_dict()
        h = attrs.get("h")
        if h is None:
            raise MlflowException(
                "The `h` parameter is required to make forecasts.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        ts = self.model.ts
        col_types = {
            ts.id_col: ts.uids.dtype,
            ts.time_col: ts.last_dates.dtype,
        }
        level = attrs.get("level")
        new_df = attrs.get("new_df")
        if new_df is not None:
            if level is not None:
                raise MlflowException(
                    "Prediction intervals are not supported in transfer learning. "
                    "Please provide either `level` or `new_df`, but not both.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            new_df = pd.DataFrame(new_df).astype(col_types)
        X_df = attrs.get("X_df")
        if X_df is not None:
            X_df = pd.DataFrame(X_df).astype(col_types)
        ids = attrs.get("ids")
        return self.model.predict(h=h, new_df=new_df, level=level, X_df=X_df, ids=ids)
