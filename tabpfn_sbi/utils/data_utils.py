import ast
import math
import os
import pickle
import uuid

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from sbi.diagnostics import check_tarp
from sbi.diagnostics.sbc import check_uniformity_frequentist

# Very simple db-like storage for results
COLUMNS = [
    "method",
    "estimator",
    "embedding_net",
    "task",
    "num_simulations",
    "seed",
    "model_id",
    "observation_ids",
    "metric",
    "value",
    "time_train",
    "time_eval",
    "cfg",
]


def init_dir(dir_path: str) -> None:
    """Initializes a directory for storing models and summary.csv.

    Args:
        dir_path (str): The path to the directory to initialize.
    """
    if not os.path.exists(dir_path + os.sep + "models"):
        os.makedirs(dir_path + os.sep + "models")

    if not os.path.exists(dir_path + os.sep + "summary.csv"):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(dir_path + os.sep + "summary.csv", index=False)


def get_summary_df(dir_path: str) -> pd.DataFrame:
    """Returns the summary.csv file as a pandas dataframe.

    Args:
        dir_path (str): The path to the directory containing summary.csv.

    Returns:
        pd.DataFrame: The summary dataframe.
    """
    df = None
    while df is None:
        # If another process is writing to the file, it might raise an exception
        try:
            df = pd.read_csv(dir_path + os.sep + "summary.csv")
        except Exception as e:
            print(e)
    return df


def generate_unique_model_id(dir_path: str) -> str:
    """Generates a unique model id for saving a model.

    Args:
        dir_path (str): The path to the directory containing summary.csv.

    Returns:
        int: A unique model id.
    """
    summary_df = get_summary_df(dir_path)
    model_ids = summary_df["model_id"].values
    if len(model_ids) == 0:
        return "0"
    elif len(model_ids) == 1:
        return "1"
    else:
        max_id = np.max(model_ids)
        return str(max_id + 1)


def generate_globally_unique_model_id() -> str:
    """Generates a unique model id for saving a model.

    Returns:
        str: UUID4 as string.
    """
    return str(uuid.uuid4())


def save_model(model: object, dir_path: str, model_id: str) -> None:
    """Saves a model to a file.

    Args:
        model (object): The model to save.
        dir_path (str): The path to the directory to save the model in.
        model_id (str): The unique id of the model.
    """
    file_name = dir_path + os.sep + "models" + os.sep + f"model_{model_id}.pkl"
    with open(file_name, "wb") as file:
        pickle.dump(model, file)


def save_summary(
    dir_path: str,
    method: str,
    estimator: str,
    embedding_net: str,
    task: str,
    num_simulations: int,
    model_id: str,
    observation_ids: list[int],
    metric: str,
    value: list[float],
    seed: int,
    time_train: float,
    time_eval: float,
    cfg: dict,
) -> None:
    """Saves a summary to the summary.csv file with thread-safe file locking.

    Args:
        dir_path (str): The path to the directory containing summary.csv.
        method (str): The method used.
        task (str): The task performed.
        num_simulations (int): The number of simulations.
        model_id (str): The unique id of the model.
        metric (str): The metric used.
        value (float): The value of the metric.
        seed (int): The seed used.
        time_train (float): The training time.
        time_eval (float): The evaluation time.
        cfg (dict): The configuration dictionary.
    """
    summary_file = os.path.join(dir_path, "summary.csv")
    lock_file = summary_file + ".lock"  # Creates a lock file next to the CSV

    # Create a file lock object
    lock = FileLock(lock_file)

    # Acquire the lock before writing
    with lock:
        summary_df = get_summary_df(dir_path)
        new_row = pd.DataFrame(
            {
                "method": method,
                "estimator": estimator,
                "embedding_net": embedding_net,
                "task": task,
                "num_simulations": num_simulations,
                "seed": seed,
                "model_id": model_id,
                "observation_ids": str(observation_ids),
                "metric": metric,
                "value": str(value),
                "time_train": str(time_train),
                "time_eval": str(time_eval),
                "cfg": str(cfg),
            },
            index=[len(summary_df)],
        )
        summary_df = pd.concat([summary_df, new_row], axis=0, ignore_index=True)
        summary_df.to_csv(summary_file, index=False)


def load_model(dir_path: str, model_id: str) -> object:
    """Loads a model from a file.

    Args:
        dir_path (str): The path to the directory containing the model.
        model_id (str): The unique id of the model.

    Returns:
        object: The loaded model.
    """
    file_name = dir_path + os.sep + "models" + os.sep + f"model_{model_id}.pkl"
    with open(file_name, "rb") as file:
        return pickle.load(file)


def query(
    name: str,
    method: str | list[str] | tuple[str] | None = None,
    estimator: str | list[str] | tuple[str] | None = None,
    embedding_net: str | list[str] | tuple[str] | None = None,
    task: str | list[str] | tuple[str] | None = None,
    num_simulations: int | list[int] | tuple[int] | None = None,
    seed: int | list[int] | tuple[int] | None = None,
    metric: str | list[str] | tuple[str] | None = None,
    reduce_fn: str | None = "auto",
    **kwargs,
) -> pd.DataFrame:
    """Queries the summary.csv file.

    Args:
        name (str): The name of the summary file.
        method (str | list[str] | tuple[str], optional): The method(s) to query. Defaults to None.
        task (str | list[str] | tuple[str], optional): The task(s) to query. Defaults to None.
        num_simulations (int | list[int] | tuple[int], optional): The number of simulation(s) to query. Defaults to None.
        seed (int | list[int] | tuple[int], optional): The seed(s) to query. Defaults to None.
        metric (str | list[str] | tuple[str], optional): The metric(s) to query. Defaults to None.

    Returns:
        pd.DataFrame: The queried summary dataframe.
    """
    summary_df = get_summary_df(name)
    query = ""
    query += to_query_string("method", method)
    query += to_query_string("estimator", estimator)
    query += to_query_string("embedding_net", embedding_net)
    query += to_query_string("task", task)
    query += to_query_string("num_simulations", num_simulations)
    query += to_query_string("seed", seed)
    query += to_query_string("metric", metric)

    if query.endswith(" & "):
        query = query[:-3]
    if query == "":
        df = summary_df
    else:
        print(query)
        df = summary_df.query(query)

    if reduce_fn == "mean":

        def apply_fn(x):
            items = eval(x)
            return sum(items) / len(items)

        df.loc[:, "value"] = df["value"].apply(apply_fn)

    elif reduce_fn == "auto":
        # If metric == c2st/swd/std_dist -> mean
        # If metric == tarp/sbc -> p-value

        def apply_fn(value, metric):
            items = ast.literal_eval(value)
            if metric in ["c2st", "swd", "standardized_distance", "nll"]:
                return sum(items) / len(items)
            elif metric == "tarp":
                alpha, ecp = items
                ecp = torch.tensor(ecp)
                alpha = torch.tensor(alpha)
                auc_of_diagonal = np.trapz(np.abs(ecp - alpha), alpha)
                return float(auc_of_diagonal)
            elif metric == "sbc":

                def convert_to_cdf(x):
                    x = np.array(x)
                    hist, *_ = np.histogram(x, bins=30, density=True)
                    histcs = np.cumsum(hist)
                    histcs = histcs / histcs[-1]
                    histcs = np.concatenate([[0], histcs])
                    alpha = np.linspace(0, 1, len(histcs))
                    return alpha, histcs

                alpha, ecp = convert_to_cdf(items)
                auc_of_diagonal = np.trapz(np.abs(ecp - alpha), alpha)
                return float(auc_of_diagonal)
            else:
                return items

        # Apply the function in a single pass
        df["value"] = df.apply(
            lambda row: apply_fn(row["value"], row["metric"]), axis=1
        )

    return df


def to_query_string(name: str, var: any, end: str = " & ") -> str:
    """Translates a variable to string for query.

    Args:
        name (str): The query argument.
        var (any): The value to query.
        end (str, optional): The ending string for the query. Defaults to " & ".

    Returns:
        str: The query string.
    """
    if var is None:
        return ""
    elif (
        var is pd.NA
        or var is torch.nan
        or var is math.nan
        or str(var) == "nan"
        or var is np.nan
    ):
        return f"{name}!={name}"
    elif isinstance(var, list) or isinstance(var, tuple):
        query = "("
        for v in var:
            if query != "(":
                query += "|"
            if isinstance(v, str):
                query += f"{name}=='{v}'"
            else:
                query += f"{name}=={v}"
        query += ")"
    else:
        if isinstance(var, str):
            query = f"({name}=='{var}')"
        else:
            query = f"({name}=={var})"
    return query + end
