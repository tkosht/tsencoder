import yaml
import pandas
import joblib
import pathlib

# from typing import Union


def load_params() -> dict:
    with open("conf/main.yml", "r") as f:
        params = yaml.full_load(f)
    return params


def dump_model(model, dump_file):
    pathlib.Path(dump_file).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, dump_file, compress=("gzip", 3))


if __name__ == "__main__":
    from dataset.auckset import DatasetCyclicAuckland

    # from sklearn.pipeline import Pipeline
    # from .score import Score

    params = load_params()

    freq = params["prediction"]["freq"]
    predict_date = params["prediction"]["predict_date"]
    predict_by = params["prediction"]["predict_by"]

    holidays_df = pandas.read_csv("data/holiday.tsv", sep="\t", header=0)
    dcaset = DatasetCyclicAuckland(freq=freq)

    # split dataset
    train_df, test_df = dcaset.split(predict_date)

    # setup params
    fit_params = {"model__" + k: v for k, v in params["model"]["fit"].items()}
    predict_params = dict(predict_by=predict_by, freq=freq)

    print("OK")
