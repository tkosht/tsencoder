import pandas
from .timeset import TimeSeriesDataset


class DatasetCyclicAuckland(TimeSeriesDataset):
    def __init__(self, data_file="data/data.tsv", freq="D"):
        self.data_df = None
        self.train_df = None
        self.test_df = None

        self.data_file = data_file
        self.freq = freq

        self._load()._aggregate()
        assert self.data_df.index.name == "ds"
        assert "y" in self.data_df.columns

    def _load(self):
        data_df = pandas.read_csv(
            self.data_file, sep="\t", header=0, index_col="datetime", parse_dates=True
        )
        data_df.index.name = "ds"
        self.data_df = data_df
        return self

    @property
    def df(self) -> pandas.DataFrame:
        return self.data_df

    @property
    def data_coumns(self) -> list:
        return [col for col in self.df.columns if col != "ds"]

    def split(self, predict_date: str) -> (pandas.DataFrame, pandas.DataFrame):
        train_df = self.data_df[:predict_date].iloc[:-1]
        test_df = self.data_df[predict_date:]
        train_df.reset_index(inplace=True)
        test_df.reset_index(inplace=True)
        assert "ds" in train_df.columns
        assert "y" in train_df.columns
        assert "ds" in test_df.columns
        assert "y" in test_df.columns
        self.train_df = train_df
        self.test_df = test_df
        return train_df, test_df
