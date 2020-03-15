import pandas


class TimeSeriesDataset(object):
    def __init__(self, data_file="data/data.tsv", freq="D"):
        self.data_df = None
        self.data_file = data_file
        self._load()._aggregate()

    def _load(self):
        data_df = pandas.read_csv(
            self.data_file, sep="\t", header=0, index_col="ds", parse_dates=True
        )
        assert "ds" in data_df.columns
        assert "y" in data_df.columns
        self.data_df = data_df
        return self

    def _aggregate(self):
        self.data_df = self.data_df.resample(
            self.freq, label="left", closed="left"
        ).sum()
        return self

    def split(self, predict_date: str):
        raise NotImplementedError()
