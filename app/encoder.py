import pandas
import torch
import torch.nn as nn
import argparse

import mlflow
import tempfile
from tensorboardX import SummaryWriter


def create_mask(n_rows, n_cols, offset=(0, 0), device=torch.device("cpu")):
    ox, oy = offset
    mask = torch.zeros((n_rows, n_cols))
    for idx in range(len(mask) - ox):
        mask[ox + idx, : oy + idx + 1] = 1
    return mask.to(device)


def create_mask_memory(n_rows, n_cols, device):
    return create_mask(n_rows, n_cols, offset=(1, 0), device=device)


def convert_transfomer_data(df: pandas.DataFrame):
    src = df.copy()
    tgt = df.copy()
    return src, tgt


class BatchIterator(object):
    def __init__(
        self, df: pandas.DataFrame, bsz=32, wsz=16, device=torch.device("cpu")
    ):
        self.cur = -1
        self.bsz = bsz
        self.wsz = wsz
        self.df = pandas.concat([df, df.head(bsz)], axis=0)
        self.df.reset_index(inplace=True, drop=True)
        self.n = len(df)
        self.device = device

    def __iter__(self):
        self.cur = self.wsz
        return self

    def __next__(self):
        if self.cur >= self.n:  # over the index size
            raise StopIteration()
        return self._get_batch()

    def _get_batch(self):
        b = self.cur
        # for b in range(self.wsz, self.n, self.bsz):
        bch_src = []
        bch_tgt = []
        for idx in range(bsz):
            df_batch = self.df[b + idx - wsz : b + idx]
            src, tgt = convert_transfomer_data(df_batch)
            bch_src.append(src.values)
            bch_tgt.append(tgt.values)
        # (N, S|T, E) -> (S|T, N, E)
        bch_src = torch.Tensor(bch_src).permute(1, 0, 2)
        bch_tgt = torch.Tensor(bch_tgt).permute(1, 0, 2)
        self.cur += self.bsz
        return bch_src.to(device), bch_tgt.to(device)


def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)


def get_args():
    psr = argparse.ArgumentParser()
    psr.add_argument("--predict-date", type=str, default="2016-01-01")
    psr.add_argument("--epoch", type=int, default=100)
    psr.add_argument("--window-size", type=int, default=17)
    psr.add_argument("--batch-size", type=int, default=32)
    psr.add_argument("--log-interval", type=int, default=10)
    args = psr.parse_args()
    return args


def log_weights(model, step):
    # writer.add_histogram("weights/transformer/weight", model.weight.data, step)
    # model.encoder.layers[0].self_attn.out_proj.weight
    for idx, layer in enumerate(model.encoder.layers):
        attn = layer.self_attn.out_proj
        writer.add_histogram(
            f"weights/encoder/self_attn/{idx}/weight", attn.weight.data, step
        )
    for idx, layer in enumerate(model.decoder.layers):
        attn = layer.self_attn.out_proj
        writer.add_histogram(
            f"weights/decoder/self_attn/{idx}/weight", attn.weight.data, step
        )


if __name__ == "__main__":
    import time
    from module.dataset.auckset import DatasetCyclicAuckland

    args = get_args()

    # parameters
    predict_date = args.predict_date

    # load data
    aucset = DatasetCyclicAuckland("data/data.tsv")
    df_train, df_test = aucset.split(predict_date)

    dim = len(aucset.data_coumns)
    n_heads = 1
    assert dim % n_heads == 0
    model = nn.Transformer(d_model=dim, nhead=n_heads, num_encoder_layers=12)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = df_train

    model.train()

    # max_epoch = 100
    # wsz = 17  # window size / bptt size
    # bsz = 32  # batch size
    # log_interval = 10
    max_epoch = args.epoch
    wsz = args.window_size  # window size / bptt size
    bsz = args.batch_size  # batch size
    log_interval = args.log_interval

    total_loss = 0

    output_dir = tempfile.mkdtemp()
    writer = SummaryWriter(output_dir)
    print("Writing TensorBoard events locally to %s\n" % output_dir)

    batch_creator = BatchIterator(df[aucset.data_coumns], bsz, wsz, device)
    start_time = time.time()
    for epoch in range(max_epoch):
        for idx, bch in enumerate(batch_creator):
            # src: (S, N, E), tgt: (T, N, E)
            src, tgt = bch
            S, T = len(src), len(tgt)
            msk_src = create_mask(S, S, device=device)
            msk_tgt = create_mask(T, T, device=device)
            msk_mem = create_mask_memory(T, S, device=device)

            # optimize
            optimizer.zero_grad()
            masks = dict(src_mask=msk_src, tgt_mask=msk_tgt, memory_mask=msk_mem)
            yhat = model(src, tgt, **masks)
            # y_dim = tgt.shape[-1]
            # loss = criterion(yhat.view(-1, y_dim), tgt)
            loss = criterion(yhat, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # print loss information
            total_loss += loss.item()
            if idx % log_interval == 0 and idx > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print(
                    f"| epoch {epoch} | {idx}/{len(df)//bsz} batches | "
                    f"lr {lr:02.2f} | ms/batch {elapsed:5.2f} | "
                    f"loss {cur_loss:5.2f}"
                )
                step = epoch * bsz + idx
                log_scalar("train_loss", total_loss, step)
                log_weights(model, step)

                total_loss = 0
                start_time = time.time()

    # Upload the TensorBoard event logs as a run artifact
    print("Uploading TensorBoard events as a run artifact...")
    mlflow.log_artifacts(output_dir, artifact_path="events")
