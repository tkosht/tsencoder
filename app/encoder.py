import pandas
import torch
import torch.nn as nn


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


if __name__ == "__main__":
    import time
    from module.dataset.auckset import DatasetCyclicAuckland

    # parameters
    predict_date = "2016-01-01"

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
    max_epoch = 100
    wsz = 17  # window size / bptt size
    bsz = 32  # batch size
    total_loss = 0
    log_interval = 10

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
                total_loss = 0
                start_time = time.time()
