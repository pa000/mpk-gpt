import config
import torch
from torch.utils.data import DataLoader
from dataset import StoptimesDataset, get_dataset
from mingpt.model import GPT  # type: ignore[import-untyped]
from mingpt.trainer import Trainer  # type: ignore[import-untyped]

def get_model(e_fmt: str, r_fmt: str, max_iters: int, type: str, min_trip_len, max_trip_len, indirect=False):
    dataset = get_dataset('train', min_trip_len, max_trip_len, indirect)
    fname = f'model_{min_trip_len}_{max_trip_len}_{e_fmt}_{r_fmt}_{type}_{max_iters}_{indirect}.pt'
    train_dataset = StoptimesDataset(dataset, e_fmt, r_fmt)

    train_config = Trainer.get_default_config()
    train_config.max_iters = max_iters
    train_config.learning_rate = config.learning_rate
    try:
        model = torch.load(fname)
        trainer = Trainer(train_config, model, train_dataset)
        print(f'Loaded from {fname}')

    except FileNotFoundError:
        model_config = GPT.get_default_config()
        model_config.vocab_size = train_dataset.get_vocab_size()
        model_config.block_size = train_dataset.get_block_size()
        model_config.model_type = "gpt-" + type

        model = GPT(model_config)

        trainer = Trainer(train_config, model, train_dataset)

        def batch_end_callback(trainer):
            if trainer.iter_num % 50 == 0:
                print(f"\riter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}", end='')
        trainer.set_callback('on_batch_end', batch_end_callback)

        print()
        trainer.run()
        print()

        print('Trained')

        model.eval()

        torch.save(model, fname)

        print(f'Saved to {fname}')

    return model
