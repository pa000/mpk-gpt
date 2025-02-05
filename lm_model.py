import config
import torch
from dataset import VehicleStopTime, get_dataset
from model import get_model

class LmModel:
    def __init__(self, e_fmt, r_fmt, max_iters, type, min_trip_len, max_trip_len, indirect=False):
        self.e_fmt = e_fmt
        self.e_fmt_len = VehicleStopTime.format_length(e_fmt)
        self.r_fmt = r_fmt
        self.r_fmt_len = VehicleStopTime.format_length(r_fmt)
        self.fmt = e_fmt + r_fmt
        self.fmt_len = self.e_fmt_len + self.r_fmt_len
        self.model = get_model(e_fmt, r_fmt, max_iters, type, min_trip_len, max_trip_len, indirect)
        self.dataset = get_dataset('test', min_trip_len, max_trip_len)
        self.max_trip_len = max_trip_len
        self.block_size = self.dataset.get_block_size(self.fmt)

        self.name = f'{min_trip_len}_{max_trip_len}_{e_fmt}_{r_fmt}_{type}_{max_iters}_{indirect}'

    # def predict_one(self, trip: list[VehicleStopTime], next_st: VehicleStopTime) -> VehicleStopTime | None:
    #     if self.max_trip_len is not None:
    #         trip = trip[-(self.max_trip_len - 1):]
            
    #     prompt_tokens = self.dataset.trip_to_tokens(trip, self.fmt)
    #     prompt_tokens += next_st.format(self.fmt)

    #     prompt_tokens = prompt_tokens[:-self.r_fmt_len]

    #     result = self.model.generate(torch.tensor([prompt_tokens], device=config.device), self.r_fmt_len)[0]

    #     pred_tokens = result[-self.fmt_len:].tolist()

    #     pred_vst = VehicleStopTime.parse(pred_tokens, self.fmt)

    #     return pred_vst

    def predict(self, batch_p: list[tuple[list[VehicleStopTime], list[VehicleStopTime], VehicleStopTime]]) -> list[VehicleStopTime | None]:
        # if len(batch) == 1:
            # return [ self.predict_one(*inp) for inp in batch ]

        if self.dataset.indirect:
            batch = [ (prf, st) for prf, _, st in batch_p ]
        else:
            batch = [ (prf + pred, st) for prf, pred, st, in batch_p ]

        if self.max_trip_len is not None:
            batch = [ (trip[-(self.max_trip_len - 1):], n) for trip, n in batch ]

           
        prompt_tokens = [ (self.dataset.trip_to_tokens(trip, self.fmt), n) for trip, n in batch ]
        prompt_tokens = [ tokens + n.format(self.fmt) for tokens, n in prompt_tokens ]

        prompt_tokens_t = torch.tensor([ self.dataset.pad(tokens, self.block_size)[0] for tokens in prompt_tokens ], device=config.device)
        prompt_tokens_t = prompt_tokens_t[:, :-self.r_fmt_len]

        result = self.model.generate(prompt_tokens_t, self.r_fmt_len)

        pred_tokens = result[:, -self.fmt_len:].tolist()

        pred_vsts = [ VehicleStopTime.parse(tokens, self.fmt) for tokens in pred_tokens ]

        return pred_vsts
