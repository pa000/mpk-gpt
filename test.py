import sys
import random
from tqdm import tqdm  # type: ignore[import-untyped]
from datetime import datetime
from dataset import VehicleStopTime, test_trips, time_diff
from dataclasses import dataclass
from travel_time_model import TravelTimeModel
from lm_model import LmModel
from collections import defaultdict

BATCH_SIZE = 100
DATASET_SIZE = 1000

@dataclass
class Prediction:
    real: VehicleStopTime
    pred: VehicleStopTime | None
    steps: int

    def bad(self) -> bool:
        return self.pred is None

    def good(self) -> bool:
        if self.pred is None:
            return False
        return (self.real.rat.hour == self.pred.rat.hour
                and self.real.rat.minute == self.pred.rat.minute)

@dataclass
class Data:
    prefix: list[VehicleStopTime]
    predicted: list[VehicleStopTime]
    stoptime: VehicleStopTime
    suffix: list[VehicleStopTime]
    steps: int

def prepare_data(trip: list[VehicleStopTime]) -> list[Data]:
    data = []
    for i in range(1, len(trip)):
        pfx = trip[:i]
        stoptime = trip[i]
        sfx = trip[i+1:]
        
        d = Data(pfx, [], stoptime, sfx, 1)
        data.append(d)
    return data

def step(data: list[Data], preds: list[VehicleStopTime]) -> list[Data]:
    new_data = []
    for d, pred in zip(data, preds):
        if pred is None:
            continue

        if len(d.suffix) == 0:
            continue

        new_vst = VehicleStopTime(pred.rat, d.stoptime.eat, d.stoptime.code, [])
        new_vst.rrat = d.stoptime.rrat
        new_prefix = d.prefix
        new_predicted = d.predicted + [new_vst] 
        new_stoptime = d.suffix[0]
        new_suffix = d.suffix[1:]
        new_steps = d.steps + 1

        new_d = Data(new_prefix, new_predicted, new_stoptime, new_suffix, new_steps)
        new_data.append(new_d)
    return new_data

def process_batch(batch: list[Data], model) -> tuple[list[Prediction], list[Data]]:
    new_preds = model.predict([(d.prefix, d.predicted, d.stoptime) for d in batch])
    preds = [ Prediction(d.stoptime, pred, d.steps) for d, pred in zip(batch, new_preds) ]
    new_batch = step(batch, new_preds)

    return preds, new_batch

def predict_dataset(dataset: list[list[VehicleStopTime]], model) -> list[Prediction]:
    preds: list[Prediction] = []

    data: list[Data] = []
    for trip in dataset:
        data.extend(prepare_data(trip))

    batch: list[Data] = []

    t = tqdm(total=len(data))
    while len(batch) > 0 or len(data) > 0:
        if len(batch) < BATCH_SIZE and len(data) > 0:
            missing = BATCH_SIZE - len(batch)
            batch.extend(data[:missing])
            data = data[missing:]

        new_preds, new_batch = process_batch(batch, model)
        preds.extend(new_preds)
        t.update(len(batch) - len(new_batch))
        batch = new_batch
    t.close()
    return preds

dt = random.sample(test_trips, DATASET_SIZE)

def test_model(model):
    print(model.name)

    preds = predict_dataset(dt, model)
    bad = [ pred for pred in preds if pred.bad() ]
    good = [ pred for pred in preds if pred.good() ]

    print('bad', len(bad) / len(preds))
    print('good', len(good) / len(preds))
    return preds

def test_and_save(model):
    preds = test_model(model)
    with open(f'preds_{model.name}.csv', 'w') as file:
        file.write('real_time,pred_time,steps\n')
        for pred in preds:
            if pred is None:
                continue
            file.write(f'{pred.real.rat.strftime("%H:%M")},{pred.pred.rat.strftime("%H:%M")},{pred.steps}\n')  # type: ignore[union-attr]

test_and_save(TravelTimeModel())
test_and_save(LmModel("C", "rT", 1000, "nano", None, 3))
test_and_save(LmModel("CrT", "rT", 1000, "nano", None, 3))
test_and_save(LmModel("C[CT]", "rT", 1000, "nano", None, 3))
