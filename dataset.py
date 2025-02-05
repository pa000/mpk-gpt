import torch
import random
import csv
import itertools
import sys
from torch.utils.data import Dataset
from typing import Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from zoneinfo import ZoneInfo

WR_TZ = ZoneInfo('Europe/Warsaw')

def adjust(dt: datetime, eat: datetime):
    if eat.day == 1 or eat.day == 2 and dt.hour > 12:
        return dt.replace(year=1990, month=1, day=1)
    else:
        return dt.replace(year=1990, month=1, day=2)

def timestamp(dt: datetime) -> int:
    return int((dt - datetime(1990, 1, 1, tzinfo=WR_TZ)).total_seconds())

def string_to_datetime(dt):
    try:
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f+00")
    except ValueError:
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S+00")

    dt = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(WR_TZ)
    return dt

def timestamp_to_datetime(t: int) -> datetime:
    return datetime(1990, 1, 1, tzinfo=WR_TZ) + timedelta(seconds=t)

def time_diff(t1, t2):
    diff1 = int((t1 - t2).total_seconds()) % 86400
    diff2 = int((t2 - t1).total_seconds()) % 86400

    return min(diff1, diff2)

def index_last(xs, x):
    return len(xs) - 1 - xs[::-1].index(x)

class VehicleStopTime:
    earlier: list[int] = []
    def __init__(self, real_arrival_time=None, expected_arrival_time=None, code=None, earlier=None, **kwargs):
        if expected_arrival_time is not None:
            if isinstance(expected_arrival_time, int):
                self.eat = timestamp_to_datetime(expected_arrival_time)
            elif isinstance(expected_arrival_time, datetime):
                self.eat = expected_arrival_time
            else:
                self.eat = timestamp_to_datetime(int(expected_arrival_time))
        if real_arrival_time is not None:
            if isinstance(real_arrival_time, int):
                self.rat = timestamp_to_datetime(real_arrival_time)
            elif isinstance(real_arrival_time, datetime):
                self.rat = adjust(real_arrival_time, self.eat)
            else:
                self.rrat = string_to_datetime(real_arrival_time)
                self.rat = adjust(self.rrat, self.eat)
        if code is not None:
            self.code = int(code)
            self.is_urban = len(str(code)) <= 5 or str(code)[0] == '1'
        if earlier is not None:
            if isinstance(earlier, list):
                self.earlier = earlier
            elif isinstance(earlier, str) and earlier != '':
                ids = list(map(int, earlier.split(';')))
                self.earlier = ids

    def __lt__(self, other):
        return self.rat < other.rat

    def __repr__(self):
        return f'{self.code} {self.rat.strftime("%H:%M:%S")} ({self.eat.strftime("%H:%M")})'

    def format_earlier(self, fmt: str) -> list[int]:
        etrips: list[list[VehicleStopTime]] = list(map(trip_by_id.get, self.earlier))  # type: ignore[arg-type]
        etrips = [ etrip[max([est.code for est in etrip].index(self.code)-1, 0):index_last([est.rrat < self.rrat for est in etrip], True) + 1] for etrip in etrips ]
        etrips = [ etrip for etrip in etrips if len(etrip) > 0 ]
        etrips = sorted(etrips, key=lambda etrip: etrip[0].rrat, reverse=True)

        etrip = etrips[0][:3] if len(etrips) > 0 else []
        result = []
        for st in etrip:
            result += st.format(fmt)

        result += [PLACEHOLDER] * VehicleStopTime.format_length(fmt) * (3 - len(etrip))

        return result

    def format(self, fmt: str) -> list[int]:
        result: list[int] = []
        target: datetime  = self.rat

        fmt_iter = iter(fmt)      
        for f in fmt_iter:
            if f == 'r':
                target = self.rat
            elif f == 'e':
                target = self.eat
            elif f in "HMS":
                result.extend(map(int, target.strftime(f'%{f}')))
            elif f == 'T':
                result.extend(map(int, str(timestamp(target)).rjust(6, "0")))
            elif f == 'C':
                result.append(token_by_code[self.code])
            elif f == '[':
                subformat = ''
                for f in fmt_iter:
                    if f == ']':
                        result.extend(self.format_earlier(subformat))
                        break
                    subformat += f
            else:
                raise ValueError(f'Invalid format directive: {f}')

        return result

    @classmethod
    def parse(cls, tokens: list[int], fmt: str):
        error = False
        def take(n, max_len: int | None=None) -> int:
            nonlocal tokens, error
            res = ''.join(map(str, tokens[:n]))
            if max_len is not None and len(res) > max_len:
                error = True
                res = '0'
            tokens = tokens[n:]
            return int(res)

        code = None
        eat = datetime(1900, 1, 1)
        rat = datetime(1900, 1, 1)
        target = { 'r': rat, 'e': eat }
        t = 'r'

        fmt_iter = iter(fmt)
        for f in fmt_iter:
            if f in "re":
                t = f
            elif f == 'H':
                try:
                    target[t] = target[t].replace(hour=take(2, 2))
                except ValueError:
                    return None
            elif f == 'M':
                try:
                    target[t] = target[t].replace(minute=take(2, 2))
                except ValueError:
                    return None
            elif f == 'S':
                try:
                    target[t] = target[t].replace(second=take(2, 2))
                except ValueError:
                    return None
            elif f == 'T':
                target[t] = take(6, 6)  # type: ignore[assignment]
            elif f == 'C':
                code = code_by_token.get(take(1))
                if code is None:
                    return None
            elif f == '[':
                subformat: str = ''
                for f in fmt_iter:
                    if f == ']':
                        take(3 * VehicleStopTime.format_length(subformat))
                        break
                    subformat += f
            else:
                raise ValueError(f'Invalid format directive {f}')

        if error:
            return None
        return VehicleStopTime(target['r'], target['e'], code)

    @classmethod
    def format_length(cls, fmt: str) -> int:
        res = 0        
        fmt_iter = iter(fmt)
        for f in fmt_iter:
            if f in "re":
                continue
            elif f in "HMS":
                res += 2
            elif f == 'T':
                res += 6
            elif f == 'C':
                res += 1
            elif f == '[':
                subfmt = ''
                for f in fmt_iter:
                    if f == ']':
                        res += 3 * cls.format_length(subfmt)
                        break
                    subfmt += f
            else:
                raise ValueError(f'Invalid format directive {f}')

        return res

def read_rows(file_name: str='stoptimes.csv'):
    file   = open(file_name)
    reader = csv.DictReader(file)

    rows = []
    for row in reader:
        rows.append(row)

    file.close()
    return rows

def group_trips(rows):
    groups = []
    ids = []
    for id, rows in itertools.groupby(rows, lambda r: r['trip_instance_id']):
        ids.append(int(id))
        groups.append(list(rows))

    return ids, groups

print('Loading dataset', file=sys.stderr)
        
rows = read_rows('stoptimes.csv')
ids, groups = group_trips(rows)
trips = [
    [ VehicleStopTime(**row) for row in group ]
    for group in groups
]
trip_by_id: dict[int, list[VehicleStopTime]] = dict(zip(ids, trips))

trips = [ trip for trip in trips if len(trip) > 1 ]
trips = [
    t for t in trips
    if all(st.is_urban for st in t)
]
trips = [
    t for t in trips
    if all(st1 < st2 for st1, st2 in zip(t, t[1:]))
]
trips = [
    t for t in trips
    if all(
        (st2.rat - st1.rat).total_seconds() <= 60 * 60
        for st1, st2 in zip(t, t[1:]))
]
all_trips = trips

TEST_PART = 4

PLACEHOLDER = 10
BASE_TOKENS = 10 + 1

train_size = len(all_trips) - len(all_trips) // TEST_PART
train_trips, test_trips = all_trips[:train_size], all_trips[train_size:]

print(f'Loaded {len(all_trips)} trips', file=sys.stderr)

codes = sorted({st.code for t in all_trips for st in t})
token_by_code = dict((c, i + BASE_TOKENS) for i, c in enumerate(codes))
code_by_token = { t : c for c, t in token_by_code.items() }

@dataclass
class DatasetInfo:
    dataset: list[list[VehicleStopTime]]
    get_block_size: Callable[[str], int]
    trip_to_tokens: Callable[[list[VehicleStopTime], str], list[int]]
    pad: Callable[[list[int], int], tuple[list[int], int]]
    vocab_size: int
    indirect: bool

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def get_dataset(split: str, min_trip_len: int | None=None, max_trip_len: int | None=None, indirect: bool=False) -> DatasetInfo:
    if split == 'train':
        trips = train_trips
    else:
        trips = test_trips

    if min_trip_len is not None:
        trips = [ t for t in trips if len(t) >= min_trip_len ]

    if max_trip_len is not None:
        trips_ = []
        for t in trips:
            if len(t) <= max_trip_len:
                trips_.append(t)
                continue

            if not indirect:
                for subtrip in [ t[i:i+max_trip_len] for i in range(len(t) - max_trip_len + 1) ]:
                    trips_.append(subtrip)
            else:
                for i in range(len(t) - max_trip_len + 1 - 1):
                     subtrip_1 = t[i:i+max_trip_len - 1]
                     for d in t[i+max_trip_len - 1:]:
                         trips_.append(subtrip_1 + [d])
                     
        trips = trips_

        if indirect:
            trips_ = []

            for t in trips:
                if len(t) <= max_trip_len:
                    trips_.append(t)
                    continue

                prefix = t[:max_trip_len - 1]
                for d in t[max_trip_len - 1:]:
                    trips_.append(prefix + [d])

            trips = trips_
            

    def get_block_size(fmt: str) -> int:
        return max(len(t) for t in trips) * VehicleStopTime.format_length(fmt)

    def trip_to_tokens(trip: list[VehicleStopTime], fmt: str) -> list[int]:
        seq = []
        for st in trip:
            seq.extend(st.format(fmt))

        return seq

    def pad(trip: list[int], block_size: int) -> tuple[list[int], int]:
        missing = block_size - len(trip)
        return [PLACEHOLDER] * missing + trip, missing

    print(f'Loaded {len(trips)} trip instances with a max of {max(map(len, trips))} stoptimes', file=sys.stderr)
    print(f'{len(trips) = }', file=sys.stderr)

    return DatasetInfo(
        trips,
        get_block_size,
        trip_to_tokens,
        pad,
        vocab_size=BASE_TOKENS + len(codes),
        indirect=indirect
    )

class StoptimesDataset(Dataset):
    def __init__(self, dataset: DatasetInfo, e_fmt: str, r_fmt: str):
        self.dataset = dataset
        self.fmt = e_fmt + r_fmt
        self.fmt_len = VehicleStopTime.format_length(self.fmt)
        self.e_fmt = e_fmt
        self.e_fmt_len = VehicleStopTime.format_length(e_fmt)
        self.block_size = dataset.get_block_size(self.fmt)

    def get_vocab_size(self):
        return self.dataset.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        trip = self.dataset[index]

        seq = self.dataset.trip_to_tokens(trip, self.fmt)

        seq, npadded = self.dataset.pad(seq, self.block_size)

        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:],  dtype=torch.long)

        # mask out the loss at the input locations
        y[:npadded+self.fmt_len] = -1
        for i in range(npadded - 1, len(y) - self.e_fmt_len + 1, self.fmt_len):
            y[i:i+self.e_fmt_len] = -1

        return x, y
