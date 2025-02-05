import sys
from dataset import VehicleStopTime, all_trips, ids, timestamp
from datetime import timedelta
from collections import defaultdict
from tqdm import tqdm  # type: ignore[import-untyped]
from zoneinfo import ZoneInfo

trips_by_stop_date = defaultdict(list)
for id, t in zip(ids, all_trips):
    for st in t:
        trips_by_stop_date[st.code, st.rrat.replace(microsecond=0, second=0, minute=0) ].append((id, t))

def find_earlier_arrivals(ost: VehicleStopTime):
    return [
        (id, t) for id, t in trips_by_stop_date[ost.code, ost.rrat.replace(microsecond=0, second=0, minute=0)] + trips_by_stop_date[ost.code, (ost.rrat - timedelta(hours=1)).replace(microsecond=0, second=0, minute=0)]
        if all(
            st.code != ost.code or
            st.rrat < ost.rrat
            and ost.rrat - st.rrat < timedelta(minutes=30)
            for st in t
        )
    ]

for t in tqdm(all_trips, file=sys.stderr):
    for st in t:
        st.earlier = find_earlier_arrivals(st)

print('real_arrival_time', 'trip_instance_id', 'expected_arrival_time', 'code', 'earlier', sep=',')
for id, trip in zip(ids, all_trips):
    for st in trip:
        earlier_str = ';'.join(id for id, _ in st.earlier)  # type: ignore[misc]
        rat_str = st.rrat.astimezone(ZoneInfo("UTC")).replace(tzinfo=None).isoformat(' ') + "+00"
        print(rat_str, id, timestamp(st.eat), st.code, earlier_str, sep=',')
