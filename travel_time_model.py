from dataset import VehicleStopTime

class TravelTimeModel:
    name: str = "travel_time"

    def predict_one(self, trip: list[VehicleStopTime], pred: list[VehicleStopTime], next_st: VehicleStopTime) -> VehicleStopTime:
        trip = trip + pred
        last = trip[-1]
        ett = next_st.eat - last.eat
        eat = last.rat + ett

        return VehicleStopTime(eat, next_st.eat, next_st.code)
        
    def predict(self, batch: list[tuple[list[VehicleStopTime], list[VehicleStopTime], VehicleStopTime]]) -> list[VehicleStopTime]:
        return [ self.predict_one(*inp) for inp in batch ]
