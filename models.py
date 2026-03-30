from dataclasses import dataclass

@dataclass
class HospitalAction:
    patient_id: int

@dataclass
class HospitalObservation:
    available_beds: int
    patients_waiting: int
    message: str

@dataclass
class HospitalState:
    step_count: int
