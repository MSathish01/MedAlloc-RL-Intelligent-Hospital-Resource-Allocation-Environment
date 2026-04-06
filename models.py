from pydantic import BaseModel, Field
from typing import List, Optional


class Patient(BaseModel):
    id: int = Field(..., description="Unique patient identifier")
    severity: str = Field(..., description="Patient severity: low, medium, or high")
    emergency: bool = Field(default=False, description="True if this is an emergency arrival")
    waiting_steps: int = Field(default=0, description="Number of steps this patient has been waiting")


class HospitalObservation(BaseModel):
    beds: int = Field(..., description="Currently available beds this step")
    total_beds: int = Field(..., description="Total bed capacity of the hospital")
    patients: List[Patient] = Field(..., description="List of patients currently waiting")
    step: int = Field(..., description="Current step number in the episode")
    max_steps: int = Field(..., description="Maximum steps before episode ends")
    difficulty: str = Field(..., description="Task difficulty: easy, medium, or hard")


class HospitalAction(BaseModel):
    allocate: int = Field(..., ge=0, description="Number of beds to allocate this step")


class StepInfo(BaseModel):
    step: int
    treated_total: int
    emergencies_seen: int
    total_reward: float


class StepResult(BaseModel):
    observation: HospitalObservation
    reward: float = Field(..., description="Reward earned this step")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score 0.0 to 1.0")
    done: bool = Field(..., description="True if the episode has ended")
    info: StepInfo


class ResetResult(BaseModel):
    observation: HospitalObservation
    reward: float = 0.0
    done: bool = False
    task: str


class GradeResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Final normalized score 0.0 to 1.0")
    total_reward: float
    treated_count: int
    steps_taken: int
    difficulty: str


class HealthResult(BaseModel):
    status: str
    environment: str
    version: str
