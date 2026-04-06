import random
from models import HospitalAction, HospitalObservation, HospitalState

class HospitalEnv:
    
    def __init__(self, difficulty="medium"):
        self.difficulty = difficulty
        self.reset()
    
    def reset(self):
        # Task difficulty levels
        if self.difficulty == "easy":
            self.available_beds = 3
            self.patients = [random.randint(1, 5) for _ in range(3)]
        elif self.difficulty == "medium":
            self.available_beds = 3
            self.patients = [random.randint(1, 5) for _ in range(5)]
        else:  # hard
            self.available_beds = 2
            self.patients = [random.randint(1, 5) for _ in range(10)]
        
        self.step_count = 0
        
        return HospitalObservation(
            available_beds=self.available_beds,
            patients_waiting=len(self.patients),
            message="New episode started"
        )
    
    def step(self, action: HospitalAction):
        self.step_count += 1
        
        if self.available_beds <= 0:
            reward = -1
            done = True
        else:
            severity = self.patients.pop(0)
            self.available_beds -= 1
            
            # Improved reward logic
            if severity >= 4:
                reward = 1.0
            elif severity >= 2:
                reward = 0.7
            else:
                reward = 0.3
            
            done = len(self.patients) == 0
        
        observation = HospitalObservation(
            available_beds=self.available_beds,
            patients_waiting=len(self.patients),
            message="Step executed"
        )
        
        return observation, reward, done
    
    @property
    def state(self):
        return HospitalState(step_count=self.step_count)
