from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import random

app = FastAPI()


state_data = {}


@app.post("/reset")
def reset():
    global state_data

    difficulty = random.choice(["easy", "medium", "hard"])

    if difficulty == "easy":
        beds = 10
        num_patients = 5
    elif difficulty == "medium":
        beds = 8
        num_patients = 8
    else:
        beds = 5
        num_patients = 10

    # Create patients with severity
    patients = []
    for i in range(num_patients):
        severity = random.choice(["low", "medium", "high"])
        patients.append({
            "id": i,
            "severity": severity
        })

    state_data = {
        "beds": beds,
        "patients": patients,
        "step": 0,
        "time": 5,
        "difficulty": difficulty
    }

    return {
        "observation": state_data,
        "reward": 0.0,
        "done": False,
        "task": difficulty
    }

# -------------------------------
# ACTION MODEL
# -------------------------------
class Action(BaseModel):
    allocate: int

# -------------------------------
# STEP
# -------------------------------
@app.post("/step")
def step(action: Action):
    global state_data

    beds = state_data["beds"]
    patients = state_data["patients"]

    allocate = min(action.allocate, beds)
    treated = patients[:allocate]
    remaining = patients[allocate:]

    reward = 0

    # -------------------------------
    # REWARD BASED ON SEVERITY
    # -------------------------------
    for p in treated:
        if p["severity"] == "high":
            reward += 3
        elif p["severity"] == "medium":
            reward += 2
        else:
            reward += 1

    # -------------------------------
    # PENALTIES
    # -------------------------------
    unused_beds = beds - allocate
    reward -= unused_beds * 0.5

    reward -= len(remaining) * 1.5

    # -------------------------------
    # TIME PRESSURE
    # -------------------------------
    state_data["time"] -= 1
    if state_data["time"] <= 0:
        reward -= 5

    # -------------------------------
    # UPDATE STATE
    # -------------------------------
    state_data["beds"] -= allocate
    state_data["patients"] = remaining
    state_data["step"] += 1

    # Dynamic new patients
    new_patients = random.randint(0, 2)
    for _ in range(new_patients):
        state_data["patients"].append({
            "id": random.randint(100, 999),
            "severity": random.choice(["low", "medium", "high"])
        })

    # -------------------------------
    # DONE CONDITION
    # -------------------------------
    done = (
        state_data["step"] >= 5
        or len(state_data["patients"]) == 0
        or state_data["beds"] <= 0
    )

    # -------------------------------
    # GRADER (0 → 1 score)
    # -------------------------------
    score = max(0.0, min(1.0, reward / 20.0))

    return {
        "observation": state_data,
        "reward": reward,
        "score": score,
        "done": done
    }

# -------------------------------
# STATE
# -------------------------------
@app.get("/state")
def state():
    return state_data

# -------------------------------
# HEALTH
# -------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

# -------------------------------
# HOME
# -------------------------------
@app.get("/")
def home():
    return {"message": "Hospital Environment Running"}

# -------------------------------
# WEB UI
# -------------------------------
@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return """
    <html>
        <head><title>Hospital Env</title></head>
        <body>
            <h1>🏥 Hospital Resource Allocation </h1>
            <p>🚑 Patients have priority levels</p>
            <p>⏱ Time pressure + dynamic arrivals</p>
            <p>📊 Reward based on efficiency</p>
            <ul>
                <li><a href="/docs">API Docs</a></li>
                <li><a href="/health">Health</a></li>
            </ul>
        </body>
    </html>
    """