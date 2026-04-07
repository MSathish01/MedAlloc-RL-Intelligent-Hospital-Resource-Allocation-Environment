from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import random

app = FastAPI(title="MedAlloc-RL")

# -------------------------------
# GLOBAL STATE
# -------------------------------
state_data = {}

TASK_CONFIG = {
    "easy":   {"beds": 10, "patients": 5,  "max_steps": 5},
    "medium": {"beds": 8,  "patients": 8,  "max_steps": 5},
    "hard":   {"beds": 5,  "patients": 10, "max_steps": 5},
}

# -------------------------------
# HELPERS
# -------------------------------
def make_patients(n: int, start_id: int = 0) -> List[dict]:
    return [
        {
            "id": start_id + i,
            "severity": random.choice(["low", "medium", "high"]),
            "emergency": False,
            "waiting_steps": 0,
        }
        for i in range(n)
    ]

def clamp_score(x):
    EPS = 1e-6
    return max(EPS, min(1 - EPS, x))

def clean_obs(s):
    return {
        "beds": s["beds"],
        "total_beds": s["total_beds"],
        "patients": s["patients"],
        "step": s["step"],
        "max_steps": s["max_steps"],
        "difficulty": s["difficulty"],
    }

# -------------------------------
# RESET
# -------------------------------
@app.post("/reset")
def reset(task: str = Query(default="easy")):
    global state_data

    cfg = TASK_CONFIG.get(task, TASK_CONFIG["easy"])

    state_data = {
        "beds": cfg["beds"],
        "total_beds": cfg["beds"],
        "patients": make_patients(cfg["patients"]),
        "step": 0,
        "max_steps": cfg["max_steps"],
        "difficulty": task,
        "total_reward": 0.0,
        "patient_id_counter": cfg["patients"],
        "treated_count": 0,
    }

    return {
        "observation": clean_obs(state_data),
        "reward": 0.0,
        "done": False
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

    if not state_data:
        return {"error": "Call /reset first"}

    beds = state_data["beds"]
    patients = state_data["patients"]

    # Priority sort
    def priority(p):
        return (
            not p["emergency"],
            {"high": 0, "medium": 1, "low": 2}[p["severity"]],
            -p["waiting_steps"]
        )

    patients = sorted(patients, key=priority)

    allocate = min(action.allocate, beds, len(patients))
    treated = patients[:allocate]
    remaining = patients[allocate:]

    reward = 0.0

    # Reward
    for p in treated:
        reward += {"high": 3, "medium": 2, "low": 1}[p["severity"]]
        if p["emergency"]:
            reward += 1
        state_data["treated_count"] += 1

    # Penalty
    for p in remaining:
        reward -= {"high": 2, "medium": 0.5, "low": 0}[p["severity"]]
        if p["emergency"]:
            reward -= 1.5
        p["waiting_steps"] += 1

    # Deterioration
    for p in remaining:
        if p["waiting_steps"] >= 2 and p["severity"] == "medium":
            p["severity"] = "high"
            reward -= 0.5
        elif p["waiting_steps"] >= 3 and p["severity"] == "low":
            p["severity"] = "medium"

    # Restore beds
    state_data["beds"] = state_data["total_beds"]

    # New patients
    new = make_patients(random.randint(0, 2), state_data["patient_id_counter"])
    state_data["patient_id_counter"] += len(new)

    if random.random() < 0.2:
        new.append({
            "id": state_data["patient_id_counter"],
            "severity": "high",
            "emergency": True,
            "waiting_steps": 0,
        })
        state_data["patient_id_counter"] += 1

    state_data["patients"] = remaining + new
    state_data["step"] += 1
    state_data["total_reward"] += reward

    done = state_data["step"] >= state_data["max_steps"]

    return {
        "observation": clean_obs(state_data),
        "reward": float(round(reward, 2)),
        "done": done
    }

# -------------------------------
# GRADER
# -------------------------------
@app.get("/grade")
def grade():
    if not state_data:
        return {"score": 0.000001}

    max_possible = state_data["total_beds"] * 3 * state_data["max_steps"]
    raw_score = state_data["total_reward"] / max_possible
    score = clamp_score(raw_score)

    return {
        "score": float(round(score, 6))
    }

# -------------------------------
# HEALTH
# -------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

# -------------------------------
# HOME (UI)
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>🏥 MedAlloc-RL Running</h1>
    <p>Hospital Resource Allocation Environment</p>
    <a href='/docs'>Open API Docs</a>
    """

# -------------------------------
# ENTRY POINT
# -------------------------------
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()