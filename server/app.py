from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import random

app = FastAPI(title="MedAlloc-RL", description="Hospital Resource Allocation RL Environment")

# -------------------------------
# GLOBAL STATE
# -------------------------------
state_data = {}

# -------------------------------
# TASK CONFIG
# -------------------------------
TASK_CONFIG = {
    "easy":   {"beds": 10, "patients": 5,  "max_steps": 5},
    "medium": {"beds": 8,  "patients": 8,  "max_steps": 5},
    "hard":   {"beds": 5,  "patients": 10, "max_steps": 5},
}

# -------------------------------
# HELPERS
# -------------------------------
def make_patients(n: int, start_id: int = 0) -> List[dict]:
    severities = ["low", "medium", "high"]
    return [
        {
            "id": start_id + i,
            "severity": random.choice(severities),
            "emergency": False,
            "waiting_steps": 0,
        }
        for i in range(n)
    ]

def severity_score(severity: str) -> int:
    return {"high": 2, "medium": 1, "low": 0}[severity]

# -------------------------------
# RESET
# -------------------------------
@app.post("/reset")
def reset(task: str = Query(default="easy", description="Task difficulty: easy, medium, hard")):
    global state_data

    if task not in TASK_CONFIG:
        task = "easy"

    cfg = TASK_CONFIG[task]

    state_data = {
        "beds":         cfg["beds"],
        "total_beds":   cfg["beds"],
        "patients":     make_patients(cfg["patients"]),
        "step":         0,
        "max_steps":    cfg["max_steps"],
        "difficulty":   task,
        "total_reward": 0.0,
        "patient_id_counter": cfg["patients"],
        "treated_count": 0,
        "emergency_count": 0,
    }

    return {
        "observation": _clean_obs(state_data),
        "reward": 0.0,
        "done": False,
        "task": task,
        "info": {
            "total_beds": cfg["beds"],
            "initial_patients": cfg["patients"],
            "max_steps": cfg["max_steps"],
        }
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

    beds     = state_data["beds"]
    patients = state_data["patients"]

    # Sort patients: emergency first, then by severity, then by waiting time
    def priority_key(p):
        sev = {"high": 0, "medium": 1, "low": 2}[p["severity"]]
        emergency_bonus = -10 if p.get("emergency") else 0
        return emergency_bonus + sev - p.get("waiting_steps", 0) * 0.1

    patients_sorted = sorted(patients, key=priority_key)

    allocate  = max(0, min(action.allocate, beds, len(patients_sorted)))
    treated   = patients_sorted[:allocate]
    remaining = patients_sorted[allocate:]

    reward = 0.0

    # Reward: treated patients
    for p in treated:
        if p["severity"] == "high":
            reward += 3.0
            if p.get("emergency"):
                reward += 1.0   # bonus for treating emergencies
        elif p["severity"] == "medium":
            reward += 2.0
        else:
            reward += 1.0
        state_data["treated_count"] += 1

    # Penalty: untreated patients
    for p in remaining:
        if p["severity"] == "high":
            reward -= 2.0
        elif p["severity"] == "medium":
            reward -= 0.5
        # Emergency untreated is worse
        if p.get("emergency"):
            reward -= 1.5

    # Penalty: wasting beds when patients are waiting
    unused = beds - allocate
    if len(patients_sorted) > 0:
        reward -= unused * 0.3

    # Increment waiting steps for remaining patients
    for p in remaining:
        p["waiting_steps"] = p.get("waiting_steps", 0) + 1

    # Patient deterioration: waiting patients get worse
    for p in remaining:
        if p["waiting_steps"] >= 2 and p["severity"] == "medium":
            p["severity"] = "high"
            reward -= 0.5   # penalty for allowing deterioration
        elif p["waiting_steps"] >= 3 and p["severity"] == "low":
            p["severity"] = "medium"

    # Beds restore each step (treated patients discharged)
    state_data["beds"] = state_data["total_beds"]
    state_data["step"] += 1
    state_data["total_reward"] += reward

    # Dynamic new arrivals (0-2 regular patients)
    counter = state_data["patient_id_counter"]
    new_arrivals = make_patients(random.randint(0, 2), start_id=counter)
    state_data["patient_id_counter"] = counter + len(new_arrivals)

    # Emergency arrival: 20% chance each step
    if random.random() < 0.2:
        emergency_patient = {
            "id": state_data["patient_id_counter"],
            "severity": "high",
            "emergency": True,
            "waiting_steps": 0,
        }
        new_arrivals.append(emergency_patient)
        state_data["patient_id_counter"] += 1
        state_data["emergency_count"] += 1

    state_data["patients"] = remaining + new_arrivals

    # Done condition
    done = (
        state_data["step"] >= state_data["max_steps"]
        or len(state_data["patients"]) == 0
    )

# Normalized score 0.0 → 1.0
    max_possible = max(1.0, state_data["total_beds"] * 3.0)
    score = round(max(0.0, min(1.0, state_data["total_reward"] / max_possible)), 3)
    return { }

# -------------------------------
# GRADER  (judges call this)
# -------------------------------
@app.get("/grade")
def grade():
    """Return normalized score for the current episode."""
    if not state_data:
        return {"score": 0.0, "error": "No episode in progress. Call /reset first."}
    max_possible = state_data["total_beds"] * 3.0 * state_data["max_steps"]
    score = round(max(0.0, min(1.0, state_data["total_reward"] / max_possible)), 3)
    return {
        "score":          score,
        "total_reward":   round(state_data["total_reward"], 2),
        "treated_count":  state_data.get("treated_count", 0),
        "steps_taken":    state_data["step"],
        "difficulty":     state_data["difficulty"],
    }

# -------------------------------
# STATE
# -------------------------------
@app.get("/state")
def get_state():
    if not state_data:
        return {"error": "No active episode. Call /reset first."}
    return _clean_obs(state_data)

# -------------------------------
# HEALTH
# -------------------------------
@app.get("/health")
def health():
    return {"status": "healthy", "environment": "MedAlloc-RL", "version": "2.0.0"}

# -------------------------------
# HOME
# -------------------------------
@app.get("/")
def home():
    return {
        "message": "MedAlloc-RL Hospital Resource Allocation Environment",
        "version": "2.0.0",
        "endpoints": {
            "reset":  "POST /reset?task=easy|medium|hard",
            "step":   "POST /step  {allocate: int}",
            "state":  "GET  /state",
            "grade":  "GET  /grade",
            "health": "GET  /health",
            "docs":   "GET  /docs",
        }
    }

# -------------------------------
# WEB UI
# -------------------------------
@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return """
    <html>
      <head>
        <title>MedAlloc-RL</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; }
          h1 { color: #1a73e8; }
          .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 13px; margin: 2px; }
          .high   { background: #fde8e8; color: #c0392b; }
          .medium { background: #fef3cd; color: #856404; }
          .low    { background: #d4edda; color: #155724; }
          a { color: #1a73e8; }
        </style>
      </head>
      <body>
        <h1>ðŸ¥ MedAlloc-RL</h1>
        <p>Hospital Resource Allocation â€” Reinforcement Learning Environment</p>
        <h3>Features</h3>
        <ul>
          <li>ðŸš‘ Priority-based patient triage (low / medium / high)</li>
          <li>âš¡ Emergency patient arrivals (20% per step)</li>
          <li>ðŸ“ˆ Patient deterioration over time</li>
          <li>â± Time pressure with step limits</li>
          <li>ðŸ“Š Normalized scoring 0.0 â†’ 1.0</li>
          <li>ðŸŽ¯ 3 difficulty levels: easy / medium / hard</li>
        </ul>
        <h3>Quick Links</h3>
        <ul>
          <li><a href="/docs">ðŸ“– API Docs (Swagger)</a></li>
          <li><a href="/health">ðŸ’š Health Check</a></li>
          <li><a href="/state">ðŸ“‹ Current State</a></li>
          <li><a href="/grade">ðŸ† Current Grade</a></li>
        </ul>
        <h3>Reward System</h3>
        <span class="badge high">High severity treated: +3.0</span>
        <span class="badge medium">Medium severity treated: +2.0</span>
        <span class="badge low">Low severity treated: +1.0</span>
        <br><br>
        <span class="badge high">High untreated: -2.0</span>
        <span class="badge high">Emergency untreated: -1.5 extra</span>
        <span class="badge medium">Wasted bed: -0.3 each</span>
      </body>
    </html>
    """

# -------------------------------
# INTERNAL HELPER
# -------------------------------
def _clean_obs(s: dict) -> dict:
    """Return only the fields agents need to see."""
    return {
        "beds":       s["beds"],
        "total_beds": s["total_beds"],
        "patients":   s["patients"],
        "step":       s["step"],
        "max_steps":  s["max_steps"],
        "difficulty": s["difficulty"],
    }

# -------------------------------
# ENTRY POINT
# -------------------------------
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
