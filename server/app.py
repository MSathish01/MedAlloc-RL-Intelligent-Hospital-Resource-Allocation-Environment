from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

# -------------------------------
# Dummy Environment State
# -------------------------------
state_data = {
    "beds": 10,
    "patients": 5,
    "step": 0
}

# -------------------------------
# Home Route
# -------------------------------
@app.get("/")
def home():
    return {"message": "Hospital Environment Running"}

# -------------------------------
# Web UI (IMPORTANT for HF)
# -------------------------------
@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return """
    <html>
        <head><title>Hospital Env</title></head>
        <body>
            <h1>🏥 Hospital Resource Allocation</h1>
            <p>✅ App is working!</p>
            <ul>
                <li><a href="/docs">API Docs</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
    </html>
    """

# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

# -------------------------------
# Reset Environment
# -------------------------------
@app.post("/reset")
def reset():
    global state_data
    state_data = {
        "beds": 10,
        "patients": 5,
        "step": 0
    }
    return {
        "observation": state_data,
        "reward": 0,
        "done": False
    }

# -------------------------------
# Action Model
# -------------------------------
class Action(BaseModel):
    allocate: int

# -------------------------------
# Step Function
# -------------------------------
@app.post("/step")
def step(action: Action):
    global state_data

    state_data["step"] += 1
    state_data["beds"] -= action.allocate
    state_data["patients"] -= action.allocate

    reward = action.allocate

    done = state_data["step"] >= 5

    return {
        "observation": state_data,
        "reward": reward,
        "done": done
    }

# -------------------------------
# Get State
# -------------------------------
@app.get("/state")
def state():
    return state_data