from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List
import random

app = FastAPI(title="MedAlloc-RL", description="Hospital Resource Allocation RL Environment")

state_data = {}

TASK_CONFIG = {
    "easy":   {"beds": 10, "patients": 5,  "max_steps": 5},
    "medium": {"beds": 8,  "patients": 8,  "max_steps": 5},
    "hard":   {"beds": 5,  "patients": 10, "max_steps": 5},
}

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

def safe_score(raw: float) -> float:
    """Always return strictly between 0 and 1."""
    score = round(max(0.001, min(0.999, raw)), 3)
    return score

@app.post("/reset")
def reset(task: str = Query(default="easy")):
    global state_data
    if task not in TASK_CONFIG:
        task = "easy"
    cfg = TASK_CONFIG[task]
    state_data = {
        "beds":               cfg["beds"],
        "total_beds":         cfg["beds"],
        "patients":           make_patients(cfg["patients"]),
        "step":               0,
        "max_steps":          cfg["max_steps"],
        "difficulty":         task,
        "total_reward":       0.0,
        "patient_id_counter": cfg["patients"],
        "treated_count":      0,
        "emergency_count":    0,
    }
    return {
        "observation": _clean_obs(state_data),
        "reward": 0.0,
        "done": False,
        "task": task,
    }

class Action(BaseModel):
    allocate: int

@app.post("/step")
def step(action: Action):
    global state_data
    if not state_data:
        return {"error": "Call /reset first"}

    beds     = state_data["beds"]
    patients = state_data["patients"]

    def priority_key(p):
        sev = {"high": 0, "medium": 1, "low": 2}[p["severity"]]
        emergency_bonus = -10 if p.get("emergency") else 0
        return emergency_bonus + sev - p.get("waiting_steps", 0) * 0.1

    patients_sorted = sorted(patients, key=priority_key)
    allocate  = max(0, min(action.allocate, beds, len(patients_sorted)))
    treated   = patients_sorted[:allocate]
    remaining = patients_sorted[allocate:]

    reward = 0.0
    for p in treated:
        if p["severity"] == "high":
            reward += 3.0
            if p.get("emergency"):
                reward += 1.0
        elif p["severity"] == "medium":
            reward += 2.0
        else:
            reward += 1.0
        state_data["treated_count"] += 1

    for p in remaining:
        if p["severity"] == "high":
            reward -= 2.0
        elif p["severity"] == "medium":
            reward -= 0.5
        if p.get("emergency"):
            reward -= 1.5

    unused = beds - allocate
    if len(patients_sorted) > 0:
        reward -= unused * 0.3

    for p in remaining:
        p["waiting_steps"] = p.get("waiting_steps", 0) + 1

    for p in remaining:
        if p["waiting_steps"] >= 2 and p["severity"] == "medium":
            p["severity"] = "high"
            reward -= 0.5
        elif p["waiting_steps"] >= 3 and p["severity"] == "low":
            p["severity"] = "medium"

    state_data["beds"] = state_data["total_beds"]
    state_data["step"] += 1
    state_data["total_reward"] += reward

    counter     = state_data["patient_id_counter"]
    new_arrivals = make_patients(random.randint(0, 2), start_id=counter)
    state_data["patient_id_counter"] = counter + len(new_arrivals)

    if random.random() < 0.2:
        state_data["patients"].append({
            "id":            state_data["patient_id_counter"],
            "severity":      "high",
            "emergency":     True,
            "waiting_steps": 0,
        })
        state_data["patient_id_counter"] += 1
        state_data["emergency_count"]    += 1

    state_data["patients"] = remaining + new_arrivals

    done = (
        state_data["step"] >= state_data["max_steps"]
        or len(state_data["patients"]) == 0
    )

    max_possible = max(1.0, state_data["total_beds"] * 3.0)
    score = safe_score(state_data["total_reward"] / max_possible)

    return {
        "observation": _clean_obs(state_data),
        "reward":      round(reward, 2),
        "score":       score,
        "done":        done,
        "info": {
            "step":             state_data["step"],
            "treated_total":    state_data["treated_count"],
            "emergencies_seen": state_data["emergency_count"],
            "total_reward":     round(state_data["total_reward"], 2),
        }
    }

@app.get("/grade")
def grade():
    if not state_data:
        return {"score": 0.5}
    max_possible = max(1.0, state_data["total_beds"] * 3.0)
    score = safe_score(state_data["total_reward"] / max_possible)
    return {
        "score":         score,
        "total_reward":  round(state_data["total_reward"], 2),
        "treated_count": state_data.get("treated_count", 0),
        "steps_taken":   state_data["step"],
        "difficulty":    state_data["difficulty"],
    }

@app.get("/state")
def get_state():
    if not state_data:
        return {"error": "No active episode. Call /reset first."}
    return _clean_obs(state_data)

@app.get("/health")
def health():
    return {"status": "healthy", "environment": "MedAlloc-RL", "version": "2.0.3"}

@app.get("/api")
def api_index():
    return {
        "message": "MedAlloc-RL Hospital Resource Allocation Environment",
        "version": "2.0.3",
        "endpoints": {
            "reset": "POST /reset?task=easy|medium|hard",
            "step": "POST /step  {allocate: int}",
            "state": "GET  /state",
            "grade": "GET  /grade",
            "health": "GET  /health",
            "docs": "GET /docs",
        },
    }


@app.get("/doc", include_in_schema=False)
def doc_redirect():
    return RedirectResponse(url="/docs")

def _interactive_html() -> str:
    return """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>MedAlloc-RL | AI Hospital Commander</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-color: #0b1120;
                --surface-color: rgba(30, 41, 59, 0.6);
                --border-color: rgba(255, 255, 255, 0.08);
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --accent-glow: rgba(56, 189, 248, 0.4);
                --accent-solid: #38bdf8;
                --accent-hover: #0ea5e9;
                
                --sev-high: #ef4444;
                --sev-med: #f59e0b;
                --sev-low: #10b981;
            }

            body {
                font-family: 'Inter', sans-serif;
                background: radial-gradient(circle at top left, #1e1b4b, #0f172a 80%);
                color: var(--text-primary);
                margin: 0;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                padding: 2rem;
                box-sizing: border-box;
            }

            .header {
                max-width: 1400px;
                margin: 0 auto 2rem auto;
                width: 100%;
                text-align: center;
            }

            h1 {
                font-size: 2.75rem;
                font-weight: 800;
                background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0 0 0.5rem 0;
                letter-spacing: -1px;
            }

            .header-sub {
                color: var(--text-secondary);
                font-size: 1.1rem;
                font-weight: 500;
            }

            .glass-panel {
                background: var(--surface-color);
                backdrop-filter: blur(16px);
                border: 1px solid var(--border-color);
                border-radius: 20px;
                padding: 1.75rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }

            .dashboard-grid {
                display: grid;
                grid-template-columns: 1fr 400px;
                gap: 2rem;
                max-width: 1400px;
                margin: 0 auto;
                width: 100%;
            }

            .controls-wrapper {
                display: flex;
                gap: 1.5rem;
                align-items: flex-end;
                margin-bottom: 2rem;
                flex-wrap: wrap;
                background: rgba(15, 23, 42, 0.4);
                padding: 1.5rem;
                border-radius: 16px;
                border: 1px solid var(--border-color);
            }

            .control-group {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }

            label {
                font-size: 0.8rem;
                font-weight: 600;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            select, input {
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid var(--border-color);
                color: white;
                padding: 0.75rem 1rem;
                border-radius: 10px;
                font-family: inherit;
                font-size: 1rem;
                outline: none;
                transition: all 0.2s;
            }

            select:focus, input:focus {
                border-color: var(--accent-solid);
                box-shadow: 0 0 0 3px var(--accent-glow);
            }

            button {
                background: var(--accent-solid);
                color: #0f172a;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 10px;
                font-weight: 700;
                font-size: 1rem;
                cursor: pointer;
                transition: all 0.2s;
                box-shadow: 0 4px 14px var(--accent-glow);
                font-family: inherit;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            button:hover:not(:disabled) {
                background: var(--accent-hover);
                transform: translateY(-2px);
                box-shadow: 0 6px 20px var(--accent-glow);
                color: white;
            }

            button:disabled {
                background: rgba(71, 85, 105, 0.5);
                color: #94a3b8;
                cursor: not-allowed;
                box-shadow: none;
            }

            #resetBtn { background: rgba(255, 255, 255, 0.1); color: white; box-shadow: none; }
            #resetBtn:hover:not(:disabled) { background: rgba(255, 255, 255, 0.2); }
            
            #useRecBtn { padding: 0.5rem 1rem; font-size: 0.85rem; background: rgba(56, 189, 248, 0.2); color: var(--accent-solid); box-shadow: none; }
            #useRecBtn:hover { background: rgba(56, 189, 248, 0.3); }

            .rec-box {
                display: flex;
                align-items: center;
                gap: 1rem;
                background: rgba(15, 23, 42, 0.8);
                border: 1px dashed var(--accent-solid);
                padding: 0.5rem 1rem;
                border-radius: 10px;
                height: 44px;
                box-sizing: border-box;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }

            .stat-card {
                background: rgba(15, 23, 42, 0.5);
                border: 1px solid rgba(255,255,255,0.05);
                padding: 1.25rem 1rem;
                border-radius: 14px;
                text-align: center;
                transition: transform 0.2s;
            }
            
            .stat-card:hover {
                transform: translateY(-3px);
                background: rgba(30, 41, 59, 0.8);
            }

            .stat-label { font-size: 0.75rem; text-transform: uppercase; color: var(--text-secondary); font-weight: 600; letter-spacing: 0.5px; }
            .stat-value { font-size: 1.75rem; font-weight: 800; margin-top: 0.5rem; color: white; }
            
            .stat-value.highlight { color: var(--accent-solid); }

            .section-title {
                font-size: 1.25rem;
                font-weight: 700;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .section-title::before {
                content: '';
                display: block;
                width: 4px;
                height: 18px;
                background: var(--accent-solid);
                border-radius: 2px;
            }

            .patients-table-container {
                background: rgba(15, 23, 42, 0.5);
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.05);
                overflow: hidden;
            }

            table { width: 100%; border-collapse: collapse; }
            th { text-align: left; padding: 1rem; color: var(--text-secondary); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; background: rgba(0,0,0,0.2); }
            td { padding: 1rem; border-top: 1px solid rgba(255,255,255,0.03); font-size: 0.95rem; }

            tr { transition: background 0.2s; }
            tr:hover td { background: rgba(255,255,255,0.03); }

            .badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.25rem 0.75rem;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .badge.high { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
            .badge.medium { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }
            .badge.low { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
            .badge.emergency { background: #ef4444; color: white; animation: pulse 1.5s infinite; box-shadow: 0 0 10px rgba(239, 68, 68, 0.5); border: none; }

            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }

            .emergency-row td {
                background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, transparent 100%) !important;
            }

            pre {
                background: #020617 !important;
                color: #38bdf8;
                padding: 1.25rem;
                border-radius: 12px;
                overflow: auto;
                font-size: 0.85rem;
                border: 1px solid rgba(255,255,255,0.05);
                max-height: 500px;
                margin: 0;
            }

            #banner {
                display: none;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                background: rgba(239, 68, 68, 0.15);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #fca5a5;
                font-weight: 600;
                text-align: center;
                animation: slideDown 0.3s ease;
            }
            
            #banner.success {
                background: rgba(16, 185, 129, 0.15);
                border-color: rgba(16, 185, 129, 0.3);
                color: #6ee7b7;
            }

            @keyframes slideDown {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            details {
                margin-top: 1.5rem;
                background: rgba(15, 23, 42, 0.4);
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 1rem;
            }
            summary {
                cursor: pointer;
                font-weight: 600;
                color: var(--text-secondary);
                outline: none;
            }
            summary:hover { color: white; }

            @media (max-width: 1024px) {
                .dashboard-grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>MedAlloc-RL Commander</h1>
            <div class="header-sub">Intelligent Hospital Resource Allocation Dashboard</div>
        </div>

        <div id="banner"></div>

        <div class="dashboard-grid">
            <div class="main-content">
                <div class="glass-panel" style="margin-bottom: 2rem;">
                    <div class="controls-wrapper">
                        <div class="control-group">
                            <label for="task">Difficulty</label>
                            <select id="task">
                                <option value="easy">Easy</option>
                                <option value="medium" selected>Medium</option>
                                <option value="hard">Hard</option>
                            </select>
                        </div>

                        <div class="control-group">
                            <label>&nbsp;</label>
                            <button id="resetBtn">Initialize</button>
                        </div>

                        <div style="width: 2px; background: var(--border-color); margin: 0 1rem; align-self: stretch;"></div>

                        <div class="control-group">
                            <label for="allocate">Allocate Beds</label>
                            <input id="allocate" type="number" min="0" value="0" />
                        </div>

                        <div class="control-group">
                            <label>AI Suggestion</label>
                            <div class="rec-box">
                                <span id="recText" style="font-weight: 700; font-size: 1.2rem; color: #f8fafc;">—</span>
                                <button id="useRecBtn">USE</button>
                            </div>
                        </div>

                        <div class="control-group" style="margin-left: auto;">
                            <label>&nbsp;</label>
                            <button id="stepBtn" style="min-width: 120px;">Execute Step</button>
                        </div>
                    </div>

                    <div id="summary" class="stats-grid"></div>

                    <div class="section-title">Patients Waiting Room</div>
                    <div class="patients-table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Patient ID</th>
                                    <th>Severity</th>
                                    <th>Status</th>
                                    <th>Wait Time</th>
                                </tr>
                            </thead>
                            <tbody id="patients"></tbody>
                        </table>
                    </div>

                    <details>
                        <summary>View Raw State JSON</summary>
                        <pre id="stateRaw" style="margin-top: 1rem;"></pre>
                    </details>
                </div>
            </div>

            <div class="sidebar">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <div class="section-title">System Log</div>
                    <div style="flex: 1; display: flex; flex-direction: column;">
                        <pre id="last" style="flex: 1; height: 100%; min-height: 500px; margin: 0;"></pre>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const bannerEl = document.getElementById('banner');
            const summaryEl = document.getElementById('summary');
            const patientsEl = document.getElementById('patients');
            const stateRawEl = document.getElementById('stateRaw');
            const lastEl = document.getElementById('last');
            const allocateEl = document.getElementById('allocate');
            const taskEl = document.getElementById('task');
            const stepBtn = document.getElementById('stepBtn');
            const resetBtn = document.getElementById('resetBtn');
            const recTextEl = document.getElementById('recText');
            const useRecBtn = document.getElementById('useRecBtn');

            let currentObs = null;
            let currentDone = false;
            let lastRecommendation = 0;

            function pretty(obj) { return JSON.stringify(obj, null, 2); }

            function setBanner(text, isSuccess=false) {
                if (!text) {
                    bannerEl.style.display = 'none';
                    bannerEl.className = '';
                    return;
                }
                bannerEl.style.display = 'block';
                bannerEl.textContent = text;
                bannerEl.className = isSuccess ? 'success' : '';
            }

            function recommendAllocate(obs) {
                if (!obs) return 0;
                const beds = Number(obs.beds || 0);
                const patients = Array.isArray(obs.patients) ? obs.patients : [];
                const highOrEmerg = patients.filter(p => p.severity === 'high' || p.emergency).length;
                const medium = patients.filter(p => p.severity === 'medium').length;
                const base = highOrEmerg > 0 ? highOrEmerg : (medium > 0 ? medium : 1);
                return Math.max(0, Math.min(base, beds));
            }

            function renderRecommendation(obs) {
                const rec = recommendAllocate(obs);
                lastRecommendation = rec;
                recTextEl.textContent = String(rec);
            }

            function renderSummary(obs) {
                if (!obs) {
                    summaryEl.innerHTML = '<div style="grid-column: 1/-1; padding: 2rem; text-align: center; color: var(--text-secondary);">No active simulation. Initialize to begin.</div>';
                    return;
                }
                const patients = Array.isArray(obs.patients) ? obs.patients : [];
                const emerg = patients.filter(p => !!p.emergency).length;
                const high = patients.filter(p => p.severity === 'high').length;
                const medium = patients.filter(p => p.severity === 'medium').length;
                
                summaryEl.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-label">Available Beds</div>
                        <div class="stat-value highlight">${obs.beds} <span style="font-size:1rem; color:#94a3b8">/ ${obs.total_beds}</span></div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Timeline</div>
                        <div class="stat-value">${obs.step} <span style="font-size:1rem; color:#94a3b8">/ ${obs.max_steps}</span></div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Total Waiting</div>
                        <div class="stat-value">${patients.length}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Critical / Emerg</div>
                        <div class="stat-value" style="color: #ef4444;">${high} / ${emerg}</div>
                    </div>
                `;
            }

            function renderPatients(obs) {
                const patients = obs && Array.isArray(obs.patients) ? obs.patients : [];
                if (patients.length === 0) {
                    patientsEl.innerHTML = '<tr><td colspan="4" style="text-align:center; padding:2rem; color:var(--text-secondary);">Waiting room is empty.</td></tr>';
                    return;
                }
                
                // Sort to show emergencies and high severity first
                const sorted = [...patients].sort((a, b) => {
                    if (a.emergency && !b.emergency) return -1;
                    if (!a.emergency && b.emergency) return 1;
                    const sevScore = {high: 3, medium: 2, low: 1};
                    return (sevScore[b.severity] || 0) - (sevScore[a.severity] || 0);
                });

                const rows = sorted.map(p => {
                    const sev = String(p.severity || 'unknown');
                    const isEmerg = p.emergency;
                    const waiting = (p.waiting_steps ?? 0);
                    
                    return `
                        <tr class="${isEmerg ? 'emergency-row' : ''}">
                            <td style="font-weight: 600; color: white;">#${p.id ?? ''}</td>
                            <td><span class="badge ${sev}">${sev}</span></td>
                            <td>${isEmerg ? '<span class="badge emergency">EMERGENCY</span>' : '<span style="color:var(--text-secondary)">Standard</span>'}</td>
                            <td>${waiting} step${waiting !== 1 ? 's' : ''}</td>
                        </tr>
                    `;
                }).join('');
                patientsEl.innerHTML = rows;
            }

            function renderRaw(obs) {
                stateRawEl.textContent = pretty(obs || {});
            }

            function setObs(obs) {
                currentObs = obs;
                if (obs && typeof obs.beds === 'number') {
                    allocateEl.max = String(obs.beds);
                    if (Number(allocateEl.value) > obs.beds) allocateEl.value = String(obs.beds);
                }
                renderSummary(obs);
                renderPatients(obs);
                renderRaw(obs);
                renderRecommendation(obs);
            }

            function setDone(done) {
                currentDone = !!done;
                stepBtn.disabled = currentDone;
                if (currentDone) setBanner('Simulation Complete! Review your score in the System Log.', true);
            }

            async function doReset() {
                setBanner('');
                lastEl.textContent = 'Initializing environment...';
                stepBtn.disabled = true;
                const task = taskEl.value;
                try {
                    const res = await fetch(`/reset?task=${encodeURIComponent(task)}`, { method: 'POST' });
                    const data = await res.json();
                    if (data.observation) {
                        setDone(false);
                        setObs(data.observation);
                        allocateEl.value = String(lastRecommendation);
                        stepBtn.disabled = false;
                        lastEl.textContent = 'Environment Ready.\\n\\n' + pretty(data);
                    } else {
                        lastEl.textContent = pretty(data);
                        setBanner('Reset failed. Check System Log.');
                    }
                } catch (e) {
                    setBanner('Network error.');
                    lastEl.textContent = e.toString();
                }
            }

            async function doStep() {
                if (currentDone) return;
                setBanner('');
                stepBtn.disabled = true;
                const allocate = Number(allocateEl.value || 0);
                lastEl.textContent = 'Executing step...';
                
                try {
                    const res = await fetch('/step', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ allocate })
                    });
                    const data = await res.json();
                    lastEl.textContent = pretty(data);

                    if (data.observation) setObs(data.observation);

                    if (data.error) {
                        setBanner(String(data.error));
                        stepBtn.disabled = false;
                        return;
                    }

                    if (data.done) {
                        setDone(true);
                        try {
                            const gradeRes = await fetch('/grade');
                            const grade = await gradeRes.json();
                            lastEl.textContent = 'FINAL REPORT\\n==========\\n\\n' + pretty({ ...data, final_grade: grade });
                        } catch (e) {}
                        return;
                    }

                    stepBtn.disabled = false;
                } catch (e) {
                    setBanner('Network error.');
                    lastEl.textContent = e.toString();
                    stepBtn.disabled = false;
                }
            }

            resetBtn.addEventListener('click', () => doReset());
            stepBtn.addEventListener('click', () => doStep());
            useRecBtn.addEventListener('click', () => { allocateEl.value = String(lastRecommendation); });
            
            // Initial render
            renderSummary(null);
            renderPatients(null);
            doReset();
        </script>
    </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home_ui():
        return _interactive_html()


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
def web_ui():
        return _interactive_html()

def _clean_obs(s: dict) -> dict:
    return {
        "beds":       s["beds"],
        "total_beds": s["total_beds"],
        "patients":   s["patients"],
        "step":       s["step"],
        "max_steps":  s["max_steps"],
        "difficulty": s["difficulty"],
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()