"""
MedAlloc-RL Inference Script
----------------------------
Runs the hospital allocation agent across all 3 task difficulties.
Output format strictly follows OpenEnv [START] / [STEP] / [END] spec.
"""

import os
import json
import requests
from openai import OpenAI

# -----------------------------------------------
# REQUIRED ENV VARS â€” set these in HF Space secrets
# -----------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# Your HF Space URL
ENV_URL = "https://msathish01-medalloc-rl-intelligent-hospital-resource-allocation-environment.hf.space"

# -----------------------------------------------
# OPENAI CLIENT (required by hackathon rules)
# -----------------------------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "sk-placeholder",
)

# -----------------------------------------------
# LLM AGENT
# -----------------------------------------------
def get_action(observation: dict) -> int:
    """Ask the LLM how many beds to allocate this step."""
    patients = observation.get("patients", [])
    beds     = observation.get("beds", 0)

    high_count      = sum(1 for p in patients if p["severity"] == "high")
    medium_count    = sum(1 for p in patients if p["severity"] == "medium")
    emergency_count = sum(1 for p in patients if p.get("emergency", False))

    prompt = f"""You are an intelligent hospital resource allocation agent.

Current hospital state:
- Available beds this step: {beds}
- Total patients waiting: {len(patients)}
- High severity patients: {high_count}
- Medium severity patients: {medium_count}
- Emergency patients: {emergency_count}
- Current step: {observation.get('step', 0)} / {observation.get('max_steps', 5)}

Reward rules:
- Treating high severity patient: +3.0
- Treating medium severity patient: +2.0
- Treating low severity patient: +1.0
- Leaving high severity untreated: -2.0
- Wasting a bed (when patients waiting): -0.3 per bed
- Emergency untreated: -1.5 extra penalty

Your goal: Maximize total reward by deciding how many beds to allocate.
Always treat emergency and high-severity patients first.
Do NOT waste beds if patients are waiting.

Reply with ONLY a single integer â€” the number of beds to allocate.
No explanation. Just the number."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        # Extract first integer found
        import re
        numbers = re.findall(r'\d+', raw)
        if numbers:
            return max(0, min(int(numbers[0]), beds))
        return fallback_action(observation)
    except Exception as e:
        return fallback_action(observation)


def fallback_action(observation: dict) -> int:
    """Greedy fallback if LLM call fails."""
    patients = observation.get("patients", [])
    beds     = observation.get("beds", 0)
    # Prioritize: emergency + high first
    priority = [p for p in patients if p.get("emergency") or p["severity"] == "high"]
    if priority:
        return min(len(priority), beds)
    medium = [p for p in patients if p["severity"] == "medium"]
    if medium:
        return min(len(medium), beds)
    return min(len(patients), beds)


# -----------------------------------------------
# EPISODE RUNNER
# -----------------------------------------------
def run_task(task: str) -> float:
    """Run one full episode and return final score."""

    # Reset environment
    res  = requests.post(f"{ENV_URL}/reset", params={"task": task}, timeout=30)
    data = res.json()
    obs  = data["observation"]

    # [START] â€” required by OpenEnv spec
    print(json.dumps({
        "[START]":         True,
        "task":            task,
        "beds":            obs["beds"],
        "initial_patients": len(obs["patients"]),
        "max_steps":       obs["max_steps"],
    }))

    step_num     = 0
    total_reward = 0.0
    final_score  = 0.0

    while True:
        allocate = get_action(obs)

        res  = requests.post(f"{ENV_URL}/step", json={"allocate": allocate}, timeout=30)
        data = res.json()

        obs          = data["observation"]
        reward       = data["reward"]
        score        = data["score"]
        done         = data["done"]
        info         = data.get("info", {})
        total_reward += reward
        final_score  = score
        step_num    += 1

        # [STEP] â€” required by OpenEnv spec
        print(json.dumps({
            "[STEP]":              step_num,
            "action_allocate":     allocate,
            "reward":              reward,
            "score":               score,
            "beds_remaining":      obs["beds"],
            "patients_waiting":    len(obs["patients"]),
            "treated_total":       info.get("treated_total", 0),
            "emergencies_seen":    info.get("emergencies_seen", 0),
            "done":                done,
        }))

        if done:
            break

    # [END] â€” required by OpenEnv spec
    print(json.dumps({
        "[END]":        True,
        "task":         task,
        "total_reward": round(total_reward, 2),
        "final_score":  round(final_score, 3),
        "steps_taken":  step_num,
    }))

    return final_score


# -----------------------------------------------
# MAIN
# -----------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("MedAlloc-RL â€” Hospital Resource Allocation")
    print("=" * 50)

    all_scores = {}

    for task in ["easy", "medium", "hard"]:
        print(f"\n{'=' * 50}")
        print(f"TASK: {task.upper()}")
        print("=" * 50)
        try:
            score = run_task(task)
            all_scores[task] = score
        except Exception as e:
            print(json.dumps({"[ERROR]": True, "task": task, "error": str(e)}))
            all_scores[task] = 0.0

    print("\n" + "=" * 50)
    print("FINAL SCORES")
    print("=" * 50)
    for task, score in all_scores.items():
        print(f"  {task:8s}: {score:.3f}")

    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  {'AVERAGE':8s}: {avg:.3f}")
    print("=" * 50)
