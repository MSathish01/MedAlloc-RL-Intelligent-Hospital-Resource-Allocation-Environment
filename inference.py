import os
import json
import requests
from openai import OpenAI

# --- Required env vars (do NOT hardcode these) ---
BASE_URL   = os.environ.get("API_BASE_URL", "https://msathish01-medalloc-rl.hf.space")
MODEL_NAME = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN   = os.environ.get("HF_TOKEN",     "")

client = OpenAI(base_url=BASE_URL if "openai" not in BASE_URL else None,
                api_key=HF_TOKEN or "sk-placeholder")

ENV_URL = "https://msathish01-medalloc-rl-intelligent-hospital-resource-allocation-environment.hf.space"

def get_action(observation: dict) -> int:
    """Ask the LLM what allocation to make."""
    prompt = f"""
You are a hospital resource allocation agent.
Current state:
- Available beds: {observation['beds']}
- Patients waiting: {json.dumps(observation['patients'])}

Decide how many beds to allocate this step.
Prioritize high-severity patients first.
Reply with ONLY a single integer (the number of beds to allocate). No explanation.
""".strip()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        return max(0, int(raw))
    except Exception:
        # Fallback: greedy allocation
        beds = observation.get("beds", 0)
        patients = observation.get("patients", [])
        high = sum(1 for p in patients if p["severity"] == "high")
        return min(max(high, 1), beds)

def run_task(task: str):
    """Run one full episode for a given task difficulty."""
    res = requests.post(f"{ENV_URL}/reset", params={"task": task})
    data = res.json()
    obs  = data["observation"]

    print(json.dumps({
        "[START]": True,
        "task":    task,
        "beds":    obs["beds"],
        "patients": len(obs["patients"]),
    }))

    step_num     = 0
    total_reward = 0.0
    final_score  = 0.0

    while True:
        allocate = get_action(obs)

        res  = requests.post(f"{ENV_URL}/step", json={"allocate": allocate})
        data = res.json()

        obs          = data["observation"]
        reward       = data["reward"]
        score        = data["score"]
        done         = data["done"]
        total_reward += reward
        final_score  = score
        step_num    += 1

        print(json.dumps({
            "[STEP]":    step_num,
            "action":    allocate,
            "reward":    reward,
            "score":     score,
            "beds_left": obs["beds"],
            "patients_remaining": len(obs["patients"]),
            "done":      done,
        }))

        if done:
            break

    print(json.dumps({
        "[END]":        True,
        "task":         task,
        "total_reward": round(total_reward, 2),
        "final_score":  round(final_score,  3),
        "steps":        step_num,
    }))

    return final_score

if __name__ == "__main__":
    scores = {}
    for task in ["easy", "medium", "hard"]:
        print(f"\n{'='*40}")
        print(f"Running task: {task}")
        print('='*40)
        scores[task] = run_task(task)

    print("\n--- FINAL SCORES ---")
    for task, score in scores.items():
        print(f"{task}: {score:.3f}")
