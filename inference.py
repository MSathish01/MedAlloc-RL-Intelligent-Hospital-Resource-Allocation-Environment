"""
MedAlloc-RL Inference Script
Output format: [START]/[STEP]/[END] plain text as required by evaluator
"""

import os
import re
import requests
from openai import OpenAI

# Required env vars
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

ENV_URL = "https://msathish-hospital-env.hf.space"
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "sk-placeholder",
)


def get_action(observation: dict) -> int:
    patients = observation.get("patients", [])
    beds = observation.get("beds", 0)

    high = sum(1 for p in patients if p["severity"] == "high")
    medium = sum(1 for p in patients if p["severity"] == "medium")
    emergency = sum(1 for p in patients if p.get("emergency", False))

    prompt = f"""You are a hospital resource allocation agent.
Available beds: {beds}
Patients waiting: {len(patients)} (high={high}, medium={medium}, emergency={emergency})
Reply with ONLY a single integer - number of beds to allocate."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        numbers = re.findall(r'\d+', raw)
        if numbers:
            return max(0, min(int(numbers[0]), beds))
    except Exception:
        pass

    # Fallback: greedy
    high_p = [p for p in patients if p["severity"] == "high" or p.get("emergency")]
    if high_p:
        return min(len(high_p), beds)
    medium_p = [p for p in patients if p["severity"] == "medium"]
    if medium_p:
        return min(len(medium_p), beds)
    return min(len(patients), beds)


def run_task(task: str) -> float:
    res = requests.post(f"{ENV_URL}/reset", params={"task": task}, timeout=30)
    data = res.json()
    obs = data["observation"]

    print(f"[START] task={task}", flush=True)

    step_num = 0
    total_reward = 0.0
    final_score = 0.0

    while True:
        allocate = get_action(obs)

        res = requests.post(f"{ENV_URL}/step", json={"allocate": allocate}, timeout=30)
        data = res.json()

        obs = data["observation"]
        reward = data["reward"]
        score = data["score"]
        done = data["done"]
        total_reward += reward
        final_score = score
        step_num += 1

        print(f"[STEP] step={step_num} reward={reward:.2f} score={score:.3f} action=allocate({allocate}) done={str(done).lower()}", flush=True)

        if done:
            break

    print(f"[END] task={task} score={final_score:.3f} steps={step_num}", flush=True)

    return final_score


if __name__ == "__main__":
    all_scores = {}

    for task in ["easy", "medium", "hard"]:
        try:
            score = run_task(task)
            all_scores[task] = score
        except Exception as e:
            print(f"[END] task={task} score=0.0 steps=0 error={str(e)}", flush=True)
            all_scores[task] = 0.0

    avg = sum(all_scores.values()) / len(all_scores)
    print(f"[SUMMARY] easy={all_scores.get('easy', 0):.3f} medium={all_scores.get('medium', 0):.3f} hard={all_scores.get('hard', 0):.3f} avg={avg:.3f}", flush=True)