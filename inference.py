import os
import re
import time
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")

ENV_URL = "https://msathish-hospital-env.hf.space"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "sk-placeholder",
)


def wake_up():
    for _ in range(10):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=15)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(6)
    return False


def api_post(url, **kwargs):
    for attempt in range(5):
        try:
            r = requests.post(url, timeout=30, **kwargs)
            if r.status_code == 503:
                time.sleep(8)
                continue
            r.raise_for_status()
            return r
        except Exception:
            time.sleep(4)
    raise Exception(f"Failed after 5 retries: {url}")


def api_get(url):
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 503:
                time.sleep(8)
                continue
            r.raise_for_status()
            return r
        except Exception:
            time.sleep(4)
    raise Exception(f"Failed after 5 retries: {url}")


def get_action(observation: dict) -> int:
    patients = observation.get("patients", [])
    beds = observation.get("beds", 0)
    if not patients or beds == 0:
        return 0

    high = sum(1 for p in patients if p["severity"] == "high")
    medium = sum(1 for p in patients if p["severity"] == "medium")
    emergency = sum(1 for p in patients if p.get("emergency", False))

    prompt = f"""You are a hospital resource allocation agent.
Available beds: {beds}
Patients waiting: {len(patients)} (high={high}, medium={medium}, emergency={emergency})
Reply with ONLY a single integer - number of beds to allocate. Prioritize high severity and emergencies."""

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

    # Greedy fallback
    high_p = [p for p in patients if p["severity"] == "high" or p.get("emergency")]
    if high_p:
        return min(len(high_p), beds)
    medium_p = [p for p in patients if p["severity"] == "medium"]
    if medium_p:
        return min(len(medium_p), beds)
    return min(len(patients), beds)


def safe_score(val):
    """Always strictly between 0 and 1."""
    return round(max(0.001, min(0.999, float(val))), 3)


def run_task(task: str) -> float:
    res = api_post(f"{ENV_URL}/reset", params={"task": task})
    data = res.json()
    obs = data["observation"]

    # [START] — required format
    print(f"[START] task={task}", flush=True)

    step_num = 0
    final_score = 0.1
    max_steps = obs.get("max_steps", 5)

    while True:
        allocate = get_action(obs)

        step_res = api_post(f"{ENV_URL}/step", json={"allocate": allocate})
        step_data = step_res.json()

        obs         = step_data.get("observation", obs)
        reward      = float(step_data.get("reward", 0.0))
        score       = safe_score(step_data.get("score", 0.1))
        done        = bool(step_data.get("done", False))
        step_num   += 1
        final_score = score

        # [STEP] — required format
        print(f"[STEP] step={step_num} reward={reward:.2f} score={score:.3f} action=allocate({allocate}) done={str(done).lower()}", flush=True)

        if done or step_num >= max_steps:
            break

    # Get final grade from /grade endpoint
    try:
        grade_res = api_get(f"{ENV_URL}/grade")
        grade_data = grade_res.json()
        final_score = safe_score(grade_data.get("score", final_score))
    except Exception:
        pass

    # [END] — required format with score
    print(f"[END] task={task} score={final_score:.3f} steps={step_num}", flush=True)
    return final_score


if __name__ == "__main__":
    wake_up()

    all_scores = {}
    for task in ["easy", "medium", "hard"]:
        try:
            score = run_task(task)
            all_scores[task] = score
        except Exception as e:
            fallback = 0.1
            print(f"[END] task={task} score={fallback:.3f} steps=1 error={str(e)}", flush=True)
            all_scores[task] = fallback

    avg = safe_score(sum(all_scores.values()) / len(all_scores))
    print(f"[SUMMARY] easy={all_scores.get('easy',0.1):.3f} medium={all_scores.get('medium',0.1):.3f} hard={all_scores.get('hard',0.1):.3f} avg={avg:.3f}", flush=True)