import os
import re
import time
import requests
from openai import OpenAI

# -------------------------------
# ENV CONFIG
# -------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

ENV_URL = "https://msathish-hospital-env.hf.space"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "sk-placeholder",
)

# -------------------------------
# WAKE SPACE
# -------------------------------
def wake_up_space():
    for _ in range(10):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=15)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(5)
    return False

# -------------------------------
# SAFE API CALL
# -------------------------------
def api_call(method, url, **kwargs):
    for _ in range(5):
        try:
            if method == "post":
                res = requests.post(url, timeout=30, **kwargs)
            else:
                res = requests.get(url, timeout=30, **kwargs)

            if res.status_code == 503:
                time.sleep(5)
                continue

            res.raise_for_status()
            return res

        except Exception:
            time.sleep(3)

    raise Exception("API failed")

# -------------------------------
# ACTION LOGIC
# -------------------------------
def get_action(obs):
    patients = obs.get("patients", [])
    beds = obs.get("beds", 0)

    if beds == 0 or not patients:
        return 0

    # LLM try
    try:
        prompt = f"Beds: {beds}, Patients: {len(patients)}. Return only number."
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )

        txt = res.choices[0].message.content.strip()
        num = re.findall(r"\d+", txt)

        if num:
            return min(int(num[0]), beds)

    except:
        pass

    # fallback
    return min(len(patients), beds)

# -------------------------------
# SAFE SCORE
# -------------------------------
def safe_score(x):
    return max(0.000001, min(0.999999, x))

# -------------------------------
# RUN TASK
# -------------------------------
def run_task(task):
    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task} env=medalloc model={MODEL_NAME}", flush=True)

    try:
        res = api_call("post", f"{ENV_URL}/reset", params={"task": task})
        data = res.json()

        obs = data["observation"]
        done = False
        max_steps = obs.get("max_steps", 5)

        while not done and steps < max_steps:

            # 🔥 STOP if no patients
            if len(obs.get("patients", [])) == 0:
                break

            steps += 1
            action = get_action(obs)

            try:
                res = api_call("post", f"{ENV_URL}/step", json={"allocate": action})
                step_data = res.json()

                reward = float(step_data.get("reward", 0.0))
                done = bool(step_data.get("done", False))

                rewards.append(round(reward, 2))

                print(
                    f"[STEP] step={steps} action=allocate({action}) "
                    f"reward={reward:.2f} done={str(done).lower()} error=null",
                    flush=True
                )

                obs = step_data.get("observation", obs)

            except Exception as e:
                print(
                    f"[STEP] step={steps} action=allocate({action}) "
                    f"reward=0.00 done=true error={str(e)}",
                    flush=True
                )
                break

        try:
            g = api_call("get", f"{ENV_URL}/grade")
            score = safe_score(float(g.json().get("score", 0.0)))
            success = True
        except:
            score = 0.001

    except Exception as e:
        print(
            f"[STEP] step=1 action=none reward=0.00 done=true error={str(e)}",
            flush=True
        )
        score = 0.001

    reward_str = ",".join([f"{r:.2f}" for r in rewards])

    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={reward_str}",
        flush=True
    )

    return score

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    wake_up_space()

    all_scores = {}

    for task in ["easy", "medium", "hard"]:
        try:
            score = run_task(task)
            all_scores[task] = safe_score(score)
        except Exception:
            print("[END] success=false steps=0 rewards=", flush=True)
            all_scores[task] = 0.001

    final_score = sum(all_scores.values()) / len(all_scores)