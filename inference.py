import requests

BASE_URL = "https://msathish-hospital-env.hf.space"

def smart_agent():
    res = requests.post(f"{BASE_URL}/reset")
    data = res.json()

    total_reward = 0

    while True:
        obs = data["observation"]

        beds = obs["beds"]
        patients = obs["patients"]

        # prioritize high severity
        high = [p for p in patients if p["severity"] == "high"]
        medium = [p for p in patients if p["severity"] == "medium"]

        if len(high) > 0:
            allocate = min(len(high), beds)
        elif len(medium) > 0:
            allocate = min(len(medium), beds)
        else:
            allocate = min(len(patients), beds)

        res = requests.post(
            f"{BASE_URL}/step",
            json={"allocate": allocate}
        )

        data = res.json()
        total_reward += data["reward"]

        print(f"Step Reward: {data['reward']} | Score: {data['score']}")

        if data["done"]:
            break

    print("\nFinal Reward:", total_reward)


if __name__ == "__main__":
    smart_agent()