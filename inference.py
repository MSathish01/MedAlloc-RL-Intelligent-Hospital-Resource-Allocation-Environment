import requests

BASE_URL = "http://localhost:7860"

# reset
res = requests.post(f"{BASE_URL}/reset").json()
print("Reset:", res)

done = False
total_reward = 0

while not done:
    action = {"patient_id": 0}
    res = requests.post(f"{BASE_URL}/step", json=action).json()
    
    total_reward += res["reward"]
    done = res["done"]
    print(f"Step - Reward: {res['reward']}, Done: {done}")

print("Final Score:", total_reward)
