import requests

class HospitalClient:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url
    
    def health(self):
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def reset(self):
        """Reset the environment"""
        response = requests.post(f"{self.base_url}/reset")
        return response.json()
    
    def step(self, patient_id=0):
        """Take a step in the environment"""
        response = requests.post(
            f"{self.base_url}/step",
            json={"patient_id": patient_id}
        )
        return response.json()
    
    def get_state(self):
        """Get current state"""
        response = requests.get(f"{self.base_url}/state")
        return response.json()

if __name__ == "__main__":
    client = HospitalClient()
    
    # Test health
    print("Health:", client.health())
    
    # Reset environment
    print("Reset:", client.reset())
    
    # Run episode
    done = False
    total_reward = 0
    
    while not done:
        result = client.step()
        total_reward += result["reward"]
        done = result["done"]
        print(f"Reward: {result['reward']}, Done: {done}")
    
    print(f"Total Reward: {total_reward}")
