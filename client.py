import requests


class HospitalEnvClient:
    def __init__(self, base_url: str):
        """
        Initialize client with base URL of the environment
        Example: https://msathish-hospital-env.hf.space
        """
        self.base_url = base_url

    def reset(self):
        """
        Start a new episode
        """
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()

    def step(self, allocate: int):
        """
        Perform an action (allocate beds)
        """
        payload = {"allocate": allocate}
        response = requests.post(f"{self.base_url}/step", json=payload)
        response.raise_for_status()
        return response.json()

    def state(self):
        """
        Get current environment state
        """
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()

    def health(self):
        """
        Check if API is running
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


# -------------------------------
# Example usage (for testing)
# -------------------------------
if __name__ == "__main__":
    BASE_URL = "https://msathish-hospital-env.hf.space"

    env = HospitalEnvClient(BASE_URL)

    print("ðŸ”„ Resetting environment...")
    result = env.reset()
    print(result)

    print("\nâž¡ Taking action: allocate = 2")
    step_result = env.step(2)
    print(step_result)

    print("\nðŸ“Š Current state:")
    print(env.state())

    print("\nðŸ’š Health check:")
    print(env.health())
