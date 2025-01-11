import numpy as np

class DummyAgent:
    def __init__(self):
        pass
    
    def get_action(self, observation):
        action = np.random.choice(2, 3)

        return action
    
if __name__ == "__main__":
    dummy_agent = DummyAgent()
    dummy_agent.get_action()