from optimizers.replay_buffer import  ReplayBuffer
# 简单示意优先级经验回放
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity=10000):
        super().__init__(capacity)
        self.priorities = []

    def push(self, transition, priority=1.0):
        super().push(transition)
        self.priorities.append(priority)
        if len(self.priorities) > self.capacity:
            self.priorities.pop(0)
