import numpy as np

class ReplayMemory(object):
    def __init__(self, args, frame_shape):
        self.args = args 
        self.frame_shape = frame_shape
        self.count = 0
        self.current = 0

        self.actions = np.empty(self.args.memory_size, dtype=np.uint8)
        self.rewards= np.empty(self.args.memory_size, dtype=np.float32)
        self.terminals = np.empty(self.args.memory_size, dtype=np.bool)
        self.next_frames = np.empty([self.args.memory_size] + self.frame_shape, dtype=np.float32)


    def add(self, action, reward, terminal, next_frame):
        self.actions[self.current] = action
        self.rewards[self.current] = reward  
        self.terminals[self.current] = terminal 
        self.next_frames[self.current] = next_frame 

        self.count = max(self.count, self.current+1)
        self.current = (self.current+1) % self.args.memory_size

class SimpleMemory(ReplayMemory):
    def __init__(self, args, frame_shape):
        super(SimpleMemory, self).__init__(args, frame_shape)
        self.prestates = np.empty([self.args.batch_size] + self.frame_shape, dtype=np.float32)
        self.poststates = np.empty([self.args.batch_size] + self.frame_shape, dtype=np.float32)

    def mini_batch(self):
        batch_indices = []
        while len(batch_indices) < self.args.batch_size:
            while True:
                idx = np.random.randint(low=1, high=self.count)
                if idx == self.current:
                    continue
                if self.terminals[idx-1]:
                    continue
                break
            self.prestates[len(batch_indices)] = self.next_frames[idx-1]
            self.poststates[len(batch_indices)] = self.next_frames[idx]
            batch_indices.append(idx)
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        terminals = self.rewards[batch_indices]
        return self.prestates, actions, rewards, terminals, self.poststates

         

