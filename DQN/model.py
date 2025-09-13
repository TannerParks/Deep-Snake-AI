import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        """This is the constructor of the class. It initializes the layers of the neural network based on the 
        input (state observations) and output (possible actions) dimensions."""
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        """This is the forward pass of the neural network. It's called when the model is called with an input tensor."""
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
    
    def save(self, path):
        """Save the model's weights to a file."""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load the model's weights from a file."""
        self.load_state_dict(torch.load(path))
        

class Trainer:
    def __init__(self, policy_model, target_model, device, gamma, learning_rate=0.001, w_decay=0.001):
        """This is the constructor of the class. It initializes the optimizer and the loss function."""
        self.policy_model = policy_model
        self.target_model = target_model
        self.device = device
        self.gamma = gamma
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=w_decay)
        self.criterion = nn.SmoothL1Loss(beta=20.0) # nn.MSELoss()

    def update_target(self):
        """Update the target model with the current model's weights."""
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()    # Set the target model to evaluation mode

    def soft_update_target(self, tau=0.001):
        """Perform a soft update of the target model by interpolating its weights with the policy model's weights."""
        target_state_dict = self.target_model.state_dict()
        policy_state_dict = self.policy_model.state_dict()

        for key in target_state_dict.keys():
            target_state_dict[key] = tau * policy_state_dict[key] + (1 - tau) * target_state_dict[key]
        
        self.target_model.load_state_dict(target_state_dict)

    def prepare_batch(self, batch):
        """This function is passed a batch of transitions and converts it to a tensor of states, actions, next states, and rewards."""
        states = torch.tensor(np.array(batch.state), device=self.device, dtype=torch.float32)
        actions = torch.tensor(batch.action, device=self.device, dtype=torch.long)
        rewards = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)

        # Process next_states: Filter out None values and convert to tensor
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
        # Filter out None values, convert the rest to tensor
        non_final_next_states_list = [s for s in batch.next_state if s is not None]

        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.tensor(np.array(non_final_next_states_list), dtype=torch.float32, device=self.device)
        else:
            non_final_next_states = torch.empty((0, *batch.state[0].shape), dtype=torch.float32, device=self.device)

        return states, actions, non_final_next_states, rewards, non_final_mask

    def optimize(self, memory, batch_size=32):
        """Optimize the Deep Q-Network (DQN) model by updating its weights based on a batch of transitions from the replay memory."""
        # Sample then Transform the batch of transitions into tensors of states, actions, next states, rewards, and masks
        transitions = memory.sample(batch_size)
        batch = memory.transform(transitions)
        states, actions, non_final_next_states, rewards, non_final_mask = self.prepare_batch(batch)

        # Convert one-hot encoded actions to indices    [0, 0, 1] -> [2]
        actions_indices = actions.max(1)[1].unsqueeze(-1)

        # Predict the Q values for the current states and gather the Q values for the actions taken
        pred = self.policy_model(states)
        current_q_values = pred.gather(1, actions_indices)
        
        # Calculate the target Q values and make terminal states' Q values 0
        target_q_values = torch.zeros(batch_size, device=self.device)
        target_q_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        # Calculate the loss between the current Q-values and the expected Q-values
        expected_q_values = rewards + self.gamma * target_q_values
        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(-1))

        # Optimize the model by updating its weights using backpropagation and gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
