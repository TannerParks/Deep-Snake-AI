import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        """This is the constructor of the class. It initializes the layers of the neural network based on the 
        input (state observations) and output (possible actions) dimensions."""
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """This is the forward pass of the neural network. It's called when the model is called with an input tensor."""
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))

        return self.layer3(x)


class Trainer:
    def __init__(self, model, target_model, gamma, learning_rate=0.00):
        """This is the constructor of the class. It initializes the optimizer and the loss function."""
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def update_target(self):
        """This function updates the target model with the current model's weights."""
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()    # Ensure the target model to evaluation mode

    def calculate_loss(self, batch):
        pass

    def optimize(self, memory, batch):
        pass