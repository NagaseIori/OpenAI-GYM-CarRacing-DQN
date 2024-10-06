import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CarRacingDQNAgent:
    def __init__(
        self,
        action_space=[
            (-1, 1, 0.5),
            (0, 1, 0.5),
            (1, 1, 0.5),  #           Action Space Structure
            (-1, 1, 0),
            (0, 1, 0),
            (1, 1, 0),  #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.5),
            (0, 0, 0.5),
            (1, 0, 0.5),  # Range        -1~1       0~1   0~1
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        ],
        frame_stack_num=5,
        memory_size=10000,
        gamma=0.99,  # discount rate
        epsilon=0.9,  # exploration rate
        epsilon_min=0.05,
        epsilon_decay=0.99995,
        learning_rate=0.001,
    ):
        self._n_actions = 5
        self.action_space = action_space
        self.frame_stack_num = frame_stack_num
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        # Set the device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Build the model and target model, and move them to the appropriate device
        self.model = self.build_model().to(self.device)
        self.target_model = self.build_model().to(self.device)
        self.update_target_model()
        # Initialize the optimizer with Adam
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=1e-7
        )

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = nn.Sequential(
            nn.Conv2d(
                self.frame_stack_num, 6, kernel_size=7, stride=3
            ),  # Convolutional layer with 6 filters
            nn.ReLU(),  # Activation function
            # nn.MaxPool2d(kernel_size=2),  # Max pooling layer
            nn.Conv2d(6, 12, kernel_size=4),  # Convolutional layer with 12 filters
            nn.ReLU(),  # Activation function
            nn.Conv2d(12, 24, kernel_size=4),  # Convolutional layer with 12 filters
            nn.ReLU(),  # Activation function
            # nn.MaxPool2d(kernel_size=2),  # Max pooling layer
            nn.Flatten(),  # Flatten the tensor
            nn.Linear(13824, 512),  # Adjusted input size for fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(
                512, 5
            ),  # Output layer with the number of actions
        )
        return model

    def update_target_model(self):
        # Copy weights from the main model to the target model
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        # Store the experience in memory
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def act(self, state):
        # Decide on an action using epsilon-greedy policy
        if np.random.rand() > self.epsilon:
            # Convert state to a tensor and pass through the model to get action values
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                act_values = self.model(state)
            # Choose the action with the highest value
            action_index = torch.argmax(act_values).item()
        else:
            # Choose a random action
            action_index = random.randrange(self._n_actions)
        return action_index

    def replay(self, batch_size):
        # Train the model using randomly sampled experiences from memory
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            # Convert state and next_state to tensors
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=self.device
            )
            # Get the current prediction for the given state
            target = self.model(state.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            if done:
                # If the episode is done, the target is simply the reward
                target[action_index] = reward
            else:
                # Otherwise, calculate the target using the target model
                t = (
                    self.target_model(next_state.unsqueeze(0))
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                target[action_index] = reward + self.gamma * np.amax(t)
            train_state.append(state)
            train_target.append(target)
        # Stack states and convert targets to tensors
        train_state = torch.stack(train_state)
        train_target = torch.tensor(
            np.array(train_target), dtype=torch.float32, device=self.device
        )
        # Perform a gradient descent step
        self.optimizer.zero_grad()
        predictions = self.model(train_state)
        loss = F.mse_loss(predictions, train_target)
        loss.backward()
        self.optimizer.step()
        # Update epsilon for exploration-exploitation tradeoff
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # Load model weights from a file and update the target model
        self.model.load_state_dict(torch.load(name))
        self.update_target_model()

    def save(self, name):
        # Save the target model weights to a file
        torch.save(self.target_model.state_dict(), name)