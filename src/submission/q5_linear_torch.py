import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.test_env import EnvTest
from utils.general import join
from core.deep_q_learning_torch import DQN
from .q3_schedule import LinearExploration, LinearSchedule

import os
import yaml
yaml.add_constructor('!join', join)

config_file = open("config/q5_linear.yml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

############################################################
# Problem 5: Linear Approximation
############################################################


class Linear(DQN):
    """
    Implementation of a single fully connected layer with Pytorch to be utilized 
    in the DQN algorithm.
    """
    ############################################################
    # Problem 5b: initializing models
    
    def initialize_models(self):
        """
        Creates the 2 separate networks (Q network and Target network). The input
        to these networks will be an image of shape self.img_height * self.img_width with 
        channels = self.n_channels * self.config["hyper_params"]["state_history"].

        Args:
            q_network (torch model): variable to store our q network implementation

            target_network (torch model): variable to store our target network implementation

        TODO:
            (1) Set self.q_network to be a linear layer with num_actions as the output 
            size. 

            (2) Set self.target_network to be the same configuration as self.q_netowrk.
            but initialized by scratch. 

        Hint:
            (1) Start by figuring out what the input size is to the networks.
            (2) Simply setting self.target_network = self.q_network is incorrect.
            (3) Consult nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) 
            which should be useful for your implementation.
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        ### START CODE HERE ###
        input_size = img_height * img_width * n_channels * self.config["hyper_params"]["state_history"]
        self.q_network = nn.Linear(input_size, num_actions)
        self.target_network = nn.Linear(input_size, num_actions)
        ### END CODE HERE ###

    ############################################################
    # Problem 5c: get_q_values

    def get_q_values(self, state, network='q_network'):
        """
        Returns Q values for all actions.

        Args:
            state (torch tensor): shape = (batch_size, img height, img width, 
                                            nchannels x config["hyper_params"]["state_history"])
            
            network (str): The name of the network, either "q_network" or "target_network"

        Returns:
            out (torch tensor): shape = (batch_size, num_actions)

        TODO: 
            Perform a forward pass of the input state through the selected network
            and return the output values.


        Hints:
            (1) Look up torch.flatten (https://pytorch.org/docs/stable/generated/torch.flatten.html)
            (2) You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ### START CODE HERE ###
        flattened_input = torch.flatten(state, start_dim = 1)
        assert network in ["q_network", "target_network"], "Invalid network name"
        if network == 'q_network':
            out = self.q_network(flattened_input)
        else:
            out = self.target_network(flattened_input)
        ### END CODE HERE ###

        return out

    ############################################################
    # Problem 5d: update_target

    def update_target(self):
        """
        The update_target function will be called periodically to copy self.q_network 
        weights to self.target_network.

        TODO: 
            Update the weights for the self.target_network with those of the
            self.q_network. 

        Hint:
            Look up loading pytorch models with load_state_dict function. 
            (https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        """

        ### START CODE HERE ###
        dir = os.path.abspath('.')
        PATH = os.path.join(dir, 'network.pkl')
        torch.save(self.q_network.state_dict(), PATH)
        self.target_network.load_state_dict(torch.load(PATH))
        ### END CODE HERE ###

    ############################################################
    # Problem 5e: calc_loss

    def calc_loss(self, q_values : torch.Tensor, target_q_values : torch.Tensor,
                    actions : torch.Tensor, rewards: torch.Tensor, done_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the MSE loss of a given step. The loss for an example is defined:
            Q_samp(s) = r if done
                        = r + gamma * max_a' Q_target(s', a') otherwise
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')
            
            target_q_values: (torch tensor) shape = (batch_size, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')
            
            actions: (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
            
            rewards: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
            
            done_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state

        TODO:
            Return the MSE loss for a given step. You may use the function description
            for guidance in your implementation.

        Hint:
            You may find the following functions useful
                - torch.max (https://pytorch.org/docs/stable/generated/torch.max.html)
                - torch.sum (https://pytorch.org/docs/stable/generated/torch.sum.html)
                - torch.nn.functional.one_hot (https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html#torch.nn.functional.one_hot)
                - torch.nn.functional.mse_loss (https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html#torch.nn.functional.mse_loss)

            You may need to use the variables:
                - self.config["hyper_params"]["gamma"]
                - self.num_actions
        """
        num_actions = self.env.action_space.n
        gamma =  self.config["hyper_params"]["gamma"]
        ### START CODE HERE ###
        max_target_q_value, _ = torch.max(target_q_values, 1)
        q_samp = rewards + gamma * torch.mul(max_target_q_value, (~done_mask).float())
        # print(max_target_q_value)
        # print(~done_mask)
        # print(torch.mul(max_target_q_value, (~done_mask).float()))
        one_hot_action = F.one_hot(actions.to(torch.int64), num_classes = num_actions)
        q_values_on_action = torch.sum(torch.mul(q_values, one_hot_action), 1)
        # print(q_values_on_action)
        # print(q_values)
        # print(one_hot_action)
        loss = nn.MSELoss()
        output = loss(q_samp, q_values_on_action)
        return output
        ### END CODE HERE ###

    ############################################################
    # Problem 5f: add_optimizer

    def add_optimizer(self):
        """
        This function sets the optimizer for our linear network

        TODO:
            Set self.optimizer to be an Adam optimizer optimizing only the self.q_network
            parameters

        Hint:
            Look up torch.optim.Adam (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)
            What are the input to the optimizer's constructor?
        """
        ### START CODE HERE ###
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-1)
        ### END CODE HERE ###
