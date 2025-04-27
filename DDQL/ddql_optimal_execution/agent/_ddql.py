from ._agent import Agent
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ._neural_net import QNet
from ddql_optimal_execution import State, get_device


class DDQL(Agent):
    """
    The DDQL class inherits from the Agent class. It is an agent that implements a Double Deep Q-Learning algorithm.

    Parameters
    ----------
    state_dict : dict, optional
        A dictionary containing the state of the agent, by default None
    greedy_decay_rate : float, optional
        The greedy decay rate, by default 0.95
    target_update_rate : int, optional
        The target update rate, by default 15
    initial_greediness : float, optional
        The initial greediness, by default 1
    mode : str, optional
        The mode, by default "train"
    lr : float, optional
        The learning rate, by default 1e-3
    state_size : int, optional
        The state size, by default 5
    initial_budget : int, optional
        The initial budget, by default 100
    horizon : int, optional
        The horizon, by default 100
    gamma : float, optional
        The gamma parameter used in the Q-Learning algorithm, by default 0.99
    quadratic_penalty_coefficient : float, optional
        The quadratic penalty coefficient used to penalize the agent for selling big quantities of stocks, by default 0.01


    Attributes
    ----------
    device : torch.device
        The device used to run the agent
    main_net : QNet
        The main neural network used to predict the Q-values of the state-action pairs
    target_net : QNet
        The target neural network used to predict the Q-values of the state-action pairs
    state_size : int
        The state size
    greedy_decay_rate : float
        The greedy decay rate
    target_update_rate : int
        The target update rate
    initial_greediness : float
        The initial greediness of the agent. It is used to determine the probability of the agent taking a random action.
    greediness : float
        The current greediness of the agent.
    mode : str
        The mode of the agent. It can be either "train" or "test".
    lr : float
        The learning rate used to update the weights of the neural network.
    gamma : float
        The gamma parameter used in the Q-Learning algorithm.
    quadratic_penalty_coefficient : float
        The quadratic penalty coefficient used to penalize the agent for selling big quantities of stocks.
    optimizer : torch.optim
        The optimizer used to update the weights of the neural network.
    loss_fn : torch.nn
        The loss function used to calculate the loss between the predicted Q-values and the target Q-values.


    """

    def __init__(
        self,
        state_dict: Optional[dict] = None,
        greedy_decay_rate: float = 0.95,
        target_update_rate: int = 15,
        initial_greediness: float = 1,
        mode: str = "train",
        lr: float = 1e-3,
        state_size: int = 5,
        initial_budget: int = 100,
        horizon: int = 100,
        gamma: float = 0.99,
        quadratic_penalty_coefficient: float = 0.01,
        verbose : bool = False
    ) -> None:
        super().__init__(initial_budget, horizon)

        self.device = get_device()
        print(f"Using {self.device} device")

        self.main_net = QNet(state_size=state_size, action_size=initial_budget).to(
            self.device
        )
        self.target_net = QNet(state_size=state_size, action_size=initial_budget).to(
            self.device
        )

        self.state_size = state_size
        self.initial_budget=initial_budget
        self.gamma = gamma

        if state_dict is not None:
            self.main_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)

        self.greedy_decay_rate = greedy_decay_rate
        self.target_update_rate = target_update_rate
        self.greediness = initial_greediness
        self.quadratic_penalty_coefficient = quadratic_penalty_coefficient

        self.mode = mode

        self.learning_step = 0

        if self.mode == "train":
            self.optimizer = optim.RMSprop(self.main_net.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()

        self.verbose = verbose

        self.dataloaders=[]

    def train(self) -> None:
        """This function sets the mode to "train" and trains the main neural network."""

        self.main_net.train()
        self.mode = "train"

    def eval(self) -> None:
        """This function sets the mode to "eval" and puts the main network in evaluation mode."""

        self.main_net.eval()
        self.mode = "eval"

    def get_action(self, state: State) -> int:
        """This function returns a tensor that is either a random binomial distribution or the index of the
        maximum value in the output of a neural network, depending on certain conditions.

        Parameters
        ----------
        state : State
            The `state` parameter is an instance of the `State` class, which contains information about the
        current state of the environment in which the agent is operating. This information typically
        includes things like the agent's current position, the state of the game board, and any other
        relevant information that the agent needs

        Returns
        -------
            an integer that represents the action to be taken based on the given state. If the `greediness`
        parameter is set and the `mode` is "train", a random binomial distribution is generated using the
        state's inventory as the number of trials and the probability of success as 1/inventory. Otherwise,
        the action is determined by the main neural network's output, which is the index of the maximum
        value in the output Q-values tensor.

        """
        
        #q_values=self.main_net(state)

        q_values=[]
        for action in range (self.initial_budget):
            input=torch.cat((state.astensor, torch.tensor([action])), dim=0)
            q_values.append(self.main_net(input))
        q_values=torch.tensor(q_values)

        return (
            #np.random.binomial(state["inventory"], 1 / state["inventory"])
            np.random.binomial(state["inventory"], 1 / (self.horizon-state["period"]))
            if np.random.rand() < self.greediness and self.mode == "train"
            else q_values.argmax().item()
        )

    def __update_target_net(self) -> None:
        """This function updates the target network by loading the state dictionary of the main network."""
        self.target_net.load_state_dict(self.main_net.state_dict())

    def __complete_target(
        self, experience_batch: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This function takes in a batch of experiences and returns the corresponding targets, actions, and
        states for training a reinforcement learning agent.

        Parameters
        ----------
        experience_batch : np.ndarray
            `experience_batch` is a numpy array containing a batch of experiences. Each experience is a
        dictionary containing information about a single step taken by the agent in the environment. The
        dictionary contains keys such as "state", "action", "reward", "next_state", and "dist2Horizon".

        Returns
        -------
            a tuple of three torch Tensors: targets, actions, and states.

        """
        targets, actions, states = (
            torch.empty(len(experience_batch)),
            torch.empty(len(experience_batch)),
            torch.empty((len(experience_batch), self.state_size)),
        )
        for i, experience in enumerate(experience_batch):  # can be vectorized
            actions[i] = experience["action"]
            states[i] = experience["state"].astensor
            if experience["dist2Horizon"] == 0:
                targets[i] = experience["reward"]

            elif experience["dist2Horizon"] == 1:
                targets[i] = (
                    experience["reward"]
                    + self.gamma
                    * experience["next_state"]["inventory"]*(
                        experience["next_state"]["Price"] - experience["state"]["Price"]
                    )
                    - self.quadratic_penalty_coefficient
                    * (experience["next_state"]["inventory"]) ** 2
                )
            else:
                
                #q_values=self.main_net(experience["next_state"])

                q_values=[]

                for action in range (self.initial_budget):
                    input=torch.cat((experience["next_state"].astensor, torch.tensor([action])), dim=0)
                    q_values.append(self.main_net(input))

                q_values=torch.tensor(q_values)
   
                best_action = q_values.argmax().item()
                # targets[i] = (
                #      experience["reward"]
                #      + self.gamma
                #      * self.target_net(experience["next_state"])[int(best_action)]
                #  )
                
                input=torch.cat((experience["next_state"].astensor, torch.tensor([best_action])), dim=0)
                targets[i] = (
                     experience["reward"]
                     + self.gamma
                     * self.target_net(input)
                )

        return targets, actions, states

    def learn(self, experience_batch: np.ndarray) -> None:
        """This function trains a neural network using a batch of experiences and updates the target network
        periodically.

        Parameters
        ----------
        experience_batch : np.ndarray
            The experience_batch parameter is a numpy array containing a batch of experiences, where each
        experience is a tuple of (state, action, reward, next_state, dist2Horizon). This batch is used to update the
        neural network's weights through backpropagation.

        """

        targets, actions, states = self.__complete_target(experience_batch)
        dataloader = DataLoader(
            TensorDataset(states, actions, targets),
            batch_size=32,
            shuffle=True,
        )
        self.dataloaders.append(dataloader)

        for batch in dataloader:
            target = batch[2]
            state=batch[0]
            action=batch[1]
            # inventory_action=inventory_action_transformer(torch.stack((state[:,-1],action), dim=1))
            # inventory_tr=torch.tensor(inventory_action[:,0])
            # action_tr=torch.tensor(inventory_action[:,1])
            # action=action_tr
            # state[:,-1]=inventory_tr
            
            
            #pred = self.main_net(batch[0])[torch.arange(len(batch[0])), batch[1].long()]

            input=torch.cat((state, action.unsqueeze(1)), dim=1)
            pred = self.main_net(input)
    
            loss = self.loss_fn(pred.squeeze(1), target)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        self.learning_step += 1
        self.greediness = max(0.01, self.greediness * self.greedy_decay_rate)
        if self.learning_step % self.target_update_rate == 0 and self.verbose==True:
            self.__update_target_net()
            print(
                f"Target network updated at step {self.learning_step} with greediness {self.greediness:.2f}"
            )
