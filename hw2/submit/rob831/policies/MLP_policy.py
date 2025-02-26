import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy

from rob831.infrastructure.utils import normalize

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from hw1
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        obs_tensor = ptu.from_numpy(observation)
        action_dist = self.forward(obs_tensor)
        action = action_dist.sample()
        
        return ptu.to_numpy(action)
    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # this raise should be left alone as it is a base class for PG
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from hw1
        if self.discrete:
            # for discrete space, take logit and return the most likley action
            logits = self.logits_na(observation)
            return distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(observation)
            std = torch.exp(self.logstd)
            return distributions.Normal(mean, std)
#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: update the policy using policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss
        # HINT4: use self.optimizer to optimize the loss. Remember to
            # 'zero_grad' first


        # Get Log probs 
        pred_acts = self.forward(observations)  # Get the action distribution
        log_probs = pred_acts.log_prob(actions)  # Compute log probabilities of taken actions

        # If multi-dim action, sum over action dimensions 
        if len(log_probs.shape) > 1:
            log_probs = log_probs.sum(dim=-1)  # Log space this is the equivalent of multiplying the independent probs. Ex: prob of action 1 * prob of action 2

        policy_loss = -(log_probs * advantages).mean()  # Compute loss (take mean like in equation)
        self.optimizer.zero_grad()  
        policy_loss.backward()  
        self.optimizer.step() 


        if self.nn_baseline:
            ## TODO: update the neural network baseline using the q_values as
            ## targets. The q_values should first be normalized to have a mean
            ## of zero and a standard deviation of one.

            # in this function we are essentiall trying to predict the q_values (we do this to reduce variance)
            q_values = normalize(q_values, q_values.mean(), q_values.std())
            
            ## HINT1: use self.baseline_optimizer to optimize the loss used for
                ## updating the baseline. Remember to 'zero_grad' first
            
            ## HINT2: You will need to convert the targets into a tensor using
                ## ptu.from_numpy before using it in the loss
            q_val_tensor = ptu.from_numpy(q_values)
            baseline_pred = self.baseline(observations)
            baseline_pred = baseline_pred.reshape(-1) # make sure the tensors are same shape
            base_loss = self.baseline_loss(baseline_pred, q_val_tensor)

            # Update baseline using baseline loss 
            self.baseline_optimizer.zero_grad()
            base_loss.backward()
            self.baseline_optimizer.step()

        train_log = {
            'Training Loss': ptu.to_numpy(policy_loss),
        }
        return train_log

    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())
