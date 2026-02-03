import os
import sys
# Ensure the project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

import numpy as np
import random
import os

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from torch.distributions import Categorical, Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def save_policy_weights(path: str, policy: nn.Module):
    torch.save({k: v.cpu() for k, v in policy.state_dict().items()}, path)

@torch.no_grad()
def load_policy_weights(path: str, policy: nn.Module, alpha: float = 1.0):
    """
    alpha = 1.0 → overwrite
    alpha = 0.0 → do nothing
    """
    state = torch.load(path, map_location="cpu")
    cur = policy.state_dict()

    if alpha == 1.0:
        policy.load_state_dict(state)
        return

    for k in cur:
        if cur[k].is_floating_point():
            cur[k].copy_(cur[k].cpu().lerp(state[k], alpha))
        else:
            cur[k].copy_(state[k])

@torch.no_grad()
def _returns(rew_tb: torch.Tensor, gamma: float):
    # rew_tb: [T,B]
    T, B = rew_tb.shape
    out = torch.zeros_like(rew_tb)
    g = torch.zeros(B, device=rew_tb.device, dtype=rew_tb.dtype)
    for t in reversed(range(T)):
        g = rew_tb[t] + gamma * g
        out[t] = g
    return out


def ppo_update(policy: nn.Module, opt: torch.optim.Optimizer, buf: dict,
               gamma=0.99, clip=0.2, ent=0.01, epochs=4, mb=2048, device="cuda"):
    # buf lists of [B,...] -> stack -> [T,B,...]
    obs   = torch.stack(buf["input"]).to(device)          # [T,B,*]
    act   = torch.stack(buf["action"]).to(device).long()  # [T,B] or [T,B,1]
    oldlg = torch.stack(buf["logits"]).to(device)         # [T,B,A]
    rew = torch.as_tensor(buf["reward"], device=device, dtype=torch.float32)  # [T,B]

    T, B = rew.shape
    N = T * B

    ret = _returns(rew, gamma)          # [T,B]
    adv = (ret - ret.mean()) / (ret.std(unbiased=False) + 1e-8)

    obs   = obs.reshape(N, *obs.shape[2:])
    act   = act.reshape(N)
    oldlg = oldlg.reshape(N, oldlg.shape[-1])
    adv   = adv.reshape(N)

    old_logp = Categorical(logits=oldlg).log_prob(act)

    idx = torch.randperm(N, device=device)
    for _ in range(epochs):
        idx = idx[torch.randperm(N, device=device)]
        for s in range(0, N, mb):
            j = idx[s:s+mb]
            logits, _, _ = policy(obs[j])                 # -> [mb,A]
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[j])

            ratio = (logp - old_logp[j]).exp()
            s1 = ratio * adv[j]
            s2 = ratio.clamp(1-clip, 1+clip) * adv[j]
            loss = -torch.min(s1, s2).mean() - ent * dist.entropy().mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    for k in buf: buf[k].clear()

class TemporalEncoder(nn.Module):
    def __init__(self, feature_dim, conv_out_channels, out_dim):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=conv_out_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=3, padding='same')
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # infer size, it is conv_out_channels * memory size
        self.linear = nn.LazyLinear(out_dim)

    def forward(self, x):
        # buffer: [buffer_size, feature_dim]
        x = x.transpose(1, 2)
        x = self.conv1(x)          # [B, buffer_size, feature_dim]
        x = self.relu(x)
        
        x = self.conv2(x)          # [B, buffer_size, feature_dim]
        x = self.relu(x)
        
        x = self.flatten(x)        # [B, conv_out_channels * buffer_size]
        x = self.linear(x)         # [B, out_dim]
        
        return x


class TrafficLightModelNeuron(nn.Module):
    def __init__(self, params_tls, hidden, num_phases):
        super(TrafficLightModelNeuron, self).__init__()
        self.global_output = 0
        self.hidden = hidden
        self.combined_input = self.hidden + self.global_output
        self.output_possibilities = num_phases

        # Global input -> linear no global for now, its packed in the state
        # Local input x X -> quantizer -> Sequential -> LSTM -> out

        # self.nn_global = nn.Linear(global_params, self.global_output)

        self.quantizer = TemporalEncoder(params_tls, hidden, self.hidden)
        self.nn_tcl = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.LeakyReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.LeakyReLU())
        
        self.head = nn.LSTM(input_size=self.hidden, hidden_size=self.hidden, num_layers=4, batch_first=False, proj_size=self.output_possibilities)

        # memory
        self.lstm_state = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def getActions(self, state_tensor):
        # Get action from the neural network, sequnce is 1, no unroll
        state_tensor = state_tensor.unsqueeze(0)
        decision_logits, self.lstm_state = self.head(state_tensor)

        # Convert logits to decision action
        temperature = 0.1
        # decision_probs = F.softmax(decision_logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(logits=decision_logits / temperature)
        decision_action = dist.sample()
        #  = torch.argmax(decision_probs, dim=-1)

        return decision_logits, decision_action

    def forward(self, x):
        x = self.quantizer(x)
        x = self.nn_tcl(x)


        if self.lstm_state == None:
            batch, _ = x.shape
            self.init_lstm_state(batch)

        decision_logits, decision_action = self.getActions(x)
        return decision_logits, decision_action, self.lstm_state


    def save_model(self, path):
        torch.save(self.state_dict(), f'{path}')
        # shutil.copyfile(f'{prefix}_{name_to_save}.{postfix}', f'{prefix}_{history_prefix}_{name_to_save}_{version}.{postfix}')

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def init_lstm_state(self, B):
        with torch.no_grad():
            hn_c = torch.zeros(B, 1, self.combined_input)
            cn_c = torch.zeros(B, 1, self.combined_input)
            self.lstm_state = (hn_c, cn_c)


    def step(self, total_loss):
        # Backpropagation step
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def choose_action(self, quantized):
        return self(quantized)


    def train(self, unroll_buffer):
        # unroll_buffer = {
        #     'reward':[],
        #     'input':[],
        #     'action':[],
        #     'logits':[],
        #     'lstm_state':[]
        # }

        # add critic here -> much more stable

        ppo_update(self, self.optimizer, unroll_buffer, device=device)



        
        # clear the graph
        self.memory = []

