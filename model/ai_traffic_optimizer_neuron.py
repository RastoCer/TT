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

def get_g_avg(l, pos, width, alpha):
    if not l:
        return 0  # Return 0 for empty list

    half_width = width // 2
    start = max(0, pos - half_width)
    end = min(len(l), pos + half_width + 1)

    weighted_sum = 0.0
    weight_total = 0.0

    for i in range(start, end):
        # Gaussian weight centered at 'pos'
        distance = i - pos
        weight = math.exp(-alpha * (distance ** 2))
        weighted_sum += l[i] * weight
        weight_total += weight

    return weighted_sum / weight_total if weight_total != 0 else 0

def getActionsSingle(net, current_wait_time, state_tensor):
    global_tensor = torch.tensor(current_wait_time, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Get action from the neural network
    decision_logits = net(state_tensor.unsqueeze(0), global_tensor)

    # Convert logits to decision action
    temperature = 0.1
    decision_probs = F.softmax(decision_logits[0] / temperature, dim=0)
    decision_action = torch.argmax(decision_probs).item()

    return decision_logits, decision_action

class TemporalEncoder(nn.Module):
    def __init__(self, feature_dim, conv_out_channels, buffer_size, out_dim):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=conv_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        self.linear = nn.Linear(conv_out_channels * buffer_size, out_dim)

    def forward(self, buffer):
        # buffer: [buffer_size, feature_dim]
        # delete this when batched
        x = buffer.T.unsqueeze(0)  # [1, feature_dim, buffer_size]
        
        x = self.conv1(x)          # [B, conv_out_channels, buffer_size]
        x = self.relu(x)
        
        x = self.conv2(x)          # [B, conv_out_channels, buffer_size]
        x = self.relu(x)
        
        x = self.flatten(x)        # [B, conv_out_channels * buffer_size]
        x = self.linear(x)         # [B, out_dim]
        
        return x


class TrafficLightOptimizerNeuron(nn.Module):
    def __init__(self, params_tls, num_phases):
        super(TrafficLightOptimizerNeuron, self).__init__()
        global_params = 1
        self.global_output = 1
        self.hidden_size_tcl = 10
        self.params_tls = params_tls
        self.combined_input = self.hidden_size_tcl + self.global_output
        self.combined_out = self.combined_input
        self.output_possibilities = num_phases
        self.look_back = 5
        self.look_back_out = 10
        self.fire_tempo = self.look_back - 1
        self.oscilator = 0 

        self.memoryInput = torch.zeros(self.look_back, self.params_tls)
        self.memoryOutput = torch.zeros(self.look_back_out, self.output_possibilities + 1  )
        self.memory = [] # hold outputs and current info

        # Global input -> linear
        # Local input x X -> quantizer -> lstm
        #                              -> LSTM -> linear -> ReLu -> Linear -> ReLu -> output

        self.nn_global = nn.Linear(global_params, self.global_output)

        self.quant_out = self.params_tls
        self.quantizer = TemporalEncoder(params_tls, params_tls, self.look_back, self.quant_out)
        self.nn_tcl = nn.LSTM(input_size=self.quant_out, hidden_size=self.hidden_size_tcl, batch_first=True)
        
        self.nn_combined = nn.LSTM(self.combined_input, hidden_size=self.combined_input, batch_first=True)

        # Fully connected layers
        self.nn_fc_layers = nn.Sequential(
            nn.Linear(self.combined_input, self.combined_out),
            nn.LeakyReLU(),
            nn.Linear(self.combined_out, int(self.combined_out/2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.combined_out/2), int(self.combined_out/2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.combined_out/2), int(self.combined_out/2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.combined_out/2), self.output_possibilities),
            nn.LeakyReLU(),
            nn.Linear(self.output_possibilities, self.output_possibilities)
        )

        # memory
        self.hn_c = None
        self.cn_c = None
        self.hn_tcl = None
        self.cn_tcl = None

        self.init_hidden_c()
        self.init_hidden_tcl()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    # how often to check data
    def tempo(self):
        self.oscilator += 1
        if self.oscilator % self.fire_tempo == 0:
            self.oscilator = 0
            return True
        return False


    def forward(self, in_tcl, in_global):
        in_tcl = in_tcl
        in_global = in_global

        out_g = self.nn_global(in_global)
        out_tcl, (self.hn_tcl, self.cn_tcl) = self.nn_tcl(in_tcl, (self.hn_tcl, self.cn_tcl))

        batch, seq_len, _ = out_tcl.shape
        # adjust global state for the batch of tcl
        out_g = out_g.unsqueeze(0).expand(batch, seq_len, -1)

        in_combined = torch.cat([out_g, out_tcl], dim=-1)

        out_combined, (self.hn_c, self.cn_c) = self.nn_combined(in_combined, (self.hn_c, self.cn_c))

        out = self.nn_fc_layers(out_combined)

        return out

    def save_model(self, path):
        torch.save(self.state_dict(), f'{path}')
        # shutil.copyfile(f'{prefix}_{name_to_save}.{postfix}', f'{prefix}_{history_prefix}_{name_to_save}_{version}.{postfix}')

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def init_hidden_tcl(self):
        with torch.no_grad():
            # 1/1 is batch sizes, adjust when batched
            self.hn_tcl = torch.zeros(1, 1, self.hidden_size_tcl)
            self.cn_tcl = torch.zeros(1, 1, self.hidden_size_tcl)

    def init_hidden_c(self):
        with torch.no_grad():
            self.hn_c = torch.zeros(1, 1, self.combined_input)
            self.cn_c = torch.zeros(1, 1, self.combined_input)


    def getLoss(sefl, loss, decision_logits, decision_action):
        decision_probs = F.softmax(decision_logits[0], dim=0)

        epsilon = 1e-10
        prob = torch.clamp(decision_probs[decision_action], min=epsilon)  # avoid log(0)

        lossAction = -torch.log(prob) * loss  # Simple policy gradient

        return lossAction


    def applyExperience(self, total_loss):
        # Backpropagation step
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_buffer(self, buffer, new_input):
    # Shift buffer: remove first row, append new_input at end
        buffer = torch.cat([buffer[1:], new_input.unsqueeze(0)], dim=0)
        return buffer

    # update all data, ignore globals for now
    def update_data(self, total_num_vehicles, currentWaitTime, state):
        self.memoryInput = self.update_buffer(self.memoryInput, torch.tensor(state, dtype=torch.float32))
        

    def choose_action(self, total_num_vehicles, currentWaitTime):
        state = self.quantizer(self.memoryInput)
        decision_logits, decision_action = getActionsSingle(self, currentWaitTime, state)
        self.memory.append((decision_logits, decision_action, currentWaitTime, total_num_vehicles))
        # self.update_buffer(self.memoryOutput, decision_logits.detach())
        return decision_action


    def applyMemoryUpdates(self, total_num_vehicles):
        baseline = 0
        
        total_loss = []
        for i in range(len(self.memory)):
            if i < self.look_back:
                continue

            decision_logits, decision_action, currentWaitTime, _ = self.memory[i]
            _, _, previous_wait_time, vehicles = self.memory[i - self.look_back]

            improvement = previous_wait_time - currentWaitTime
            baseline = 0.9 * baseline + 0.1 * improvement
            advantage = improvement - baseline

            loss_tensor = self.getLoss(advantage, decision_logits, decision_action)
            print(f"C:{currentWaitTime}, P:{previous_wait_time}, L:{loss_tensor}, D:{decision_action}, A:{advantage}")
            print(decision_action, loss_tensor)

            total_loss.append(loss_tensor)
        total_loss_tensor = torch.stack(total_loss)
        self.applyExperience(total_loss_tensor.mean())
        
        # clear the graph
        self.memory = []

