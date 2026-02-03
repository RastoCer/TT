import os
import sys
import json
import traci
import xml.etree.ElementTree as ET
import socket
import torch
from contextlib import closing
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.ai_traffic_optimizer_neuron import TrafficLightModelNeuron
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


FIXED_LENGTH = 5

def _find_free_port() -> int:
    # Ask OS for a free port
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

class SumoSimulationAPI:
    def __init__(self, config_file, sumo_binary="sumo-gui", label='0', port=None):
        self.config_file = os.path.expanduser(config_file)
        self.sumo_binary = sumo_binary
        self.step_length = 1  # Default to 1 second if not in the config file
        self.simulation_duration = None
        self.running = False
        self.label = label
        self.port = int(port) if port is not None else _find_free_port()
        self.conn = None
        self._parse_config_file()

    def _parse_config_file(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")
        try:
            tree = ET.parse(self.config_file)
            root = tree.getroot()
            time_element = root.find(".//time")
            if time_element is not None:
                self.step_length = float(time_element.get("step-length", self.step_length))
                print(f'self.step_length: {self.step_length}')
                begin = float(time_element.get("begin", 0))
                end = float(time_element.get("end", 60))  # Default to 3600 seconds if no end time is set
                self.simulation_duration = end - begin if end > begin else None
        except ET.ParseError as e:
            raise RuntimeError(f"Failed to parse the configuration file: {e}")

    def start_simulation(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")
        try:
            cmd = [self.sumo_binary, "-c", self.config_file, "--step-length", str(self.step_length)]
            traci.start(cmd, port=self.port, label=self.label)
            self.conn = traci.getConnection(self.label)

            self.running = True
        except Exception as e:
            raise RuntimeError(f"Failed to start SUMO simulation: {e}")

    def stop_simulation(self):
        if self.running:
            try:
                traci.close()
                self.running = False
                self.conn = None
            except Exception as e:
                pass
                # raise RuntimeError(f"Failed to stop SUMO simulation: {e}")

    def simulation_step(self):
        if not self.running:
            raise RuntimeError("Simulation is not running. Call start_simulation() first.")
        try:
            self.conn.simulationStep()
        except Exception as e:
            raise RuntimeError(f"Failed to advance simulation step: {e}")

    def get_traffic_light_ids(self):
        return self.conn.trafficlight.getIDList()

    def set_light(self, traffic_light_id, phase):
        if traffic_light_id not in traci.trafficlight.getIDList():
            raise ValueError(f"Traffic light ID '{traffic_light_id}' does not exist.")
        try:
            self.conn.trafficlight.setPhase(traffic_light_id, phase)
        except Exception as e:
            raise RuntimeError(f"Failed to set phase for traffic light '{traffic_light_id}': {e}")

    def get_average_wait_time(self):
        total_wait_time = 0
        num_vehicles = 0
        for vehicle_id in self.conn.vehicle.getIDList():
            total_wait_time += self.conn.vehicle.getWaitingTime(vehicle_id)
            num_vehicles += 1
        return total_wait_time / num_vehicles if num_vehicles > 0 else 0

    def get_num_vehicles(self):
        return len(self.conn.vehicle.getIDList())

    def get_total_travel_time(self):
        total_travel_time = 0
        for vehicle_id in self.conn.vehicle.getIDList():
            total_travel_time += self.conn.vehicle.getTimeLoss(vehicle_id)
        return total_travel_time
    
    def get_wait_time_tls(self, tl_id):
        wait = 0
        total_lanes = 0
        for t in (self.conn.lane.getTraveltime(lane) for lane in self.conn.trafficlight.getControlledLanes(tl_id)):
            wait += t
            total_lanes += 1
        wait /= total_lanes
        return wait

    def get_state(self, tl_id):
        def fix_length(data, length=FIXED_LENGTH, default=0):
            """Pad or truncate data to ensure a fixed length."""
            return tuple(data[:length]) + (default,) * max(0, length - len(data))
        
        vehicle_numbers = tuple(
            self.conn.lane.getLastStepVehicleNumber(lane) for lane in self.conn.trafficlight.getControlledLanes(tl_id))
        travel_times = tuple(self.conn.lane.getTraveltime(lane) for lane in self.conn.trafficlight.getControlledLanes(tl_id))

        # Adjust to fixed length
        fixed_vehicle_numbers = fix_length(vehicle_numbers, FIXED_LENGTH)
        fixed_travel_times = fix_length(travel_times, FIXED_LENGTH)

        state = fixed_vehicle_numbers + fixed_travel_times + (self.conn.trafficlight.getSpentDuration(tl_id), )
        return state


class BatchedSumoSimulationAPI:
    def __init__(self, config_file, gui=False, batch_size=2):
        self.envs = []
        self.batch_size = int(batch_size)


        sumo_binary0 = "sumo-gui" if gui else "sumo"
        sumo_binary = "sumo"

        for i in range(self.batch_size):
            env = SumoSimulationAPI(
                config_file=config_file,
                sumo_binary=sumo_binary0 if i==0 else sumo_binary, # ran only one env in gui
                label=f"{i}",
                port=_find_free_port()
            )
            self.envs.append(env)

        self.step_length = self.envs[0].step_length

    def start_simulation(self):
        for env in self.envs:
            env.start_simulation()

    def stop_simulation(self):
        for env in self.envs:
            env.stop_simulation()

    def simulation_step(self):
        for env in self.envs:
            env.simulation_step()

    
    def get_traffic_light_ids(self):
        # -> list[list[str]] : one list per env
        return self.envs[0].get_traffic_light_ids()

    def get_average_wait_time(self):
        # -> list[float]
        return [env.get_average_wait_time() for env in self.envs]

    def get_num_vehicles(self):
        # -> list[int]
        return [env.get_num_vehicles() for env in self.envs]

    def get_total_travel_time(self):
        # -> list[float]
        return [env.get_total_travel_time() for env in self.envs]
    
    def set_light(self, traffic_light_id, phases):
        phases = phases.squeeze(0).numpy()
        for env, ph in zip(self.envs, phases):
            env.set_light(traffic_light_id, ph)

    def get_wait_time_tls(self, tl_id):
        # -> list[float]
        return [env.get_wait_time_tls(tl_id) for env in self.envs]
    
    def get_state(self, tl_id):
        # -> list[tuple] : one state tuple per env
        return [env.get_state(tl_id) for env in self.envs]
    
    def _quantizer_to_batch_tensor(self, memory):
            step_feats = []
            for total, wait, avg, states in memory:
                total  = torch.as_tensor(total, dtype=torch.float32, device=device)
                wait   = torch.as_tensor(wait,  dtype=torch.float32, device=device)
                avg    = torch.as_tensor(avg,   dtype=torch.float32, device=device)
                states = torch.as_tensor(states,dtype=torch.float32, device=device)

                # ensure [B,1] for scalars
                if total.ndim == 1: total = total.unsqueeze(-1)
                if wait.ndim  == 1: wait  = wait.unsqueeze(-1)
                if avg.ndim   == 1: avg   = avg.unsqueeze(-1)

                # per-timestep feature vector: [B, X+3]
                feat_t = torch.cat([states, total, wait, avg], dim=-1)
                step_feats.append(feat_t)

            # stack over sequence: list of [B,F] -> [S,B,F] -> [B,S,F]
            x = torch.stack(step_feats, dim=0).transpose(0, 1).contiguous()
            return x

    def run_simulation(self, use_ai, model=None, simulation_time=3600):
        total_steps = int(simulation_time / self.step_length)
        metrics = {
            "average_wait_time": [],
            "total_travel_time": []
        }

        self.start_simulation()

        memory_size = 5
        quantizer_memory = deque(maxlen=memory_size)
        fire_rate = memory_size + 1

        unroll_buffer = {
            'reward':[],
            'input':[],
            'action':[],
            'logits':[],
            'lstm_state':[]
        }

        try:
            for step in range(total_steps):
                self.simulation_step()
                avg_wait_time = self.get_average_wait_time()
                total_num_vehicles = self.get_num_vehicles()
                total_travel_time = self.get_total_travel_time()
                metrics["average_wait_time"].append(avg_wait_time)
                metrics["total_travel_time"].append(total_travel_time)

                if use_ai:
                    # only one TLS
                    traffic_lights = ['cluster_1887857140_2423244102_4176275741_4799917759_#7more']#self.get_traffic_light_ids()
                    for tl_id in traffic_lights:
                        states = self.get_state(tl_id)
#                        print(states)

                        wait = self.get_wait_time_tls(tl_id)
                        quantizer_memory.append((total_num_vehicles, wait, avg_wait_time, states))

                        # we don't need to change tls every second
                        if not step % fire_rate == memory_size:
                            continue
                        
                        quantized = self._quantizer_to_batch_tensor(quantizer_memory)

                        # instead of light, set programs
                        with torch.no_grad():
                            decision_logits, decision_action, lstm_state = model.choose_action(quantized)
                        self.set_light(tl_id, decision_action)

                        unroll_buffer["action"].append(decision_action)
                        unroll_buffer["logits"].append(decision_logits)
                        unroll_buffer["lstm_state"].append(lstm_state)
                        unroll_buffer["reward"].append(avg_wait_time)
                        unroll_buffer["input"].append(quantized)

            model.train(unroll_buffer)

            return metrics
        finally:
            self.stop_simulation()


def save_results(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(results, f)


def visualize_results(ai_metrics):
    # plt.plot(no_ai_metrics["average_wait_time"], label="Without AI")
    plt.plot(ai_metrics["average_wait_time"], label="With AI")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Average Wait Time (s)")
    plt.legend()
    plt.title("Average Wait Time Comparison")
    plt.show()
    plt.savefig('plot.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="data/sumo_network/osm.sumocfg", help="Path to SUMO .sumocfg file")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO-GUI")
    args = parser.parse_args()

    gui = args.gui

    # config_file, sumo_binary="sumo-gui":
    api = BatchedSumoSimulationAPI(config_file=args.config, gui=args.gui)
    # each tls needs its own model!
    # logic = self.envs[0].conn.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)=cluster_1887857140_2423244102_4176275741_4799917759_#7more
    # num_phases = len(logic[0].phases)
    # print(num_phases) = 11
    model = TrafficLightModelNeuron(FIXED_LENGTH*2 + 4, 64, num_phases=11).to(device)
    model.set_optimizer(optim.Adam(model.parameters(), lr=0.0001))
    model_path = "data/models/traffic_light_model.torch"

    # Try loading pre-trained model
    if os.path.exists(model_path):
        model.load_model(model_path)

    # Run simulations for 3600 seconds
    simulation_time = 100  # Run simulation for N seconds
    ai_metrics = api.run_simulation(use_ai=True, model=model, simulation_time=simulation_time)

    model.save_model(model_path)

    # Save results
    save_results(ai_metrics, "data/results/ai_metrics.json")

    # Visualize results
    visualize_results(ai_metrics)
