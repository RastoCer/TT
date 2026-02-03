import os
import sys
import json
import traci
import xml.etree.ElementTree as ET
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.ai_traffic_optimizer_neuron import TrafficLightOptimizerNeuron
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse

FIXED_LENGTH = 5

class SumoSimulationAPI:
    def __init__(self, config_file, sumo_binary="sumo-gui"):
        self.config_file = os.path.expanduser(config_file)
        self.sumo_binary = sumo_binary
        self.step_length = 1  # Default to 1 second if not in the config file
        self.simulation_duration = None
        self.running = False
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
            traci.start([self.sumo_binary, "-c", self.config_file, "--step-length", str(self.step_length)])
            self.running = True
        except Exception as e:
            raise RuntimeError(f"Failed to start SUMO simulation: {e}")

    def stop_simulation(self):
        if self.running:
            try:
                traci.close()
                self.running = False
            except Exception as e:
                raise RuntimeError(f"Failed to stop SUMO simulation: {e}")

    def simulation_step(self):
        if not self.running:
            raise RuntimeError("Simulation is not running. Call start_simulation() first.")
        try:
            traci.simulationStep()
        except Exception as e:
            raise RuntimeError(f"Failed to advance simulation step: {e}")

    def get_traffic_light_ids(self):
        return traci.trafficlight.getIDList()

    def set_light(self, traffic_light_id, phase):
        if traffic_light_id not in traci.trafficlight.getIDList():
            raise ValueError(f"Traffic light ID '{traffic_light_id}' does not exist.")
        try:
            yy = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
            if yy != 'GGGgGrrrrrrr':
                return
            traci.trafficlight.setPhase(traffic_light_id, phase)
        except Exception as e:
            raise RuntimeError(f"Failed to set phase for traffic light '{traffic_light_id}': {e}")

    def get_average_wait_time(self):
        total_wait_time = 0
        num_vehicles = 0
        for vehicle_id in traci.vehicle.getIDList():
            total_wait_time += traci.vehicle.getWaitingTime(vehicle_id)
            num_vehicles += 1
        return total_wait_time / num_vehicles if num_vehicles > 0 else 0

    def get_num_vehicles(self):
        return len(traci.vehicle.getIDList())

    def get_total_travel_time(self):
        total_travel_time = 0
        for vehicle_id in traci.vehicle.getIDList():
            total_travel_time += traci.vehicle.getTimeLoss(vehicle_id)
        return total_travel_time

def get_wait_time_tls(tl_id):
    wait = 0
    total_lanes = 0
    for t in (traci.lane.getTraveltime(lane) for lane in traci.trafficlight.getControlledLanes(tl_id)):
        wait += t
        total_lanes += 1
    wait /= total_lanes
    return wait

def get_state(tl_id):
    def fix_length(data, length=FIXED_LENGTH, default=0):
        """Pad or truncate data to ensure a fixed length."""
        return tuple(data[:length]) + (default,) * max(0, length - len(data))

    vehicle_numbers = tuple(
        traci.lane.getLastStepVehicleNumber(lane) for lane in traci.trafficlight.getControlledLanes(tl_id))
    travel_times = tuple(traci.lane.getTraveltime(lane) for lane in traci.trafficlight.getControlledLanes(tl_id))

    # Adjust to fixed length
    fixed_vehicle_numbers = fix_length(vehicle_numbers, FIXED_LENGTH)
    fixed_travel_times = fix_length(travel_times, FIXED_LENGTH)

    state = fixed_vehicle_numbers + fixed_travel_times + (traci.trafficlight.getSpentDuration(tl_id), )
    return state


def run_simulation(use_ai, api, optimizer=None, simulation_time=3600):
    total_steps = int(simulation_time / api.step_length)
    metrics = {
        "average_wait_time": [],
        "total_travel_time": []
    }

    api.start_simulation()
    try:
        for step in range(total_steps):
            api.simulation_step()
            avg_wait_time = api.get_average_wait_time()
            total_num_vehicles = api.get_num_vehicles()
            total_travel_time = api.get_total_travel_time()
            metrics["average_wait_time"].append(avg_wait_time)
            metrics["total_travel_time"].append(total_travel_time)

            if use_ai:
                traffic_lights = api.get_traffic_light_ids()
                for tl_id in traffic_lights:
                    state = get_state(tl_id)
                    print(state)

                    wait = get_wait_time_tls(tl_id)

                    optimizer.update_data(total_num_vehicles, wait+avg_wait_time, state)

                    # we don't need to get new state every second
                    if not optimizer.tempo():
                        continue

                    action = optimizer.choose_action(total_num_vehicles, wait+avg_wait_time)
                    api.set_light(tl_id, action)

        optimizer.applyMemoryUpdates(total_num_vehicles)

        return metrics
    finally:
        api.stop_simulation()


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

    sumo_binary = "sumo-gui" if args.gui else "sumo"

    # config_file, sumo_binary="sumo-gui":
    api = SumoSimulationAPI(config_file=args.config, sumo_binary=sumo_binary)
    optimizer = TrafficLightOptimizerNeuron(FIXED_LENGTH*2 + 1, num_phases=4)
    optimizer.set_optimizer(optim.Adam(optimizer.parameters(), lr=0.001))
    model_path = "data/models/traffic_light_model.npy"
    model_path = "data/models/traffic_light_model12.torch"

    # Try loading pre-trained model
    if os.path.exists(model_path):
        optimizer.load_model(model_path)
    optimizer.train()

    #no_ai_metrics=None
    # Run simulations for 3600 seconds
    simulation_time = 900  # Run simulation for 3600 seconds (1 hour)
    # no_ai_metrics = run_simulation(use_ai=False, api=api, simulation_time=simulation_time)
    ai_metrics = run_simulation(use_ai=True, api=api, optimizer=optimizer, simulation_time=simulation_time)

    optimizer.save_model(model_path)

    # Save results
    # save_results(no_ai_metrics, "data/results/no_ai_metrics.json")
    save_results(ai_metrics, "data/results/ai_metrics.json")

    # Visualize results
    visualize_results(ai_metrics)
