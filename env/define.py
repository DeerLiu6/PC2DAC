import numpy as np
import random
import logging
import math
from random import randint
from env import traffic

logging.basicConfig(level=logging.WARNING)

class Action(object):
    def __init__(self):
        self.move = None
        self.offload = []
collecting_channel_param = {'suburban': (4.88, 0.43, 0.1, 21),
                            'urban': (9.61, 0.16, 1, 20),
                            'dense-urban': (12.08, 0.11, 1.6, 23),
                            'high-rise-urban': (27.23, 0.08, 2.3, 34)}
collecting_params = collecting_channel_param['urban']
a = collecting_params[0]
b = collecting_params[1]
yita0 = collecting_params[2]
yita1 = collecting_params[3]
carrier_f = 2.5e9

class Sensor(object):
    sensor_count = 0
    def __init__(self, position, sensor_move_r):
        self.no = Sensor.sensor_count
        Sensor.sensor_count += 1
        self.position = position
        self.computing_rate = 100
        self.sensor_move_r = sensor_move_r
        self.action = Action()

        self.sensor_data_gen_rate = 1
        self.gen_threshold = 0.3
        self.total_data = {}
        self.lam = 1e3
        self.sensor_max_data_size = 2e3
        self.d2d_count = 0
        self.d2d_flag = False
        self.ptr = 0.2
        self.h = 5
        self.noise = 1e-13
        self.sensor_bandwidth = 1e3
        self.noise_power = 1e-13 * self.sensor_bandwidth

class EdgeServer(object):
    server_count = 0
    def __init__(self, pos, server_collect_r):
        self.no = EdgeServer.server_count
        EdgeServer.server_count += 1
        self.position = pos
        self.computing_rate = 1000
        self.server_collect_r = server_collect_r

        # self.total_data = {}

class EdgeUav(object):
    edge_count = 0
    def __init__(self, pos, uav_obs_r, uav_collect_r, uav_move_r):
        self.no = EdgeUav.edge_count
        EdgeUav.edge_count += 1
        self.position = pos
        self.computing_rate = 500
        self.action = Action()

        self.uav_obs_r = uav_obs_r
        self.uav_collect_r = uav_collect_r
        self.uav_move_r = uav_move_r

        self.position_x = []
        self.position_y = []
        self.position_x_first = []
        self.position_y_first = []
        self.position_x_last = []
        self.position_y_last = []

        self.h = 5
        self.ptr_col = 0.2

        # self.total_data = {}
        # self.state = AgentState()

class MEC_world(object):
    def __init__(self, map_size, uav_num, server_num, sensor_num, uav_obs_r, uav_collect_r, server_collect_r, uav_move_r, sensor_move_r):
        self.map_size = map_size
        self.DS_state = np.ones([map_size, map_size, 2])
        self.uavs = []
        self.servers = []
        self.sensors = []
        self.sensor_count = sensor_num
        self.server_count = server_num
        self.uav_count = uav_num
        self.uav_obs_r = uav_obs_r
        self.uav_collect_r = uav_collect_r
        self.server_collect_r = server_collect_r
        self.uav_move_r = uav_move_r
        self.sensor_move_r = sensor_move_r

        self.sensor_delay = []
        self.sensor_energy = []

        self.all_sensors_age = 0
        self.max_sensors_age = 0
        self.d2d_num = []
        self.sensor_position = [random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num), random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num)]
        for i in range(sensor_num):
            self.sensors.append(Sensor(np.array([self.sensor_position[0][i], self.sensor_position[1][i]]), self.sensor_move_r))
        self.server_position = [(map_size / 4, map_size / 4, 200 - map_size / 4, 200 - map_size / 4),(map_size / 4, 200 - map_size / 4, map_size / 4, 200 - map_size / 4)]
        for i in range(server_num):
            self.servers.append(EdgeServer(np.array([int(self.server_position[0][i]), int(self.server_position[1][i])]), server_collect_r))
        self.uav_position = [random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], uav_num), random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], uav_num)]
        for i in range(uav_num):
            self.uavs.append(EdgeUav(np.array([self.uav_position[0][i], self.uav_position[1][i]]), self.uav_obs_r, uav_collect_r, self.uav_move_r))

    def step(self):
        self.sensor_delay = []
        self.sensor_energy = []
        for i, sensor in enumerate(self.sensors):

            if (sum(sensor.action.offload) and sensor.total_data != {}):
                data_size = 0
                position_offload = [0,0]
                for g in sensor.total_data:
                    data_size += sensor.total_data[g]
                position_index = sensor.action.offload.index(1)
                if position_index == 0: 
                    self.sensor_delay.append(data_size / sensor.computing_rate)
                    energy = self.mobile_device_energy(dist=0, offloading_index=0, data_size=data_size, rho=1, kappa=1e-28 )
                    self.sensor_energy.append(energy)
                    sensor.total_data = {}
                    continue
                elif position_index == 1: 
                    position_offload = [50,50]
                    if self.is_point_in_circle(sensor.position, position_offload, self.server_collect_r):
                        self.server_transmit_and_process(sensor, data_size, position_offload, 0)
                        continue
                elif position_index == 2:
                    position_offload = [50,150]
                    if self.is_point_in_circle(sensor.position, position_offload, self.server_collect_r):
                        self.server_transmit_and_process(sensor, data_size, position_offload, 1)
                        continue
                elif position_index == 3:
                    position_offload = [150,50]
                    if self.is_point_in_circle(sensor.position, position_offload, self.server_collect_r):
                        self.server_transmit_and_process(sensor, data_size, position_offload, 2)
                        continue
                elif position_index == 4:
                    position_offload = [150,150]
                    if self.is_point_in_circle(sensor.position, position_offload, self.server_collect_r):
                        self.server_transmit_and_process(sensor, data_size, position_offload, 3)
                        continue
                elif position_index == 5:
                    position_offload = self.uavs[0].position
                    if self.is_point_in_circle(sensor.position, position_offload, self.uav_collect_r) and sensor.total_data:
                        self.uav_collect_and_process(sensor, data_size, position_offload, 0)
                        continue
                elif position_index == 6:
                    position_offload = self.uavs[1].position
                    if self.is_point_in_circle(sensor.position, position_offload, self.uav_collect_r) and sensor.total_data:
                        self.uav_collect_and_process(sensor, data_size, position_offload, 1)
                        continue
                elif position_index == 7:
                    position_offload = self.uavs[2].position
                    if self.is_point_in_circle(sensor.position, position_offload, self.uav_collect_r) and sensor.total_data:
                        self.uav_collect_and_process(sensor, data_size, position_offload, 2)
                        continue
                elif position_index == 8:
                    position_offload = self.uavs[3].position
                    if self.is_point_in_circle(sensor.position, position_offload, self.uav_collect_r) and sensor.total_data:
                        self.uav_collect_and_process(sensor, data_size, position_offload, 3)
                        continue
                elif position_index == 9:
                    if len(sensor.total_data) > 20:
                        closest_sensor = None
                        closest_distance = float('inf')
                        for j, other_sensor in enumerate(self.sensors):
                            if i != j:
                                distance = np.linalg.norm(np.array(sensor.position) - np.array(other_sensor.position))
                                if distance < closest_distance:
                                    closest_distance = distance
                                    closest_sensor = other_sensor
                        if closest_sensor is not None and self.is_point_in_circle(sensor.position, closest_sensor.position, self.sensor_move_r):
                            if (data_size > 1800):
                                self.sensor_delay.append(2)
                            else:
                                dist = np.linalg.norm(np.array(sensor.position) - np.array(closest_sensor.position))
                                transmit_or_collect_delay = data_size / self.transmit_rate(dist, sensor)
                                sensor_or_uav_process_delay = data_size / closest_sensor.computing_rate
                                self.sensor_delay.append(transmit_or_collect_delay + sensor_or_uav_process_delay)
                                energy = self.mobile_device_energy(dist, closest_sensor, data_size)
                                self.sensor_energy.append(energy)
                            sensor.total_data = {}
                            sensor.d2d_flag = True
                            continue
                self.sensor_delay.append(0)
                self.sensor_energy.append(0)

        PF = 1 / 2
        self.sensor_delay = list(map(lambda x: 1 / (PF*x) if x != 0 else 0, self.sensor_delay))
        self.sensor_energy = list(map(lambda x: 1 / (PF*x) if x != 0 else 0, self.sensor_energy))
        for i in range(len(self.sensor_delay)):
            if self.sensor_delay[i] > 1:
                self.sensor_delay[i] = 1
            self.sensor_delay[i] = round(self.sensor_delay[i], 3)

        for sensor in self.sensors:
            self.all_sensors_age = self.all_sensors_age + len(sensor.total_data)
            if len(sensor.total_data) > self.max_sensors_age:
                self.max_sensors_age = len(sensor.total_data)

        self.DS_state = np.ones([self.map_size, self.map_size, 2])
        
        for i, sensor in enumerate(self.sensors):
            new_data = sensor.sensor_data_gen_rate * np.random.poisson(sensor.lam)
            new_data = min(new_data, sensor.sensor_max_data_size)

            # if new_data >= self.sensor_max_data_size or random.random() >= self.gen_threshold:
            #     return
            if new_data:
                if sensor.total_data:
                    last_key = list(sensor.total_data.keys())[-1]
                    sensor.total_data[last_key+1] = new_data
                else:
                    sensor.total_data[0] = new_data
            count = 0
            move_dict = {}
            for x in range(-self.sensor_move_r, self.sensor_move_r + 1):
                y_l = int(np.floor(np.sqrt(self.sensor_move_r**2 - x**2)))
                for y in range(-y_l, y_l + 1):
                    move_dict[count] = np.array([y, x])
                    count += 1

            move = random.sample(list(move_dict.values()), 1)[0]
            if traffic.traffic_move(sensor, move, i) :

                if self.if_sensor_within_range(sensor):
                    self.DS_state[sensor.position[1], sensor.position[0]] = [sum(sensor.total_data.values()), len(sensor.total_data)]
                continue
        # for server in self.servers:
        #     self.DS_state[server.position[1], server.position[0]] = [2,0]
        # for uav in self.uavs:
        #     self.DS_state[uav.position[1], uav.position[0]] = [3,0]

    def is_point_in_circle(self, point, circle_center, radius):
        distance = np.sqrt((circle_center[0] - point[0])**2 + (circle_center[1] - point[1])**2)
        return distance <= radius

    def server_transmit_and_process(self, sensor, data_size, position_offload, server_num):

        if(data_size >1800):
            self.sensor_delay.append(2)
        else :

            dist = np.linalg.norm(np.array(sensor.position) - np.array(position_offload))
            transmit_or_collect_delay = data_size / self.transmit_rate(dist, sensor)
            server_or_uav_process_delay = data_size / self.servers[server_num].computing_rate
            self.sensor_delay.append(transmit_or_collect_delay + server_or_uav_process_delay)
            energy = self.mobile_device_energy(dist, server_num, data_size)
            self.sensor_energy.append(energy)
        sensor.total_data = {}

    def uav_collect_and_process(self, sensor, data_size, position_offload, uav_num):
        if(data_size >1800):
            self.sensor_delay.append(2)
        else:
            dist = np.linalg.norm(np.array(sensor.position) - np.array(position_offload))

            transmit_or_collect_delay = data_size / self.transmit_rate(dist, sensor)
            server_or_uav_process_delay = data_size / self.uavs[uav_num].computing_rate
            self.sensor_delay.append(transmit_or_collect_delay + server_or_uav_process_delay)
            energy = self.mobile_device_energy(dist, uav_num, data_size)
            self.sensor_energy.append(energy)
        sensor.total_data = {}

    def if_sensor_within_range(self, sensor): 
        points = [
            sensor.position,
            self.servers[0].position,
            self.servers[1].position,
            self.servers[2].position,
            self.servers[3].position,
            self.uavs[0].position,
            self.uavs[1].position,
            self.uavs[2].position,
            self.uavs[3].position,
        ]
        distances = []
        for point in points:
            distance = np.linalg.norm(np.array(points[0]) - np.array(point))
            distances.append(distance)
        if (
            distances[1] > 30 and
            distances[2] > 30 and
            distances[3] > 30 and
            distances[4] > 30 and
            distances[5] > 40 and
            distances[6] > 40 and
            distances[7] > 40 and
            distances[8] > 40
        ):
            return True
        return False
    

    def transmit_rate(self, dist, sensor):

        if (dist == 0):
            dist = 1
        W = 1e6 * sensor.sensor_bandwidth
        Pl = 1 / (1 + a * np.exp(-b * (np.arctan(sensor.h / dist) - a)))
        fspl = (4 * np.pi * carrier_f * dist / (3e8))**2
        L = Pl * fspl * 10**(yita0 / 20) + 10**(yita1 / 20) * fspl * (1 - Pl)
        transmit_rate = W * np.log2(1 + sensor.ptr / (L * sensor.noise * W))
        return transmit_rate/100
    
    def mobile_device_energy(self, dist, offloading_index, data_size, rho=1, kappa=1e-28):
        if offloading_index == 0:
            l_self = data_size / dist
            e_self = rho * kappa * (dist ** 3) * l_self
            return e_self
        elif offloading_index <= 5:
            l_mec =  data_size / self.transmit_rate(self, dist, offloading_index) + data_size / EdgeServer.computing_rate
            e_mec = 0.15 * l_mec + rho * kappa * (EdgeServer.computing_rate ** 3) * l_mec
            return e_mec
        elif offloading_index <= 8:
            l_uav = data_size / self.transmit_rate(self, dist, offloading_index)  + data_size / EdgeUav.computing_rate
            e_uav = 0.15 * l_uav + rho * kappa * (EdgeUav.computing_rate ** 3) * l_uav
            return e_uav
        elif offloading_index == 9:
            l_d2d = data_size / self.transmit_rate(self, dist, offloading_index)  + data_size / dist
            e_d2d = 0.1 * l_d2d + rho * kappa * ( dist ** 3) * l_d2d
            return e_d2d
        else:
            raise ValueError("Invalid offloading_index provided.")

    # def collect_rate(dist, sensor, uav):
    #     Pl = 1 / (1 + a * np.exp(-b * (np.arctan(uav.h / dist) - a)))
    #     L = Pl * yita0 + yita1 * (1 - Pl)
    #     gamma = uav.ptr_col / (L * sensor.noise_power**2)
    #     collect_rate = sensor.sensor_bandwidth * np.log2(1 + gamma)
    #     collect_rate = 8000
    #     return collect_rate
