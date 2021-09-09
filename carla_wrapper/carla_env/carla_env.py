import glob
import os
import random
import sys
import time
from typing_extensions import final

import gym
import pygame
import numpy as np
import math

from gym import spaces
from carla_env.carla_sync_mode import CarlaSyncMode
from carla_env.carla_weather import Weather
from agents.navigation.roaming_agent import RoamingAgent

from carla_env.speedlimit_autopilot import SpeedAutopilot

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


class CarlaEnv(gym.Env):

    def __init__(self,
                 render,
                 carla_port,
                 changing_weather_speed,
                 frame_skip,
                 observations_type,
                 traffic,
                 vehicle_name,
                 map_name,
                 autopilot):

        super(CarlaEnv, self).__init__()
        self.render_display = render
        self.changing_weather_speed = float(changing_weather_speed)
        self.frame_skip = frame_skip
        self.observations_type = observations_type
        self.traffic = traffic
        self.vehicle_name = vehicle_name
        self.map_name = map_name
        self.autopilot = autopilot
        self.actor_list = []
        self.count = 0

        #self.mean = 0.0 
        #self.std = 0.8

        self.distance_to_centerline_list = [0]
        self.heading_list = [0]
        self.steering_list = [0]
        self.display_image_list = []

        # initialize rendering
        if self.render_display:
            pygame.init()
            self.render_display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()
            self.clock = pygame.time.Clock()

        # initialize client with timeout
        self.client = carla.Client('localhost', carla_port)
        self.client.set_timeout(4.0)

        # initialize world and map
        self.world = self.client.load_world(self.map_name)
        self.map = self.world.get_map()

        # remove old vehicles and sensors (in case they survived)
        self.world.tick()
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter('*vehicle*'):
            print('Warning: removing old vehicle')
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print('Warning: removing old sensor')
            sensor.destroy()

        # create vehicle
        self.vehicle = None
        self.vehicles_list = []
        self._reset_vehicle()
        self.actor_list.append(self.vehicle)

        # initialize blueprint library
        blueprint_library = self.world.get_blueprint_library()

        # spawn camera for rendering
        if self.render_display:
            self.camera_display = self.world.spawn_actor(
                blueprint_library.find('sensor.camera.rgb'),
                carla.Transform(carla.Location(x=-6.5, z=3.8), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
            self.actor_list.append(self.camera_display)

        # spawn camera for pixel observations
        if self.observations_type == 'pixel':
            bp = blueprint_library.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(84))
            bp.set_attribute('image_size_y', str(84))
            bp.set_attribute('fov', str(84))
            location = carla.Location(x=3, y=0, z=2) #(3,0,2)
            self.camera_vision = self.world.spawn_actor(
                bp, carla.Transform(location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)
            self.actor_list.append(self.camera_vision)

        # context manager initialization
        if self.render_display and self.observations_type == 'pixel':
            self.sync_mode = CarlaSyncMode(self.world, self.camera_display, self.camera_vision, fps=20)
        elif self.render_display and self.observations_type == 'state':
            self.sync_mode = CarlaSyncMode(self.world, self.camera_display, fps=20)
        elif not self.render_display and self.observations_type == 'pixel':
            self.sync_mode = CarlaSyncMode(self.world, self.camera_vision, fps=20)
        elif not self.render_display and self.observations_type == 'state':
            self.sync_mode = CarlaSyncMode(self.world, fps=20)
        else:
            raise ValueError('Unknown observation_type. Choose between: state, pixel')

        # weather
        #self.world.set_weather(carla.WeatherParameters.ClearSunset)
        self.weather = Weather(self.world, self.changing_weather_speed) #no weather changes


        # collision detection
        self.collision = False
        sensor_blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(sensor_blueprint, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # initialize autopilot
        self.agent = RoamingAgent(self.vehicle)
        #self.agent = SpeedAutopilot(self.vehicle)

        # get initial observation
        if self.observations_type == 'state':
            obs = self._get_state_obs()
        else:
            obs = np.zeros((84, 84, 3))

        # gym environment specific variables
        #np.array([0, -1, 0]), np.array([+1, +1, +1])
        self.action_space = spaces.Box(-1. , 1., shape=(2,), dtype='float32')
        #self.action_space = spaces.Box(np.array([-0.25 , -1.]), np.array([0.25, 1.]), shape=(2,), dtype='float32')
        self.obs_dim = obs.shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.obs_dim, dtype=np.uint8)

    def reset(self):
        self._reset_vehicle()
        self.world.tick()
        self._reset_other_vehicles()
        self.world.tick()
        self.count = 0
        self.collision = False
        obs, _, _, _ = self.step([0, 0])
        print('RESETED')
        return obs

    def _reset_vehicle(self):
        # choose random spawn point from self selected spawn points
        spawn_points = []
        spawn_points.append(carla.Transform(carla.Location(x=4.3, y=-157.2, z= 0.2), carla.Rotation(0, -90, 0))) #starting point dataset FUNKTIONIERT !!!!!!!!
        #spawn_points.append(carla.Transform(carla.Location(x=262.2, y=41.8, z= 0.2), carla.Rotation(0, 0, 0))) #starting point dataset Town06
        #spawn_points.append(carla.Transform(carla.Location(x=63.5, y=-342.4, z= 0.2), carla.Rotation(0, -37.7, 0))) #here FUNKTIONIERT!!!!!!!!!
        #spawn_points.append(carla.Transform(carla.Location(x=5.6, y=128.9, z= 0.2), carla.Rotation(0, -90.3, 0))) #here FUNTKIONIERT !!!!!!!!!!
        #spawn_points.append(carla.Transform(carla.Location(x=-59.8, y=356.4, z= 0.2), carla.Rotation(0, 130.1, 0))) #here FUNKTIONIERT
        #spawn_points.append(carla.Transform(carla.Location(x=8.2, y=-65.7, z= 0.2), carla.Rotation(0, -90, 0))) 
        #spawn_points.append(carla.Transform(carla.Location(x=-9.9, y=-212.6, z= 0.2), carla.Rotation(0, 90, 0)))
        #spawn_points.append(carla.Transform(carla.Location(x=145.6, y=237.5, z= 0.2), carla.Rotation(0, -0.4, 0))) #other spawnpoint on left lane
        #spawn_points.append(carla.Transform(carla.Location(x=-7.8, y=304.0, z= 0.2), carla.Rotation(0, -66.4, 0))) #Kurve
        #spawn_points.append(carla.Transform(carla.Location(x=-5.1, y=-240.9, z= 0.2), carla.Rotation(0, 97.2, 0)))

        #init_transforms = self.world.get_map().get_spawn_points()
        #vehicle_init_transform = random.choice(init_transforms)
        vehicle_init_transform = random.choice(spawn_points)



        # create the vehicle
        if self.vehicle is None:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find('vehicle.' + self.vehicle_name)
            if vehicle_blueprint.has_attribute('color'):
                color = random.choice(vehicle_blueprint.get_attribute('color').recommended_values)
                vehicle_blueprint.set_attribute('color', color)
            self.vehicle = self.world.spawn_actor(vehicle_blueprint, vehicle_init_transform)
        else: 
            self.vehicle.set_transform(vehicle_init_transform)

    def _reset_other_vehicles(self):
        if not self.traffic:
            return

        # clear out old vehicles
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.world.tick()
        self.vehicles_list = []

        # initialize traffic manager
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.global_percentage_speed_difference(30.0)
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        # choose random spawn points
        num_vehicles = 0
        init_transforms = self.world.get_map().get_spawn_points()
        init_transforms = np.random.choice(init_transforms, num_vehicles)

        # spawn vehicles
        batch = []
        for transform in init_transforms:
            transform.location.z += 0.1  # otherwise can collide with the road it starts on
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True)))

        for response in self.client.apply_batch_sync(batch, False):
            self.vehicles_list.append(response.actor_id)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.vehicles_list.append(response.actor_id)

    def _compute_action(self):
        return self.agent.run_step()

    def step(self, action):
        rewards = []
        next_obs, done, info = np.array([]), False, {}
        for _ in range(self.frame_skip):
            if self.autopilot:
                vehicle_control = self._compute_action()
                steer = float(vehicle_control.steer)
                if vehicle_control.throttle > 0.0 and vehicle_control.brake == 0.0:
                    throttle_brake = vehicle_control.throttle
                elif vehicle_control.brake > 0.0 and vehicle_control.throttle == 0.0:
                    throttle_brake = - vehicle_control.brake # should be - vehicle_control.brake
                else:
                    throttle_brake = 0.0
                action = [throttle_brake, steer]
                print('action in auto pilot')
            next_obs, reward, done, info = self._simulator_step(action)
            rewards.append(reward)
            if done:
                break
        return next_obs, np.mean(rewards), done, info

    def _simulator_step(self, action):
        if self.render_display:
            if should_quit():
                return
            self.clock.tick()
        
        #noise = np.random.normal(self.mean, self.std, size = len(action))
        #action = np.clip(action + noise, -1.0, 1.0)

        # calculate actions
        throttle_brake = float(action[0])
        steer = float(action[1])
        self.steering_list.append(steer)
        if throttle_brake >= 0.0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake
        
        #action_list = [throttle_brake, steer]
        print('action', action)

        #if action_list[0] >= 0.0:
        #    noise = np.random.normal(self.mean, self.std, size = len(action_list))
        #    action_list = np.clip(action_list + noise, -1.0, 1.0)
        #else:
        #    noise = np.random.normal(self.mean, self.std, size = len(action_list))
        #    action_list = np.clip(action_list + noise, -1.0, 1.0)
        #    action_list[1] = 0.0
#
#
        #throttle = action_list[0]
        #brake = action_list[1]
        #steer = action_list[2]

        # apply control to simulation
        vehicle_control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )

        #print('vehicle_control', vehicle_control)

        self.vehicle.apply_control(vehicle_control)

        # advance the simulation and wait for the data
        if self.render_display and self.observations_type == 'pixel':
            snapshot, display_image, vision_image = self.sync_mode.tick(timeout=2.0)
        elif self.render_display and self.observations_type == 'state':
            snapshot, display_image = self.sync_mode.tick(timeout=2.0)
        elif not self.render_display and self.observations_type == 'pixel':
            snapshot, vision_image = self.sync_mode.tick(timeout=2.0)
        elif not self.render_display and self.observations_type == 'state':
            self.sync_mode.tick(timeout=2.0)
        else:
            raise ValueError('Unknown observation_type. Choose between: state, pixel')

        # Weather evolves
        self.weather.tick() #no weather changes


        # draw the display
        if self.render_display:
            draw_image(self.render_display, display_image)
            self.render_display.blit(self.font.render('Frame: %d' % self.count, True, (255, 255, 255)), (8, 10))
            self.render_display.blit(self.font.render('Thottle: %f' % throttle, True, (255, 255, 255)), (8, 28))
            self.render_display.blit(self.font.render('Steer: %f' % steer, True, (255, 255, 255)), (8, 46))
            self.render_display.blit(self.font.render('Brake: %f' % brake, True, (255, 255, 255)), (8, 64))
            self.render_display.blit(self.font.render(str(self.weather), True, (255, 255, 255)), (8, 82))
            pygame.display.flip()

        # get reward and next observation
        reward, done, info = self._get_reward(action)
        if self.observations_type == 'state':
            next_obs = self._get_state_obs()
        else:
            next_obs = self._get_pixel_obs(vision_image)

        # increase frame counter
        self.count += 1

        return next_obs, reward, done, info

    def _get_pixel_obs(self, vision_image):
        bgra = np.array(vision_image.raw_data).reshape(84, 84, 4)
        bgr = bgra[:, :, :3]
        rgb = np.flip(bgr, axis=2)
        return rgb

    def _get_state_obs(self):
        transform = self.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        x_pos = location.x
        y_pos = location.y
        z_pos = location.z
        pitch = rotation.pitch
        yaw = rotation.yaw
        roll = rotation.roll
        acceleration = vector_to_scalar(self.vehicle.get_acceleration())
        angular_velocity = vector_to_scalar(self.vehicle.get_angular_velocity())
        velocity = vector_to_scalar(self.vehicle.get_velocity())
        return np.array([x_pos,
                         y_pos,
                         z_pos,
                         pitch,
                         yaw,
                         roll,
                         acceleration,
                         angular_velocity,
                         velocity], dtype=np.float64)

    def _get_reward(self, action):

        vehicle_location = self.vehicle.get_location()
        follow_waypoint_reward = self._get_follow_waypoint_reward(vehicle_location)
        done, collision_reward = self._get_collision_reward()
        self.action = action

        rel_speed, base_reward, recenter_reward, speed_reward, line_reward, heading_reward, lane_leaving_reward, steering_diff_reward, action_reward, steering_angle_reward, distance_to_centerline = self._get_additional_reward(self.action)
        cost = self._get_cost()

        vehicle_velocity = self.vehicle.get_velocity()
        vehicle_velocity = np.around(np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2), 2)

        ###############
        #total_reward = 100 * follow_waypoint_reward + 100 * collision_reward
        #total_reward = 100 * collision_reward + base_reward + speed_reward + line_reward + heading_reward + lane_leaving_reward

        #if rel_speed > 0.05:
        #    total_reward = base_reward + recenter_reward + 400 * collision_reward
        #else: 
        #    total_reward = -0.4

        #total_reward = 0.6 * speed_reward + 0.1 * heading_reward + 0.2 * line_reward + 0.1 * steering_diff_reward
        if vehicle_velocity <= 0:
            total_reward = action_reward
        else:
            print('speed_reward', speed_reward)
            print('heading_reward', heading_reward)
            print('line_reward', line_reward)
            print('steering_diff_reward', steering_diff_reward)
            print('steering_angle_reward', steering_angle_reward)
            total_reward = 0.2 * speed_reward + 0.1 * heading_reward + 0.5 * line_reward + 0.2 * steering_diff_reward + steering_angle_reward

        print('total_reward', total_reward)
        print('distance_to_centerline', distance_to_centerline)
        ###############
        info_dict = dict()
        #info_dict['follow_waypoint_reward'] = follow_waypoint_reward
        info_dict['speed_reward'] = speed_reward
        info_dict['heading_reward'] = heading_reward
        info_dict['line_reward'] = line_reward
        info_dict['steering_diff_reward'] = steering_diff_reward
        info_dict['steering_angle_reward'] = steering_angle_reward
        info_dict['vehicle_velocity'] = vehicle_velocity
        info_dict['collision_reward'] = collision_reward
        info_dict['distance_to_centerline'] = distance_to_centerline
        info_dict['cost'] = cost
        

        return total_reward, done, info_dict

    def _get_follow_waypoint_reward(self, location):
        nearest_wp = self.map.get_waypoint(location, project_to_road=True)
        distance = np.sqrt(
            (location.x - nearest_wp.transform.location.x) ** 2 +
            (location.y - nearest_wp.transform.location.y) ** 2
        )
        return - distance
    
    def _get_additional_reward(self, action):
        vehicle_location = self.vehicle.get_location()

        vehicle_velocity = self.vehicle.get_velocity()
        vehicle_velocity = np.around(np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2), 2)

        vehicle_acceleration = self.vehicle.get_acceleration()
        speed_limit = self.vehicle.get_speed_limit()

        self.action = action

        #heading
        nearest_wp = self.map.get_waypoint(vehicle_location, project_to_road=True)
        lane_width = nearest_wp.lane_width
        vehicle_heading = self.vehicle.get_transform().rotation.yaw
        wp_heading = nearest_wp.transform.rotation.yaw
        delta_heading = np.abs(vehicle_heading - wp_heading)

        if delta_heading < 180:
            final_delta_heading = delta_heading
        elif delta_heading > 180 and delta_heading <= 360:
            final_delta_heading = 360 - delta_heading
        else:
            final_delta_heading = delta_heading - 360

        #final_delta_heading.append(heading_list) 
        self.heading_list.append(final_delta_heading)

        #distance to center line
        distance = - self._get_follow_waypoint_reward(vehicle_location)
        if final_delta_heading > 90:
            distance_to_centerline = np.around(nearest_wp.lane_width - distance, 2)
        else:
            distance_to_centerline = np.around(distance, 2)
        
        #print('distance_to_centerline', distance_to_centerline)
        
        #distance_to_centerline.append(distance_to_centerline_list)
        self.distance_to_centerline_list.append(distance_to_centerline)
        
        ##########################
        #rewards
        speed_reward = self._get_speed_reward(vehicle_velocity, speed_limit)
        line_reward = self._get_line_reward(distance_to_centerline)
        heading_reward = self._get_heading_reward(final_delta_heading)
        lane_leaving_reward = self._get_lane_leaving_reward(distance_to_centerline)
        ###########################

        d_max = 5

        d_center = self.distance_to_centerline_list[-1]
        d_center_prev = self.distance_to_centerline_list[-2]

        heading = self.heading_list[-1]
        heading_prev = self.heading_list[-2]


        ####### NEW REWARD FUNCTION ##########
        rel_speed = (vehicle_velocity / speed_limit)
        base_reward = self._get_base_reward(distance_to_centerline, rel_speed)
        recenter_reward = self._get_recenter_reward(d_max, d_center, d_center_prev, heading, heading_prev)
        ########################################

        steering = self.steering_list[-1]
        steering_prev = self.steering_list[-2]

        steering_diff_reward = self._get_steering_reward(steering, steering_prev)

        action_reward = self._get_action_reward(self.action)

        steering_angle_reward = self._get_steering_angle_reward(self.action)

        return rel_speed, base_reward, recenter_reward, speed_reward, line_reward, heading_reward, lane_leaving_reward, steering_diff_reward, action_reward, steering_angle_reward, distance_to_centerline
    
    def _get_steering_angle_reward(self, action):

        self.action = action

        action_steering = self.action[1]

        if action_steering <= 0.3 and action_steering >= -0.3:
            steering_angle_reward = 0
        elif action_steering > 0.3 and action_steering <= 1:
            steering_angle_reward = 6.05394 * action_steering**4 - 16.3259 * action_steering**3 + 12.9245 * action_steering**2 - 4.11591 * action_steering + 0.463328
        elif action_steering < -0.3 and action_steering >= -1: 
            steering_angle_reward = 6.05394 * (-action_steering)**4 - 16.3259 * (-action_steering)**3 + 12.9245 * (-action_steering)**2 - 4.11591 * (-action_steering) + 0.463328
        else:
            steering_angle_reward = 0
        return steering_angle_reward

    def _get_action_reward(self, action):

        self.action = action
        
        action_brake = self.action[0]

        if action_brake < 0:
            action_reward = -0.009621952 + 0.273692*action_brake + 0.2148355*action_brake**2 + 0.938688*action_brake**3
        else: 
            action_reward = 0
        
        print('action_reward', action_reward)

        return action_reward
    
    def _get_steering_reward(self, steering, steering_prev):

        abs_steering = abs(steering)
        abs_steering_prev = abs(steering_prev)

        diff = abs(abs_steering - abs_steering_prev)

        if diff >= 0.419:
            steering_diff_reward = 0
        elif diff >= 0 and diff < 0.419:
            steering_diff_reward = -5.7 * (diff)**2 + 1
        else:
            steering_diff_reward = 0

        return steering_diff_reward
    
    def _get_recenter_reward(self, d_max, d_center, d_center_prev, heading, heading_prev):
        
        if d_center < d_center_prev:
            r_find_lane = 1 - (d_center / d_max)
        else: 
            r_find_lane = 0 

        if heading <= heading_prev and d_center < 0.5:
            r_keeplane = 1 - np.sin(min(heading, 90)) 
        else: 
            r_keeplane = 0
        
        recenter_reward = 2 * (r_find_lane + r_keeplane)

        return recenter_reward

    def _get_base_reward(self, distance_to_centerline, rel_speed):

        reward_distant = 5 -(max(distance_to_centerline - 0.3, 0) + 1)**4

        if reward_distant > 0: 
            reward_center = reward_distant
        else: 
            reward_center = 0 

        p = distance_to_centerline - 1

        reward_speed = np.sqrt(rel_speed)

        if distance_to_centerline <= 1:
            reward_sum = reward_center + reward_speed
        else:
            reward_sum = reward_center + reward_speed - p

        if reward_sum > -1:
            base_reward = reward_sum
        else: 
            base_reward = -1       

        return base_reward

    def _get_speed_reward(self, current_speed, speed_limit):
        print('current_speed', current_speed)

        if current_speed <= 0:
            #speed_reward =  -0.03746917 + 2.111163*current_speed + 1.765929*current_speed**2 + 0.6194876*current_speed**3
            speed_reward = 0
        elif current_speed <= 15 and current_speed >= 0:
            #speed_reward = 0.001098417*current_speed**2.004401
            #speed_reward = 0.00000753259*current_speed**4-0.000464678*current_speed**3+0.0078757*current_speed**2+0.0118926*current_speed
            speed_reward = -0.0000109073*current_speed**4 + 0.000450624*current_speed**3-0.00896782*current_speed**2 + 0.136606*current_speed
        elif current_speed >= 15 and current_speed <= 20:
            #speed_reward = -0.4088191 + (1.342667 - -0.4088191)/(1 + (current_speed/30.54382)**79.46682)
            #speed_reward = 0.0000607315*current_speed**4-0.00462176*current_speed**3+0.100925*current_speed**2 -0.474584*current_speed
            speed_reward = 0.00708549 * current_speed**4 - 0.522372*current_speed**3 + 14.4249*current_speed**2 - 176.871 * current_speed + 812.776
        else:
            speed_reward = 0
        
        return speed_reward

    def _get_line_reward(self, distance_to_centerline):
        if distance_to_centerline >= 0 and distance_to_centerline <= 0.6:
            #line_reward = -349003.1 + (0.9335018 - -349003.1)/(1 + (distance_to_centerline/9.754655)**4.598628)
            line_reward = 0.249716 * distance_to_centerline**4 - 7.54956 * distance_to_centerline**3 + 2.04709 * distance_to_centerline**2 - 0.23102 * distance_to_centerline + 1
        else:
            line_reward = 0
        return line_reward
    
    def _get_heading_reward(self, final_delta_heading):
        if final_delta_heading < 0:
            heading_reward = 0
        elif final_delta_heading >= 0 and final_delta_heading <= 15:
            heading_reward = 0.0000429958 * final_delta_heading**4 - 0.00241822 * final_delta_heading**3 + 0.0457238 * final_delta_heading**2 - 0.353536 * final_delta_heading + 1
            #heading_reward =  -0.02852283 + (1.000078 - -0.02852283)/(1 + (final_delta_heading/1.045035)**1.211741)
        else:
            heading_reward = 0
        return heading_reward
    
    def _get_lane_leaving_reward(self, distance_to_centerline):
        if distance_to_centerline >= 1.6:
            lane_leaving_reward = -20
        else:
            lane_leaving_reward = 0
        return lane_leaving_reward

    def _get_collision_reward(self):
        if not self.collision:
            return False, 0
        else:
            return True, -1

    def _get_cost(self):
        # TODO: define cost function
        return 0

    def _on_collision(self, event):
        other_actor = get_actor_name(event.other_actor)
        self.collision = True
        self._reset_vehicle()

    def close(self):
        for actor in self.actor_list:
            actor.destroy()
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        time.sleep(0.5)
        pygame.quit()
    
    def render(self, mode):
        pass


def vector_to_scalar(vector):
    scalar = np.around(np.sqrt(vector.x ** 2 +
                               vector.y ** 2 +
                               vector.z ** 2), 2)
    return scalar


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def get_actor_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

