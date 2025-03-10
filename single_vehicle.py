import argparse
import os
import time
import carla
import numpy as np
import cv2
import pygame
import random
from pygame.locals import K_ESCAPE
from pygame.locals import K_q


def is_spawn_point_on_driveway(world, spawn_point):
    carla_map = world.get_map()
    waypoint = carla_map.get_waypoint(spawn_point.location,
                                      project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    return waypoint.lane_type == carla.LaneType.Driving


def check_spawn_points(world, spawn_points):
    valid_spawn_points = []
    for i, spawn_point in enumerate(spawn_points):
        is_driveway = is_spawn_point_on_driveway(world, spawn_point)
        if is_driveway:
            valid_spawn_points.append(spawn_point)

    print(f"Found {len(valid_spawn_points)} valid spawn points on driveways out of {len(spawn_points)} total")

    if not valid_spawn_points:
        print("Warning: No valid spawn points found on driveways, using first available spawn point")
        valid_spawn_points.append(spawn_points[0])

    return valid_spawn_points


class World(object):
    def __init__(self, client, args):
        self.args = args
        self.client = client
        self.recording = True
        self.current_frame = 0
        self.vehicles = []
        self.display = None
        self.image_surface = None

        # 目录设置
        self.img_dir = 'img'
        self.video_dir = 'video'
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # 录像设置
        self.fps = 30.0
        self.video_writer = None
        self.recording_start_time = None
        self.recording_duration = 25.0  # 25秒
        self.initial_stabilization_time = 3.0  # 初始稳定时间

        # 加载地图
        self.client.load_world('Town05')
        self.world = self.client.get_world()

        # 清理现有车辆
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor.destroy()

        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 30.0
        self.world.apply_settings(settings)

        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        carla_map = self.world.get_map()

        # 获取有效的spawn点
        valid_spawn_points = check_spawn_points(self.world, spawn_points)
        selected_spawn_point = random.choice(valid_spawn_points)

        # 车辆蓝图选择
        available_models = blueprint_library.filter('vehicle.*')
        try:
            c_car_bp = next(bp for bp in available_models if
                            'mustang' in bp.id.lower())
        except StopIteration:
            print("Specified car models not found, using a random vehicle model")
            c_car_bp = random.choice(available_models)

        # 车辆参数设置
        self.base_speed = 8
        self.vehicle_details = []

        # 创建ego车辆
        print(f"Spawning ego vehicle at {selected_spawn_point.location}")
        ego_vehicle = self.world.spawn_actor(c_car_bp, selected_spawn_point)

        ego_info = {
            'actor': ego_vehicle,
            'model': c_car_bp.id,
            'initial_speed': self.base_speed,
            'spawn_point': selected_spawn_point,
            'name': 'Ego'
        }
        self.vehicle_details.append(ego_info)
        self.vehicles.append(ego_vehicle)

        # 获取ego车辆前方的路点
        waypoint = carla_map.get_waypoint(selected_spawn_point.location)

        # 找到前方同一车道上的点
        next_waypoints = []
        current_waypoint = waypoint

        # 向前找大约30米的点，确保在同一车道
        distance = 0
        while distance < 15:
            next_wps = current_waypoint.next(5.0)  # 每次前进5米
            if not next_wps:
                break
            current_waypoint = next_wps[0]  # 保持在同一车道
            next_waypoints.append(current_waypoint)
            distance += 5.0

        if next_waypoints:
            # 选择前方适当距离的路点
            front_waypoint = next_waypoints[1]

            # 创建B车的spawn点，使用与ego车相同的高度
            spawn_transform = carla.Transform(
                front_waypoint.transform.location + carla.Location(z=0.5),  # 稍微抬高以避免与地面碰撞
                front_waypoint.transform.rotation
            )

            # 为B车选择不同的车型
            try:
                b_car_bp = random.choice([bp for bp in available_models if 'cybertruck' in bp.id.lower() and bp.id != c_car_bp.id])
            except:
                b_car_bp = random.choice(available_models)

            # 生成B车
            print(f"Spawning B vehicle at {spawn_transform.location}, {distance} meters ahead of ego")
            b_vehicle = self.world.spawn_actor(b_car_bp, spawn_transform)

            # 将B车添加到车辆列表
            b_info = {
                'actor': b_vehicle,
                'model': b_car_bp.id,
                'initial_speed': self.base_speed - 5,  # 让B车速度稍慢，便于观察
                'spawn_point': spawn_transform,
                'name': 'B-Vehicle'
            }
            self.vehicle_details.append(b_info)
            self.vehicles.append(b_vehicle)

            a_spawn_point = None
            lane_description = None

            # 找到比B车更前方10米的路点
            extended_waypoints = []
            current_waypoint = front_waypoint
            distance_from_b = 0

            while distance_from_b < 10:  # 向前10米
                next_wps = current_waypoint.next(5.0)
                if not next_wps:
                    break
                current_waypoint = next_wps[0]
                extended_waypoints.append(current_waypoint)
                distance_from_b += 5.0

            # 如果成功找到更前方的路点
            if extended_waypoints:
                # 使用最后一个路点作为参考
                final_waypoint = extended_waypoints[-1]

                left_lane_waypoint = final_waypoint.get_left_lane()
                right_lane_waypoint = final_waypoint.get_right_lane()

                if left_lane_waypoint and left_lane_waypoint.lane_type == carla.LaneType.Driving:
                    a_spawn_point = carla.Transform(
                        left_lane_waypoint.transform.location + carla.Location(z=0.5),
                        left_lane_waypoint.transform.rotation
                    )
                    lane_description = "left lane"
                # 如果左车道不可用，检查右车道
                elif right_lane_waypoint and right_lane_waypoint.lane_type == carla.LaneType.Driving:
                    a_spawn_point = carla.Transform(
                        right_lane_waypoint.transform.location + carla.Location(z=0.5),
                        right_lane_waypoint.transform.rotation
                    )
                    lane_description = "right lane"

                # 如果找到可用的生成点
                if a_spawn_point:
                    # 为A车选择车型（之前的代码保持不变）
                    try:
                        a_car_bp = random.choice([
                            bp for bp in available_models
                            if 'model3' in bp.id.lower() and bp.id not in [c_car_bp.id, b_car_bp.id]
                        ])
                    except:
                        a_car_bp = random.choice([
                            bp for bp in available_models
                            if bp.id not in [c_car_bp.id, b_car_bp.id]
                        ])

                    # 生成A车，速度最快
                    a_base_speed = self.base_speed + 10  # 比其他车快10km/h

                    print(f"Spawning A vehicle in {lane_description}, 10m ahead of B vehicle")
                    a_vehicle = self.world.spawn_actor(a_car_bp, a_spawn_point)

                    # 将A车添加到车辆列表
                    a_info = {
                        'actor': a_vehicle,
                        'model': a_car_bp.id,
                        'initial_speed': a_base_speed,
                        'spawn_point': a_spawn_point,
                        'name': 'A-Vehicle'
                    }
                    self.vehicle_details.append(a_info)
                    self.vehicles.append(a_vehicle)

        # 相机设置
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.width))
        camera_bp.set_attribute('image_size_y', str(args.height))
        camera_bp.set_attribute('fov', '110')

        camera_transform = carla.Transform(
            carla.Location(x=1.2, y=0.0, z=1.7),
            carla.Rotation(pitch=-5, yaw=0)
        )
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=ego_vehicle
        )

        # 交通管理器设置
        self.traffic_manager = client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.global_percentage_speed_difference(10)
        self.traffic_manager.set_hybrid_physics_mode(True)

        # 注册相机传感器回调
        self.camera.listen(lambda image: self._parse_image(image))

        # 车辆行为设置
        ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        self._set_vehicle_speed(ego_vehicle, self.base_speed)
        b_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        self._set_vehicle_speed(b_vehicle, self.base_speed - 5)
        a_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        self._set_vehicle_speed(a_vehicle, self.base_speed + 5)

        # 初始化世界状态
        self.world.tick()

    def _set_vehicle_speed(self, vehicle, target_speed):
        forward_vector = vehicle.get_transform().get_forward_vector()
        velocity = carla.Vector3D(
            x=forward_vector.x * target_speed / 3.6,
            y=forward_vector.y * target_speed / 3.6,
            z=0
        )
        vehicle.set_target_velocity(velocity)
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 0)

    def _parse_image(self, image):
        # 图像处理和保存逻辑
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        bgr_array = np.copy(array[:, :, ::-1])
        rgb_array = array

        # Pygame显示
        if self.display is not None:
            pygame_surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
            self.image_surface = pygame_surface

        # 记录和保存图像
        if self.recording and self.recording_start_time is not None:
            elapsed_time = time.time() - self.recording_start_time

            # 只在初始稳定时间后开始录制
            if elapsed_time > self.initial_stabilization_time:
                frame_filename = os.path.join(self.img_dir, f"frame_{self.current_frame:06d}.png")
                cv2.imwrite(frame_filename, bgr_array)

                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(
                        os.path.join(self.video_dir, 'recording.mp4'),
                        fourcc, self.fps,
                        (image.width, image.height)
                    )

                self.video_writer.write(bgr_array)
                self.current_frame += 1

    def tick(self, clock, display):
        self.display = display
        self.world.tick()

        # 初始稳定处理
        if self.recording_start_time is None:
            self.recording_start_time = time.time()

        # 渲染相机图像
        if self.image_surface is not None:
            display.blit(self.image_surface, (0, 0))

        # 录制结束检查
        if self.recording and (time.time() - self.recording_start_time) > (
                self.recording_duration + self.initial_stabilization_time):
            print("Recording Details:")
            for vehicle_info in self.vehicle_details:
                print(f"Vehicle: {vehicle_info['name']}, Model: {vehicle_info['model']}, "
                      f"Speed: {vehicle_info['initial_speed']:.1f} km/h")

            print(f"Recording completed after {self.recording_duration} seconds")
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                print(f"Video saved to {self.video_dir}/recording.mp4")

    def destroy(self):
        # 清理资源
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        if self.video_writer:
            self.video_writer.release()

        for actor in [self.camera] + self.vehicles:
            if actor:
                actor.destroy()


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Two Vehicle Simulation")

        world = World(client, args)
        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)

            # 传递display给world.tick方法，使其能够渲染图像
            world.tick(clock, display)

            # 处理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYUP:
                    if event.key == K_ESCAPE or event.key == K_q:
                        return

            # 更新显示
            pygame.display.flip()

    finally:
        if world is not None:
            world.destroy()
        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(description='CARLA Scenario')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)'
    )
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)'
    )
    argparser.add_argument(
        '--width',
        default=800,
        type=int,
        help='Width of the window (default: 800)'
    )
    argparser.add_argument(
        '--height',
        default=600,
        type=int,
        help='Height of the window (default: 600)'
    )
    args = argparser.parse_args()

    try:
        game_loop(args)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nExit')


if __name__ == '__main__':
    main()