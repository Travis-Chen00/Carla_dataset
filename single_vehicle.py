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
    """
    检查给定的spawn point是否在drive way上

    :param world: Carla的world对象
    :param spawn_point: 要检查的spawn point
    :return: 布尔值，表示是否在drive way上
    """
    # 获取地图
    carla_map = world.get_map()

    # 使用carla_map的waypoint功能
    waypoint = carla_map.get_waypoint(spawn_point.location,
                                      project_to_road=True,  # 投影到最近的道路
                                      lane_type=carla.LaneType.Driving)  # 限定为行驶车道

    # 检查waypoint的lane type是否为Driving
    is_on_driveway = waypoint.lane_type == carla.LaneType.Driving

    return is_on_driveway


def check_spawn_points(world, spawn_points):
    """
    检查所有spawn点并返回有效的spawn点（位于drive way上的点）
    """
    valid_spawn_points = []

    for i, spawn_point in enumerate(spawn_points):
        is_driveway = is_spawn_point_on_driveway(world, spawn_point)
        print(f"Spawn Point {i}: {'Drive Way' if is_driveway else 'Not Drive Way'}")
        print(f"Location: {spawn_point.location}")

        if is_driveway:
            valid_spawn_points.append(spawn_point)

    print(f"Found {len(valid_spawn_points)} valid spawn points on driveways out of {len(spawn_points)} total")

    # 确保我们至少有一个有效点
    if not valid_spawn_points:
        print("Warning: No valid spawn points found on driveways, using first available spawn point")
        valid_spawn_points.append(spawn_points[0])

    return valid_spawn_points


class World(object):
    def __init__(self, client, args):
        self.args = args
        self.world = None
        self.client = client
        self.recording = True
        self.current_frame = 0
        self.vehicles = []
        self.display = None  # 添加显示属性
        self.image_surface = None  # 添加pygame图像表面

        # 创建保存目录
        self.img_dir = 'img'
        self.video_dir = 'video'
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # 录像设置
        self.fps = 30.0
        self.video_writer = None

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

        # 获取有效的spawn点（位于drive way上的点）
        valid_spawn_points = check_spawn_points(self.world, spawn_points)

        # 选择一个随机的有效spawn点作为C车的spawn点
        selected_spawn_point = random.choice(valid_spawn_points)

        available_models = blueprint_library.filter('vehicle.*')
        # 尝试找到指定的车型，如果没有就随机选择一个
        try:
            c_car_bp = next(bp for bp in available_models if
                            'mustang' in bp.id.lower() or 'etron' in bp.id.lower() or 'tt' in bp.id.lower())
        except StopIteration:
            print("Specified car models not found, using a random vehicle model")
            c_car_bp = random.choice(available_models)

        # 保存各车型的ID
        self.vehicle_blueprints = [c_car_bp]
        self.vehicle_models = [bp.id for bp in self.vehicle_blueprints]

        # 基础速度设置
        self.base_speed = random.uniform(30, 45)  # 30-45 km/h
        # B车速度比C车快20%
        b_car_speed = self.base_speed * 1.2
        # 存储车辆详细信息的列表
        self.vehicle_details = []

        # 创建ego车辆(C车)
        print(f"Spawning ego vehicle at {selected_spawn_point.location}")
        ego_vehicle = self.world.spawn_actor(c_car_bp, selected_spawn_point)

        # C车信息记录
        ego_info = {
            'actor': ego_vehicle,
            'model': c_car_bp.id,
            'initial_speed': self.base_speed,
            'spawn_point': selected_spawn_point,
            'name': 'Ego'  # C车
        }
        self.vehicle_details.append(ego_info)
        self.vehicles.append(ego_vehicle)

        # 为B车找一个合适的生成点
        # 方法1：从其他可用的有效spawn点中选择一个在前方的点
        b_car_spawn_point = None
        ego_location = selected_spawn_point.location
        ego_forward = selected_spawn_point.get_forward_vector()

        # 创建一个包含距离信息的列表
        spawn_points_with_distance = []
        for point in valid_spawn_points:
            if point != selected_spawn_point:  # 排除C车所在点
                # 计算位置差向量
                diff_vector = point.location - ego_location
                # 计算向量点乘来确定是否在前方
                forward_dot = ego_forward.x * diff_vector.x + ego_forward.y * diff_vector.y

                # 如果点乘为正，则该点在ego车前方
                if forward_dot > 0:
                    distance = ego_location.distance(point.location)
                    spawn_points_with_distance.append((point, distance, forward_dot))

        # 根据与ego车的距离排序，优先选择距离适中的点
        if spawn_points_with_distance:
            # 按距离排序
            spawn_points_with_distance.sort(key=lambda x: abs(x[1] - 10.0))  # 尝试找距离约10米的点
            b_car_spawn_point = spawn_points_with_distance[0][0]
            actual_distance = spawn_points_with_distance[0][1]
            print(f"Found suitable spawn point for B car at distance {actual_distance:.2f}m")

        # 如果没找到合适的点，尝试方法2：直接在车道上前方创建新的生成点
        if b_car_spawn_point is None:
            try:
                # 获取当前车辆的waypoint
                waypoint = self.world.get_map().get_waypoint(ego_location)

                # 获取前方10米的waypoint
                next_waypoint = waypoint.next(10.0)[0]  # 获取10米前的waypoint

                # 创建一个新的transform
                b_car_transform = carla.Transform(
                    next_waypoint.transform.location,
                    next_waypoint.transform.rotation
                )

                # 尝试先检查这个位置是否有碰撞
                if not self.world.cast_ray(b_car_transform.location, b_car_transform.location + carla.Location(z=2.0)):
                    b_car_spawn_point = b_car_transform
                    print(f"Created new spawn point for B car using waypoint")
            except Exception as e:
                print(f"Failed to create spawn point using waypoint: {e}")

        # 如果仍然没找到点，尝试方法3：搜索现有车道上没有障碍物的点
        if b_car_spawn_point is None:
            print("Searching for any available spawn point...")
            # 筛选出所有可用的spawn点
            potential_points = []

            # 检查每个点是否有障碍物
            for point in valid_spawn_points:
                if point != selected_spawn_point:
                    # 检查该位置是否有碰撞
                    collision = False
                    # 简单的碰撞检测：向上发射一个短射线看是否有碰撞
                    if not self.world.cast_ray(point.location, point.location + carla.Location(z=2.0)):
                        potential_points.append(point)

            if potential_points:
                b_car_spawn_point = random.choice(potential_points)
                print(f"Selected a random available spawn point for B car")

        # 如果所有方法都失败了
        if b_car_spawn_point is None:
            print("Warning: Could not find a suitable spawn point for B car. Skipping B car creation.")
        else:
            # 尝试找到不同于C车的车型 (例如选择一个运动型车辆)
            try:
                b_car_bp = next(bp for bp in available_models if
                                ('coupe' in bp.id.lower() or 'sport' in bp.id.lower()) and bp.id != c_car_bp.id)
            except StopIteration:
                print("Specific B car model not found, using a random different model")
                available_b_models = [bp for bp in available_models if bp.id != c_car_bp.id]
                b_car_bp = random.choice(available_b_models) if available_b_models else random.choice(available_models)

            try:
                # 创建B车
                b_vehicle = self.world.spawn_actor(b_car_bp, b_car_spawn_point)

                # B车信息记录
                b_info = {
                    'actor': b_vehicle,
                    'model': b_car_bp.id,
                    'initial_speed': b_car_speed,
                    'spawn_point': b_car_spawn_point,
                    'name': 'B'  # B车
                }
                self.vehicle_details.append(b_info)
                self.vehicles.append(b_vehicle)

                print(f"Successfully spawned B car with model {b_car_bp.id}")
            except Exception as e:
                print(f"Failed to spawn B car: {e}")

        # KITTI风格相机 - 安装在C车上
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.width))
        camera_bp.set_attribute('image_size_y', str(args.height))
        camera_bp.set_attribute('fov', '110')

        # 调整相机位置，确保能看到前方车辆
        camera_transform = carla.Transform(
            carla.Location(x=1.2, y=0.0, z=1.7),  # 中心位置，高度接近人眼
            carla.Rotation(pitch=-5, yaw=0)  # 略微向下
        )
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=ego_vehicle  # 附加到C车
        )

        # 创建交通管理器用于控制车辆行为
        self.traffic_manager = client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

        # 注册相机传感器回调
        self.camera.listen(lambda image: self._parse_image(image))

        # 录制设置
        self.recording_start_time = time.time()
        self.recording_duration = 25.0  # 25秒

        # 设置所有车辆的自动驾驶和初始速度
        for vehicle_info in self.vehicle_details:
            vehicle = vehicle_info['actor']
            # 设置自动驾驶
            vehicle.set_autopilot(True, self.traffic_manager.get_port())

            # 设置初始速度
            forward_vector = vehicle_info['spawn_point'].get_forward_vector()
            initial_velocity = carla.Vector3D(
                x=forward_vector.x * vehicle_info['initial_speed'] / 3.6,
                y=forward_vector.y * vehicle_info['initial_speed'] / 3.6,
                z=0
            )
            vehicle.set_target_velocity(initial_velocity)

        # 设置车辆行为
        # 设置车辆行为
        for vehicle_info in self.vehicle_details:
            vehicle = vehicle_info['actor']

            if vehicle_info['name'] == 'Ego':
                self.traffic_manager.auto_lane_change(vehicle, False)  # 禁止变道
                self.traffic_manager.force_lane_change(vehicle, False)  # 不强制变道
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 0)  # 保持设定速度
                self.traffic_manager.distance_to_leading_vehicle(vehicle, 5)
                self.traffic_manager.update_vehicle_lights(vehicle, True)

            elif vehicle_info['name'] == 'B':
                self.traffic_manager.auto_lane_change(vehicle, False)  # 禁止变道
                self.traffic_manager.force_lane_change(vehicle, False)  # 不强制变道
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -20)  # 速度比默认快20%
                self.traffic_manager.distance_to_leading_vehicle(vehicle, 3)  # 设置较小的跟车距离
                self.traffic_manager.update_vehicle_lights(vehicle, True)

        # 初始世界状态
        self.world.tick()

    def _parse_image(self, image):
        # 创建可写的NumPy数组副本
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        # 创建可写副本
        bgr_array = np.copy(array[:, :, ::-1])  # BGR (OpenCV) 格式
        rgb_array = array  # RGB 格式用于Pygame

        # 创建pygame表面
        if self.display is not None:
            # 将numpy数组转换为pygame表面
            pygame_surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
            self.image_surface = pygame_surface  # 保存表面以便在tick方法中渲染

        # 记录车辆详细信息
        for vehicle_info in self.vehicle_details:
            vehicle = vehicle_info['actor']
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            location = vehicle.get_location()

            # 更新车辆信息
            vehicle_info.update({
                'current_speed': speed,
                'location': {
                    'x': location.x,
                    'y': location.y,
                    'z': location.z
                }
            })

        # 保存图像和视频
        if self.recording:
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
        # 保存display引用
        self.display = display

        # 更新世界
        self.world.tick()

        # 定期检查并调整车辆速度和行为
        if self.current_frame % 30 == 0:  # 每秒调整一次（假设30fps）
            for vehicle_info in self.vehicle_details:
                vehicle = vehicle_info['actor']

                # 获取当前速度
                current_speed = vehicle_info.get('current_speed', 0)
                target_speed = self.base_speed

                # 如果当前速度低于目标速度的80%，恢复速度
                if current_speed < target_speed * 0.8:
                    forward_vector = vehicle.get_transform().get_forward_vector()
                    velocity = carla.Vector3D(
                        x=forward_vector.x * target_speed / 3.6,
                        y=forward_vector.y * target_speed / 3.6,
                        z=0
                    )
                    vehicle.set_target_velocity(velocity)
                    print(f"调整车辆 {vehicle_info['name']} 速度至 {target_speed:.2f} km/h")

        # 渲染相机图像到pygame窗口
        if self.image_surface is not None:
            display.blit(self.image_surface, (0, 0))

        # 检查录制时间是否已经结束
        if self.recording and (time.time() - self.recording_start_time) > self.recording_duration:
            print("Recording Details:")
            for vehicle_info in self.vehicle_details:
                print(f"Vehicle {vehicle_info['name']}:")
                print(f"  Model: {vehicle_info['model']}")
                print(f"  Initial Speed: {vehicle_info['initial_speed']:.2f} km/h")
                print(f"  Current Speed: {vehicle_info.get('current_speed', 'N/A'):.2f} km/h")
                if 'location' in vehicle_info:
                    print(
                        f"  Location: x={vehicle_info['location']['x']:.2f}, y={vehicle_info['location']['y']:.2f}, z={vehicle_info['location']['z']:.2f}")
                else:
                    print(f"  Location: N/A")
                print()

            print(f"Recording completed after {self.recording_duration} seconds")
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                print(f"Video saved to {self.video_dir}/recording.mp4")

    def destroy(self):
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
        pygame.display.set_caption("Single Vehicle Simulation")

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
    argparser = argparse.ArgumentParser(description='Single Vehicle')
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


if __name__ == "__main__":
    main()
