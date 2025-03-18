import argparse
import os
import time
import carla
import numpy as np
import cv2
import random
import shutil
import pygame
from pygame.locals import K_ESCAPE
from pygame.locals import K_q

from utils import *


class World(object):
    def __init__(self, client, args):
        self.args = args
        self.client = client
        self.recording = True
        self.vehicles = []
        self.display_surface = None

        # 创建保存目录
        self.video_dir = os.path.join('junction')
        os.makedirs(self.video_dir, exist_ok=True)

        self.town = "Town02"
        self.fps = 30.0
        self.recording_duration = 10.0  # 单个视频的录制时长，秒
        self.expected_frames = int(self.fps * self.recording_duration)
        self.frame_buffer = []

        # 加载地图
        self.client.load_world(self.town)
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager()

        # 清理现有车辆
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor.destroy()

        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.fps
        self.world.apply_settings(settings)

        # 配置TrafficManager
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(99)
        self.traffic_manager.global_percentage_speed_difference(20)

        # 获取地图和蓝图库
        carla_map = self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()

        # 找到交叉路口生成点
        intersection_spawn_points = self._find_comprehensive_intersection_spawn_points(carla_map)

        if not intersection_spawn_points:
            print("No intersection spawn points found!")
            return

        # 随机选择一个交叉路口生成点
        selected_spawn_point = random.choice(intersection_spawn_points)

        # 选择车辆蓝图（Tesla Model 3）
        available_models = self._filter_vehicle_blueprints(blueprint_library.filter('vehicle.*'))
        try:
            ego_car_bp = next(bp for bp in available_models if 'model3' in bp.id.lower())
        except StopIteration:
            print("Tesla Model 3 not found, using a random vehicle model")
            ego_car_bp = random.choice(available_models)

        # 生成ego车辆
        print(f"Spawning ego vehicle at intersection: {selected_spawn_point.location}")
        ego_vehicle = self.world.spawn_actor(ego_car_bp, selected_spawn_point)
        self.vehicles.append(ego_vehicle)

        forward_vector = selected_spawn_point.get_forward_vector()
        b_car_bp = next(bp for bp in available_models if 'cybertruck' in bp.id.lower())

        offset = 10  # B车在前方10米（可以根据需要调整）
        max_attempts = 10  # 最大尝试次数

        # 尝试找到可用的生成位置
        for attempt in range(max_attempts):
            # 基于选定的ego A车生成点，使用偏移逻辑计算B车的位置
            spawn_point_vehicle = carla.Transform(
                carla.Location(
                    x=selected_spawn_point.location.x + offset * forward_vector.x,
                    y=selected_spawn_point.location.y + offset * forward_vector.y,
                    z=selected_spawn_point.location.z + 0.1  # 高度略微偏移
                ),
                selected_spawn_point.rotation
            )

            # 检查生成点是否空闲并生成B车
            vehicle_b = self.world.try_spawn_actor(b_car_bp, spawn_point_vehicle)
            if vehicle_b:
                print(f"B car successfully spawned after {attempt + 1} attempts.")
                self.vehicles.append(vehicle_b)
                break
            else:
                # 如果生成失败，增加偏移尝试新位置
                offset += 5  # 每次增加偏移距离
        else:
            print("Failed to spawn B car after multiple attempts.")
            return  # 或者退出函数

        # 相机设置
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
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

        # 注册相机传感器回调
        self.camera.listen(lambda image: self._parse_image(image))

        # 设置车辆为自动驾驶
        ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        self.traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 0)
        vehicle_b.set_autopilot(True, self.traffic_manager.get_port())
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle_b, 0)

        # 初始化世界状态
        self.world.tick()
        print("Recording started...")

    def _find_comprehensive_intersection_spawn_points(self, carla_map):
        """
        找到交叉路口生成点
        """
        spawn_points = carla_map.get_spawn_points()
        intersection_spawn_points = []

        for spawn_point in spawn_points:
            waypoint = carla_map.get_waypoint(spawn_point.location)
            search_distances = [15.0, 20.0]

            for distance in search_distances:
                next_waypoints = waypoint.next(distance)
                if any(wp.is_junction for wp in next_waypoints):
                    intersection_spawn_points.append(spawn_point)
                    break

            if len(intersection_spawn_points) > 10:
                break

        if not intersection_spawn_points:
            print("Warning: No spawn points found near intersection. Using all spawn points.")
            return spawn_points

        return intersection_spawn_points

    def _filter_vehicle_blueprints(self, blueprints):
        """
        过滤车辆蓝图
        """
        filtered_blueprints = []
        for bp in blueprints:
            # 过滤掉商用车和特殊车辆
            if bp.id.lower() not in ['vehicle.ford.ambulance', 'vehicle.ford.firetruck']:
                filtered_blueprints.append(bp)
        return filtered_blueprints

    def tick(self, clock=None):
        """
        更新世界状态，检查是否应该结束录制
        """
        self.world.tick()

        # 如果已经收集了足够的帧数，停止录制并生成视频
        if len(self.frame_buffer) >= self.expected_frames:
            if self.recording:
                print(f"Recording completed. Collected {len(self.frame_buffer)} frames.")
                self._save_video()
                self.recording = False
                return False  # 告诉主循环结束

        # 显示进度
        if len(self.frame_buffer) % 30 == 0:  # 每秒显示一次进度
            print(f"Recording: {len(self.frame_buffer)}/{self.expected_frames} frames")

        return True  # 继续运行

    def _parse_image(self, image):
        """
        处理相机图像
        """
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_array = array[:, :, :3][:, :, ::-1]

            # 用于显示
            self.display_surface = pygame.image.frombuffer(
                bgr_array.tobytes(), (image.width, image.height), 'RGB')

            # 只有在录制状态下才添加帧
            if self.recording and len(self.frame_buffer) < self.expected_frames:
                self.frame_buffer.append(bgr_array.copy())

        except Exception as e:
            print(f"Error in _parse_image: {e}")
        finally:
            del array
            import gc
            gc.collect()

    def _save_video(self):
        """
        将缓存的帧保存为单个视频
        """
        try:
            video_path = os.path.join(self.video_dir, 'carla_recording.mp4')

            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = self.frame_buffer[0].shape[:2]
            video_writer = cv2.VideoWriter(
                video_path,
                fourcc, self.fps,
                (width, height)
            )

            # 写入所有帧
            for frame in self.frame_buffer:
                video_writer.write(frame)

            video_writer.release()
            print(f"Video saved to {video_path}")

        except Exception as e:
            print(f"Error saving video: {e}")

    def destroy(self):
        """
        清理资源
        """
        print("Cleaning up resources...")
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        self.traffic_manager.set_synchronous_mode(False)

        # 如果还没保存视频，且有足够帧数，则保存
        if self.recording and len(self.frame_buffer) > 0:
            self._save_video()

        for actor in [self.camera] + self.vehicles:
            if actor:
                actor.destroy()

        print("Cleanup complete!")

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        world = World(client, args)

        clock = pygame.time.Clock()
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA Scenario")

        running = True
        while running:
            clock.tick_busy_loop(60)
            world.tick(clock)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYUP:
                    if event.key == K_ESCAPE or event.key == K_q:
                        running = False

            # 如果有可用的显示表面，则绘制
            if world.display_surface is not None:
                # 缩放图像以适应显示窗口
                scaled_surface = pygame.transform.scale(world.display_surface, (args.width, args.height))
                display.blit(scaled_surface, (0, 0))

            pygame.display.flip()

    finally:
        if world is not None:
            world.destroy()
        pygame.quit()

def main():
    argparser = argparse.ArgumentParser(description='CARLA Dataset Generator')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='CARLA server host')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='CARLA server port')
    argparser.add_argument('--width', default=800, type=int, help='Image width')
    argparser.add_argument('--height', default=600, type=int, help='Image height')
    args = argparser.parse_args()

    try:
        game_loop(args)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDataset generation complete')


if __name__ == '__main__':
    main()