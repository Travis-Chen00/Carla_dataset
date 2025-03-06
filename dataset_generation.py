#!/usr/bin/env python

import argparse
import logging
import time
import os
import numpy as np
import pygame
import carla
import cv2
import random
from pygame.locals import K_ESCAPE
from pygame.locals import K_q


class World(object):
    def __init__(self, client, args):
        self.args = args
        self.world = None
        self.client = client
        self.recording = True
        self.current_frame = 0
        self.vehicles = []

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

        # 选择一个直路段的生成点
        spawn_point = next(
            point for point in spawn_points
            if point.location.x > 0 and point.location.y > 0
        )

        # 三辆车的车型 - 使用通过筛选确保存在的车型
        # 获取可用的车型列表
        available_models = blueprint_library.filter('vehicle.*')

        # 为A车选择紧凑型轿车
        a_car_bp = next(bp for bp in available_models if
                        'model3' in bp.id.lower() or 'a4' in bp.id.lower() or 'prius' in bp.id.lower())

        # 为B车选择小型货车/卡车
        b_car_bp = next(bp for bp in available_models if 'cybertruck' in bp.id.lower())

        # 为C车选择轿车
        c_car_bp = next(bp for bp in available_models if
                        'mustang' in bp.id.lower() or 'etron' in bp.id.lower() or 'tt' in bp.id.lower())

        # 保存各车型的ID
        self.vehicle_blueprints = [a_car_bp, b_car_bp, c_car_bp]
        self.vehicle_models = [bp.id for bp in self.vehicle_blueprints]

        # 基础速度设置
        self.base_speed = random.uniform(45, 55)  # 45-55 km/h
        # B车速度设置比基础速度更快
        self.b_car_speed_factor = 1.3  # B车速度是基础速度的130%

        # 存储车辆详细信息的列表
        self.vehicle_details = []

        # 首先创建我们的ego车辆(C车)，其他车辆将放在它的前方
        ego_vehicle = self.world.spawn_actor(c_car_bp, spawn_point)

        # C车信息记录
        ego_info = {
            'actor': ego_vehicle,
            'model': c_car_bp.id,
            'initial_speed': self.base_speed,
            'spawn_point': spawn_point,
            'name': 'C'  # C车
        }
        self.vehicle_details.append(ego_info)
        self.vehicles.append(ego_vehicle)

        # 在ego车前方放置A和B车，距离合适，确保在视野内
        forward_vector = spawn_point.get_forward_vector()

        # A和B车型，修改车辆生成位置以避免碰撞
        for i, vehicle_bp in enumerate([a_car_bp, b_car_bp]):  # 只处理前两个车型 (A和B)
            # A车在最前方，B车在中间
            if i == 0:  # A车
                offset = 20  # A车在前方20米
                lane_offset = -7.0  # A车在左车道
            else:  # B车
                offset = 10  # B车在前方10米
                lane_offset = 3.5  # B车在右车道

            spawn_point_vehicle = carla.Transform(
                carla.Location(
                    x=spawn_point.location.x + offset * forward_vector.x,
                    y=spawn_point.location.y + offset * forward_vector.y + lane_offset,
                    z=spawn_point.location.z + 0.1  # 轻微提高高度避免碰撞
                ),
                spawn_point.rotation
            )

            # 尝试多个可能的位置，直到成功生成车辆
            max_attempts = 10
            vehicle = None
            for attempt in range(max_attempts):
                try:
                    vehicle = self.world.spawn_actor(vehicle_bp, spawn_point_vehicle)
                    print(f"成功生成车辆 {chr(65 + i)} (模型: {vehicle_bp.id})")
                    print(f"位置: x={spawn_point_vehicle.location.x}, y={spawn_point_vehicle.location.y}")
                    break
                except RuntimeError as e:
                    print(f"尝试 {attempt + 1}: 无法生成车辆 {chr(65 + i)}: {e}")
                    # 调整生成位置
                    spawn_point_vehicle.location.x += 1.0
                    spawn_point_vehicle.location.z += 0.1

            if vehicle is None:
                print(f"无法生成车辆 {chr(65 + i)}，跳过")
                continue

            # 设置速度：B车最快，A和C车速度相似
            if i == 1:  # B车速度最快
                initial_speed = self.base_speed * self.b_car_speed_factor
            else:  # A车速度与基础速度相同
                initial_speed = self.base_speed

            vehicle_info = {
                'actor': vehicle,
                'model': vehicle_bp.id,
                'initial_speed': initial_speed,
                'spawn_point': spawn_point_vehicle,
                'name': chr(65 + i)  # A或B
            }

            self.vehicle_details.insert(i, vehicle_info)
            self.vehicles.insert(i, vehicle)

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
        def _parse_image(self, image):
            # 创建可写的NumPy数组副本
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]

            # 创建可写副本
            bgr_array = np.copy(array[:, :, ::-1])

            # 添加车辆标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 第一辆车A绿色
            cv2.putText(bgr_array,
                        f"Vehicle A: {self.vehicle_details[0]['model']} (Left Lane)",
                        (50, 50),
                        font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # 第二辆车B红色
            cv2.putText(bgr_array,
                        f"Vehicle B: {self.vehicle_details[1]['model']} (Truck)",
                        (50, 100),
                        font, 1, (0, 0, 255), 2, cv2.LINE_AA)

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

        # 将 _parse_image 绑定为实例方法
        self._parse_image = _parse_image.__get__(self)

        # 注册监听器
        self.camera.listen(self._parse_image)

        # 创建交通管理器用于控制车辆行为
        self.traffic_manager = client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

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

        # 配置交通管理器，实现变道
        for i, vehicle_info in enumerate(self.vehicle_details):
            vehicle = vehicle_info['actor']

            # A车保持在左车道
            if vehicle_info['name'] == 'A':
                # 不变道，保持在左车道
                self.traffic_manager.auto_lane_change(vehicle, False)  # 禁止变道
                self.traffic_manager.force_lane_change(vehicle, False)  # 不强制变道
                self.traffic_manager.random_right_lanechange_percentage(vehicle, 0)  # 禁止右变道
                self.traffic_manager.random_left_lanechange_percentage(vehicle, 0)  # 禁止左变道
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 0)  # 保持速度，与C车相似

            elif vehicle_info['name'] == 'B':
                # B车保持在当前车道，并且速度最快
                self.traffic_manager.auto_lane_change(vehicle, False)  # 禁止变道
                self.traffic_manager.force_lane_change(vehicle, False)  # 不强制变道
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -30)  # 速度快30%

            elif vehicle_info['name'] == 'C':
                # C车保持在当前车道
                self.traffic_manager.auto_lane_change(vehicle, False)  # 禁止变道
                self.traffic_manager.force_lane_change(vehicle, False)  # 不强制变道
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 0)  # 保持与A车相似的速度

            # 通用设置
            self.traffic_manager.distance_to_leading_vehicle(vehicle, 5 + i * 2)
            self.traffic_manager.update_vehicle_lights(vehicle, True)

        # 录制设置
        self.recording_start_time = time.time()
        self.recording_duration = 25.0  # 25秒

        # 初始世界状态
        self.world.tick()

    def tick(self, clock):
        self.world.tick()

        # 定期检查并调整车辆速度和行为
        if self.current_frame % 30 == 0:  # 每秒调整一次（假设30fps）
            for vehicle_info in self.vehicle_details:
                vehicle = vehicle_info['actor']

                # 获取当前速度
                current_speed = vehicle_info.get('current_speed', 0)

                # 根据车辆类型调整速度
                if vehicle_info['name'] == 'A':
                    # A车保持基础速度
                    target_speed = self.base_speed

                elif vehicle_info['name'] == 'B':
                    # B车保持较高速度
                    target_speed = self.base_speed * self.b_car_speed_factor

                elif vehicle_info['name'] == 'C':
                    # C车速度与A相似
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

        # 检查录制是否完成
        if self.recording and (time.time() - self.recording_start_time) > self.recording_duration:
            print("Recording Details:")
            for vehicle_info in self.vehicle_details:
                print(f"Vehicle {vehicle_info['name']}:")
                print(f"  Model: {vehicle_info['model']}")
                print(f"  Initial Speed: {vehicle_info['initial_speed']:.2f} km/h")
                print(f"  Current Speed: {vehicle_info.get('current_speed', 'N/A'):.2f} km/h")
                print(f"  Location: {vehicle_info.get('location', 'N/A')}")
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

        world = World(client, args)

        clock = pygame.time.Clock()
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA Scenario")

        while True:
            clock.tick_busy_loop(60)
            world.tick(clock)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYUP:
                    if event.key == K_ESCAPE or event.key == K_q:
                        return

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