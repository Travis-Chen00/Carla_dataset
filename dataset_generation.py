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

        # 前方三辆车的车型
        vehicle_models = ['model3', 'audi', 'mustang']

        # 基础速度设置（保持一致性）
        base_speed = random.uniform(45, 55)  # 45-55 km/h

        # 存储车辆详细信息的列表
        self.vehicle_details = []

        # 生成前方三辆车
        for i, model in enumerate(vehicle_models):
            vehicle_bp = blueprint_library.filter(model)[0]
            
            # 精确控制车辆间距，间隔约5米
            offset = (i + 1) * 5  # 每辆车间隔5米
            spawn_point_vehicle = carla.Transform(
                carla.Location(
                    x=spawn_point.location.x + offset * spawn_point.get_forward_vector().x,
                    y=spawn_point.location.y + offset * spawn_point.get_forward_vector().y,
                    z=spawn_point.location.z
                ),
                spawn_point.rotation
            )

            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point_vehicle)
            
            # 设置速度（微小波动，保持基本一致）
            speed = base_speed + random.uniform(-1, 1)
            forward_vector = spawn_point_vehicle.get_forward_vector()
            velocity = carla.Vector3D(
                x=forward_vector.x * speed / 3.6,
                y=forward_vector.y * speed / 3.6,
                z=0
            )
            vehicle.set_target_velocity(velocity)
            
            # 设置自动驾驶
            vehicle.set_autopilot(True)
            
            # 存储车辆详细信息
            vehicle_info = {
                'actor': vehicle,
                'model': model,
                'initial_speed': speed,
                'spawn_point': spawn_point_vehicle,
                'name': chr(65 + i)  # A, B, C
            }
            self.vehicle_details.append(vehicle_info)
            self.vehicles.append(vehicle)

        # KITTI风格相机 - 安装在最后一辆车（C车）上
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.width))
        camera_bp.set_attribute('image_size_y', str(args.height))
        camera_bp.set_attribute('fov', '110')

        # KITTI标准相机位置 - 精确调整
        camera_transform = carla.Transform(
            carla.Location(x=1.5, y=-0.3, z=1.7),  # 左侧微微偏置，高度接近人眼
            carla.Rotation(pitch=-5, yaw=0)  # 略微向下
        )
        self.camera = self.world.spawn_actor(
            camera_bp, 
            camera_transform, 
            attach_to=self.vehicles[-1]  # 附加到最后一辆车（C车）
        )
        self.camera.listen(self._parse_image)

        # 录制设置
        self.recording_start_time = time.time()
        self.recording_duration = 10.0

        # 初始世界状态
        self.world.tick()

    def _parse_image(self, image):
        # 创建可写的NumPy数组副本
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        
        # 创建可写副本
        bgr_array = np.copy(array[:, :, ::-1])

        # 添加车辆标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, vehicle_info in enumerate(self.vehicle_details[:-1]):  # 不包括最后一辆车（C车）
            color = (0, 255, 0) if i == 0 else (0, 0, 255)  # A绿色，B红色
            cv2.putText(bgr_array, 
                        f"Vehicle {vehicle_info['name']}: {vehicle_info['model']}", 
                        (50, 50 + i*50), 
                        font, 1, color, 2, cv2.LINE_AA)

        # 记录车辆详细信息
        for vehicle_info in self.vehicle_details:
            vehicle = vehicle_info['actor']
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
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

    def tick(self, clock):
        self.world.tick()

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
        print('\nExit.')

if __name__ == '__main__':
    main()