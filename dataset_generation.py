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

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = []
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self.vehicle_info = {}

    def tick(self, world, clock):
        self._notifications.clear()
        self.frame = world.frame
        self.simulation_time = world.simulation_time
        self.server_fps = world.server_fps
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Time:    % 16.0f s' % self.simulation_time,
            'Vehicles: A, B, C (Ego)',
            'Recording: % 16.1f s' % (10.0 - (time.time() - world.recording_start_time)),
            '']
        if len(self.vehicle_info) > 0:
            self._info_text += [
                'Ego Vehicle: % 20s' % self.vehicle_info.get('name', ''),
                'Speed:   % 15.0f km/h' % (self.vehicle_info.get('speed', 0.0) * 3.6),
                '']

    def render(self, display):
        pass  # 不再渲染HUD

class World(object):
    def __init__(self, client, hud, args):
        self.args = args
        self.world = None
        self.client = client
        self.hud = hud
        self.player = None
        self.camera = None
        self.recording = True
        self.current_frame = 0

        # 创建保存目录
        self.img_dir = 'img'
        self.video_dir = 'video'
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # 录像设置
        self.fps = 30.0
        self.video_writer = None

        # 加载复杂地图
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
        spawn_point = spawn_points[0]

        # 基础速度设置
        base_speed = random.uniform(40, 60)
        speed_variation = 5

        # 车辆生成
        vehicle_models = ['model3', 'audi', 'mustang']
        self.vehicles = []

        for i, model in enumerate(vehicle_models):
            vehicle_bp = blueprint_library.filter(model)[0]
            
            # 稍微分散生成点
            offset = i * 15
            spawn_point_vehicle = carla.Transform(
                carla.Location(
                    x=spawn_point.location.x + offset * spawn_point.get_forward_vector().x,
                    y=spawn_point.location.y + offset * spawn_point.get_forward_vector().y,
                    z=spawn_point.location.z
                ),
                spawn_point.rotation
            )

            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point_vehicle)
            self.vehicles.append(vehicle)

            # 设置速度
            speed = base_speed + random.uniform(-speed_variation, speed_variation)
            forward_vector = spawn_point_vehicle.get_forward_vector()
            velocity = carla.Vector3D(
                x=forward_vector.x * speed / 3.6,
                y=forward_vector.y * speed / 3.6,
                z=0
            )
            vehicle.set_target_velocity(velocity)

            # 主车为最后一辆
            if i == len(vehicle_models) - 1:
                self.player = vehicle

        # KITTI风格相机
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.width))
        camera_bp.set_attribute('image_size_y', str(args.height))
        camera_bp.set_attribute('fov', '110')

        # KITTI标准相机位置
        camera_transform = carla.Transform(
            carla.Location(x=1.5, y=0.5, z=2.0),  # 右侧偏置，高2米
            carla.Rotation(pitch=-10, yaw=0)  # 略微向下10度
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.player)
        self.camera.listen(self._parse_image)

        # 录制设置
        self.recording_start_time = time.time()
        self.recording_duration = 10.0

        # 初始世界状态
        self.world.tick()

    def _parse_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        bgr_array = array[:, :, ::-1]

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

class Controller(object):
    def __init__(self, world):
        self.world = world
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("KITTI-style CARLA Recording")
        self._running = True
        self._main_loop()

    def _main_loop(self):
        try:
            while self._running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                       (event.type == pygame.KEYUP and 
                        (event.key == K_ESCAPE or event.key == K_q)):
                        self._running = False

                self.clock.tick_busy_loop(60)
                self.world.tick(self.clock)

                if not self.world.recording and self._running:
                    print("Recording completed. Press ESC to exit.")
                    break

        finally:
            self.world.destroy()
            pygame.quit()

def game_loop(args):
    pygame.init()
    pygame.font.init()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        # 不再创建显示窗口，因为我们只关注相机录制
        hud = HUD(args.width, args.height)
        world = World(client, hud, args)
        
        controller = Controller(world)

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()

def main():
    argparser = argparse.ArgumentParser(description='KITTI-style CARLA Recording')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='Server host')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='Server port')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='Image resolution')
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    print("KITTI-style CARLA Recording Script")
    print(f"- Recording {10} seconds of gameplay")
    print("- Frames saved to 'img' folder")
    print("- Video saved to 'video' folder")

    game_loop(args)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')