#!/usr/bin/env python

import argparse
import logging
import time
import os
import numpy as np
import pygame
import carla
import cv2
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
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if item != '':
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18


class World(object):
    def __init__(self, client, hud):
        self.world = None
        self.client = client
        self.hud = hud
        self.player = None
        self.camera = None
        self.display = None
        self.image = None
        self.frame = 0
        self.simulation_time = 0
        self.server_fps = 0
        self.vehicles = []
        self.vehicle_a = None
        self.vehicle_b = None
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

        # 加载最简单的地图 - Town01或Town02通常比较简单
        self.client.load_world('Town01')
        self.world = self.client.get_world()

        # 移除现有的车辆
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor.destroy()

        # 设置同步模式，使帧率更稳定
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 30.0
        self.world.apply_settings(settings)

        # 获取蓝图库
        blueprint_library = self.world.get_blueprint_library()

        # 找到一个好的车道
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]  # 使用第一个生成点，通常是大道

        # 车辆C - 主视角车辆（ego vehicle）
        vehicle_bp = blueprint_library.filter('model3')[0] if blueprint_library.filter('model3') else \
        blueprint_library.filter('vehicle.*')[0]
        self.player = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicles.append(self.player)
        print(f"Spawned vehicle C (ego): {self.player.type_id}")

        # 车辆A - 在C的前方约15米
        vehicle_bp_a = blueprint_library.filter('audi')[0] if blueprint_library.filter('audi') else \
        blueprint_library.filter('vehicle.*')[0]
        spawn_point_a = carla.Transform(
            carla.Location(
                x=spawn_point.location.x + 15 * spawn_point.get_forward_vector().x,
                y=spawn_point.location.y + 15 * spawn_point.get_forward_vector().y,
                z=spawn_point.location.z
            ),
            spawn_point.rotation
        )
        self.vehicle_a = self.world.spawn_actor(vehicle_bp_a, spawn_point_a)
        self.vehicles.append(self.vehicle_a)
        print(f"Spawned vehicle A: {self.vehicle_a.type_id}")

        # 车辆B - 在A和C之间，离C约7米
        vehicle_bp_b = blueprint_library.filter('mustang')[0] if blueprint_library.filter('mustang') else \
        blueprint_library.filter('vehicle.*')[0]
        spawn_point_b = carla.Transform(
            carla.Location(
                x=spawn_point.location.x + 7 * spawn_point.get_forward_vector().x,
                y=spawn_point.location.y + 7 * spawn_point.get_forward_vector().y,
                z=spawn_point.location.z
            ),
            spawn_point.rotation
        )
        self.vehicle_b = self.world.spawn_actor(vehicle_bp_b, spawn_point_b)
        self.vehicles.append(self.vehicle_b)
        print(f"Spawned vehicle B: {self.vehicle_b.type_id}")

        # 创建并安装相机（视角抬高）
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(hud.dim[0]))
        camera_bp.set_attribute('image_size_y', str(hud.dim[1]))
        camera_bp.set_attribute('fov', '100')  # 增大视场角

        # 设置驾驶员视角位置，高度抬高
        camera_transform = carla.Transform(carla.Location(x=-2.0, z=3.0), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.player)

        # 设置回调函数处理相机图像
        self.camera.listen(self._parse_image)

        # 更新HUD的车辆信息
        self.hud.vehicle_info = {
            'name': self.player.type_id,
            'speed': 0.0
        }

        # 为所有车辆设置自动驾驶，并确保它们沿着同一路径行驶
        for vehicle in self.vehicles:
            vehicle.set_autopilot(True)

        # 设置录制时间
        self.recording_start_time = time.time()
        self.recording_duration = 10.0  # 10秒钟

        # 等待一帧以确保世界已更新
        self.world.tick()

    def tick(self, clock):
        if self.player:
            velocity = self.player.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            self.hud.vehicle_info['speed'] = speed

        # 让世界更新
        self.world.tick()

        snapshot = self.world.get_snapshot()
        self.frame = snapshot.frame
        self.simulation_time = snapshot.timestamp.elapsed_seconds
        self.server_fps = 1.0 / snapshot.timestamp.delta_seconds if snapshot.timestamp.delta_seconds > 0 else 0
        self.hud.tick(self, clock)

        # 检查是否应该结束录制
        if self.recording and (time.time() - self.recording_start_time) > self.recording_duration:
            print(f"Recording completed after {self.recording_duration} seconds")
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"Video saved to {self.video_dir}/recording.mp4")

    def render(self, display):
        if self.image is not None:
            display.blit(self.image, (0, 0))
        self.hud.render(display)

    def _parse_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]

        # 为OpenCV转换颜色通道顺序（RGB到BGR）
        # 修复错误：创建数组的可写副本
        bgr_array = np.copy(array[:, :, ::-1])

        # 添加标签到图像上
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bgr_array, "Vehicle A", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr_array, "Vehicle B", (50, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr_array, "Vehicle C (Ego)", (50, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 保存图像帧到文件
        if self.recording:
            frame_filename = os.path.join(self.img_dir, f"frame_{self.current_frame:06d}.png")
            cv2.imwrite(frame_filename, bgr_array)

            # 初始化视频写入器（如果还没初始化）
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    os.path.join(self.video_dir, 'recording.mp4'),
                    fourcc,
                    self.fps,
                    (image.width, image.height)
                )

            # 写入视频帧
            self.video_writer.write(bgr_array)
            self.current_frame += 1

        # 转换为Pygame表面
        array = array[:, :, ::-1]  # 转回RGB给Pygame
        self.image = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def destroy(self):
        # 恢复异步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        # 清理视频写入器
        if self.video_writer is not None:
            self.video_writer.release()

        # 销毁所有车辆和传感器
        actors = [self.camera] + self.vehicles
        for actor in actors:
            if actor is not None:
                actor.destroy()


class Controller(object):
    def __init__(self, world):
        self.world = world
        self.clock = pygame.time.Clock()
        self._running = True
        self._main_loop()

    def _main_loop(self):
        try:
            while self._running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                    elif event.type == pygame.KEYUP:
                        if event.key == K_ESCAPE or event.key == K_q:
                            self._running = False

                # 游戏循环更新
                self.clock.tick_busy_loop(60)
                self.world.tick(self.clock)
                self.world.render(self.world.display)
                pygame.display.flip()

                # 如果录制完成且仍在运行，显示结束信息
                if not self.world.recording and self._running:
                    font = pygame.font.Font(None, 36)
                    text = font.render("Recording completed, press ESC to exit", True, (255, 255, 255))
                    self.world.display.blit(text, (self.world.hud.dim[0] // 2 - 180, self.world.hud.dim[1] - 50))
                    pygame.display.flip()

        finally:
            # 退出时清理
            self.world.destroy()
            pygame.quit()


def game_loop(args):
    pygame.init()
    pygame.font.init()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Multi-Vehicle Recording")

        hud = HUD(args.width, args.height)
        world = World(client, hud)
        world.display = display

        controller = Controller(world)

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Multi-Vehicle Recording')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        default=True,
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print("CARLA Multi-Vehicle Recording Script")
    print("- Recording 10 seconds of gameplay")
    print("- Frames saved to 'img' folder")
    print("- Video saved to 'video' folder")
    print("- Press ESC or Q to quit")

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    try:
        import sys
        import cv2

        main()
    except ImportError:
        print("ERROR: Please install the required libraries:")
        print("pip install numpy pygame opencv")