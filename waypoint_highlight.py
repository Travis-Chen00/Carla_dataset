import carla
import pygame
import numpy as np
import math

def visualize_carla_spawn_points():
    # 连接到CARLA服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # 获取世界和地图
    client.load_world('Town05')
    world = client.get_world()
    carla_map = world.get_map()

    # 获取所有生成点
    spawn_points = carla_map.get_spawn_points()
    print(f"找到 {len(spawn_points)} 个生成点")

    # 初始化pygame
    pygame.init()

    # 设置地图大小和比例尺
    map_width = 1024
    map_height = 768
    screen = pygame.display.set_mode((map_width, map_height))
    pygame.display.set_caption('CARLA 生成点可视化')

    # 计算地图边界以确定比例尺
    spawn_locations = np.array([[sp.location.x, sp.location.y] for sp in spawn_points])
    min_x, min_y = np.min(spawn_locations, axis=0) - 50
    max_x, max_y = np.max(spawn_locations, axis=0) + 50

    # 计算比例尺
    scale_x = map_width / (max_x - min_x)
    scale_y = map_height / (max_y - min_y)
    scale = min(scale_x, scale_y) * 0.9  # 留出一些边距

    def world_to_screen(world_pos):
        """将世界坐标转换为屏幕坐标"""
        screen_x = (world_pos[0] - min_x) * scale
        screen_y = map_height - (world_pos[1] - min_y) * scale
        return int(screen_x), int(screen_y)

    # 主循环
    running = True
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 10)

    # 定义颜色
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 添加额外的键盘事件处理
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 填充屏幕为黑色（背景）
        screen.fill(BLACK)

        # 绘制所有生成点
        for i, spawn_point in enumerate(spawn_points):
            # 获取生成点的位置和朝向
            x, y = spawn_point.location.x, spawn_point.location.y
            yaw = math.radians(spawn_point.rotation.yaw)

            # 计算前向向量端点
            forward_x = x + 5 * math.cos(yaw)
            forward_y = y + 5 * math.sin(yaw)

            # 转换为屏幕坐标
            screen_pos = world_to_screen((x, y))
            screen_forward = world_to_screen((forward_x, forward_y))

            # 绘制红色点表示生成点
            pygame.draw.circle(screen, RED, screen_pos, 3)

            # 绘制方向线（蓝色）
            pygame.draw.line(screen, BLUE, screen_pos, screen_forward, 1)

            # 显示生成点编号
            text_surface = font.render(str(i), True, WHITE)
            screen.blit(text_surface, (screen_pos[0] + 5, screen_pos[1] - 5))

        # 显示总生成点数量
        total_points_text = font.render(f'Total Spawn Points: {len(spawn_points)}', True, WHITE)
        screen.blit(total_points_text, (10, 10))

        # 更新显示
        pygame.display.flip()
        clock.tick(60)  # 限制帧率为60FPS

    # 退出Pygame
    pygame.quit()

# 运行可视化
if __name__ == '__main__':
    visualize_carla_spawn_points()