import argparse
import time
import carla
import numpy as np
import cv2
import pygame
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
        pygame.display.set_caption("Single Vehicle Simulation")

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
