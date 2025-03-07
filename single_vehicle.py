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