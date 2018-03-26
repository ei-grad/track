#!/usr/bin/env python

from os import getenv
import logging
import math
import os

import numpy as np
from shapely.geometry import LineString

import pygame

from Box2D import b2

import cairo

import gi
gi.require_version('Rsvg', '2.0')  # noqa
from gi.repository import Rsvg

from svgpathtools import svg2paths

from top_down_car import TDCar


def load_svg(filename):
    svg = Rsvg.Handle.new_from_file(filename)
    dims = svg.get_dimensions()
    data = np.zeros((dims.width, dims.height, 4), dtype=np.uint8)
    surface = cairo.ImageSurface.create_for_data(
        data, cairo.FORMAT_ARGB32,
        dims.width, dims.height,
        dims.width * 4
    )
    ctx = cairo.Context(surface)
    svg.render_cairo(ctx)
    # XXX: blue / red is broken
    # data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3] = \
    #     data[:, :, 2], data[:, :, 1], data[:, :, 0], data[:, :, 3]
    return pygame.image.frombuffer(data.tostring(), (dims.width, dims.height), "RGBA")


def traced_path(path):
    points = []
    X = np.linspace(0, 1)
    for segment in path:
        trace = segment.poly()(X)
        for i in trace:
            points.append((i.real, i.imag))
    if points[-1] != points[0]:
        points.append(points[0])
    return LineString(points).simplify(0.25)


def rot_center(image, angle):
    """rotate a Surface, maintaining position."""
    loc = image.get_rect().center
    rot_sprite = pygame.transform.rotate(image, angle)
    rot_sprite.get_rect().center = loc
    return rot_sprite


class App(b2.contactListener):
    def __init__(self, track_filename=getenv('TRACK', 'track.svg')):
        super(App, self).__init__()

        self.screen = None

        self.track_image = load_svg(track_filename)
        self.car_image = load_svg('car.svg')

        self.size = self.weight, self.height = self.track_image.get_size()
        self.ppm = 2.7
        self.target_fps = 20
        self.time_step = 1. / self.target_fps
        self.clock = pygame.time.Clock()

        paths, attributes = svg2paths(track_filename)
        self.svg_paths = {j['id']: i for i, j in zip(paths, attributes)}
        self.lbound_linestring = traced_path(self.svg_paths['lbound'])
        self.rbound_linestring = traced_path(self.svg_paths['rbound'])

        self.ray_angles = [
            0, 180,
            10, -10,
            90, -90,
            30, -30,
            45, -45,
            135, -135,
        ]
        self.ray_length = 100.
        self.rays = [
            b2.vec2(self.ray_length * math.sin(x),
                    self.ray_length * math.cos(x))
            for x in [math.radians(i) for i in self.ray_angles]
        ]

        # self.checkpoints = [
        #     b2.vec2(i.start.real / self.ppm,
        #             (self.size[1] - i.start.imag) / self.ppm)
        #     for i in self.svg_paths['track']
        # ]
        self.checkpoints = [
            b2.vec2(i.real / self.ppm,
                    (self.size[1] - i.imag) / self.ppm)
            for i in [self.svg_paths['track'].point(i)
                      for i in np.linspace(0, 1, endpoint=False)]
        ]
        self.checkpoint_radius = 20.

        self.world = b2.world(gravity=(0, 0), doSleep=True)
        self.world.contactListener = self

        self.boundary = self.world.CreateStaticBody()
        self.boundary.CreateEdgeChain([
            (x / self.ppm, (self.size[1] - y) / self.ppm)
            for x, y in self.lbound_linestring.coords
        ])
        self.boundary.CreateEdgeChain([
            (x / self.ppm, (self.size[1] - y) / self.ppm)
            for x, y in self.rbound_linestring.coords
        ])
        for i in self.boundary.fixtures:
            # TODO: implement ground
            # i.sensor = True
            i.friction = 0.9

        self.car = None
        self.init_car()

        self.key_map = {
            pygame.K_w: 'up',
            pygame.K_s: 'down',
            pygame.K_a: 'left',
            pygame.K_d: 'right',
        }
        self.pressed_keys = set()

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass

    def init_screen(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)

    def init_car(self, angle=None):

        if self.car is not None:
            self.car.destroy()

        if angle is None:
            # XXX: make random switch
            # angle = np.random() * math.pi * 2.
            p = self.checkpoints[1] - self.checkpoints[0]
            angle = math.atan2(p.y, p.x) - math.pi / 2.

        self.car = TDCar(
            self.world,
            position=b2.vec2(self.checkpoints[0].x, self.checkpoints[0].y),
            angle=angle,
            tire_kwargs=dict(
                dimensions=(0.2, 0.8),
                max_forward_speed=40.0,
                max_backward_speed=-10.0,
                max_drive_force=280.,
            ),
            density=0.08,
            rays=self.rays,
        )

        # TODO: make front-wheel / rear drive switch
        # self.car.tires[0].max_drive_force = self.car.tires[1].max_drive_force = 0
        # self.car.tires[2].max_drive_force = self.car.tires[3].max_drive_force = 0

        while (
                self.checkpoints[self.car.next_checkpoint] - self.car.body.worldCenter
        ).length < self.checkpoint_radius:
            self.car.next_checkpoint += 1

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
            ):
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in self.key_map:
                    self.pressed_keys.add(self.key_map[event.key])
            elif event.type == pygame.KEYUP:
                if event.key in self.key_map and self.key_map[event.key] in self.pressed_keys:
                    self.pressed_keys.remove(self.key_map[event.key])
        return True

    def update(self, pressed_keys):
        self.car.update(pressed_keys, self.target_fps)
        d = self.checkpoints[self.car.next_checkpoint] - self.car.body.worldCenter
        if d.length < self.checkpoint_radius:
            self.car.next_checkpoint = (self.car.next_checkpoint + 1) % len(self.checkpoints)
            if self.car.next_checkpoint == 0:
                self.car.laps += 1
        self.world.Step(self.time_step, 1, 1)
        self.world.ClearForces()

    def render(self):
        self.screen.blit(self.track_image, (0, 0))
        for tire in self.car.tires:
            body = tire.body
            for fixture in body:
                shape = fixture.shape
                vertices = [(body.transform * v) * self.ppm for v in shape.vertices]
                vertices = [(v[0], self.size[1] - v[1]) for v in vertices]
                pygame.draw.polygon(self.screen, (0, 0, 0, 255), vertices)
        b = self.car.body
        angle = math.degrees(b.angle)
        if abs(angle) < 2.:
            angle = 0.
        elif abs(angle - 180.) < 2.:
            angle = 180.
        img = rot_center(self.car_image, angle)
        self.screen.blit(
            img,
            ((b.worldCenter[0]) * self.ppm - img.get_rect().center[0],
             (self.size[1] - (b.worldCenter[1]) * self.ppm) - img.get_rect().center[1])
        )

        if 'DEBUG' in os.environ:
            self.render_debug()

        pygame.display.flip()

    def render_debug(self):

        for body in self.world.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = [(body.transform * v) * self.ppm for v in shape.vertices]
                vertices = [(v[0], self.size[1] - v[1]) for v in vertices]
                pygame.draw.lines(self.screen, (0, 0, 255, 255), True, vertices)

        for p0, p1 in self.car.rays:
            self.draw_ray(p0, p1)

        for n, i in enumerate(self.checkpoints):
            if n == self.car.next_checkpoint:
                color = (0, 255, 0, 255)
            else:
                color = (255, 0, 0, 255)
            pygame.draw.circle(
                self.screen, color,
                self.b2_to_pygame_point(i),
                int(self.checkpoint_radius * self.ppm),
                1
            )

        font = pygame.font.Font(None, 20)
        state = self.get_state()
        fmt = "STATE: " + " ".join("%.2f" for i in range(len(state)))
        self.screen.blit(font.render(
            fmt % state, True, pygame.Color('white')
        ), (20, 20))
        self.screen.blit(font.render(
            "FPS: %d" % self.clock.get_fps(), True, pygame.Color('white')
        ), (20, 40))

    def draw_ray(self, p0, p1):
        pygame.draw.line(
            self.screen, (0, 0, 255, 255),
            self.b2_to_pygame_point(p0),
            self.b2_to_pygame_point(p1),
        )
        p2, _ = self.get_boundary_intersection(p0, p1)
        if p2 is not None:
            pygame.draw.circle(
                self.screen, (255, 0, 0, 255),
                self.b2_to_pygame_point(p2),
                3,
            )

    def b2_to_pygame_point(self, p):
        return (int(p[0] * self.ppm), int(self.size[1] - p[1] * self.ppm))

    def cleanup(self):
        pygame.quit()

    def play(self):
        self.init_screen()
        while True:
            if not self.check_events():
                break
            self.clock.tick(self.target_fps)
            self.update(self.pressed_keys)
            self.render()
        self.cleanup()

    def execute(self, actions):
        buttons = set()
        if actions['accel'] == 1:
            buttons.add('up')
        elif actions['accel'] == 2:
            buttons.add('down')
        if actions['turn'] == 1:
            buttons.add('left')
        elif actions['turn'] == 2:
            buttons.add('right')
        checkpoint = self.car.next_checkpoint
        self.update(buttons)
        if checkpoint != self.car.next_checkpoint:
            return 1
        return 0

    def get_state(self):

        ret = []

        max_speed = self.car.tires[0].max_forward_speed
        b = self.car.body
        cp = self.checkpoints[self.car.next_checkpoint] - b.position

        # state[0] - forward speed
        ret.append(b.GetWorldVector((0, 1)).dot(b.linearVelocity) / max_speed)

        # state[1] - lateral speed
        ret.append(b.GetWorldVector((1, 0)).dot(b.linearVelocity) / max_speed)

        # state[2] - angular speed
        ret.append(b.angularVelocity)

        # state[3] - distance to checkpoint
        ret.append(min(cp.length - self.checkpoint_radius, self.ray_length) / self.ray_length)

        # state[4] - angle to checkpoint
        cp_angle = math.atan2(cp.y, cp.x)
        b_angle = (b.angle + math.pi / 2.) % (math.pi * 2.)
        if b_angle > math.pi:
            b_angle = - (math.pi * 2 - b_angle)
        angle = (cp_angle - b_angle) % (math.pi * 2)
        if angle > math.pi:
            angle = - (math.pi * 2 - angle)
        ret.append(angle / math.pi)

        # distances to track boundary
        for p0, p1 in self.car.rays:
            p2, frac = self.get_boundary_intersection(p0, p1)
            if p2 is not None:
                ret.append(frac)
            else:
                ret.append(1.0)

        return tuple(ret)

    def get_boundary_intersection(self, p1, p2):
        input = b2.rayCastInput(p1=p1, p2=p2, maxFraction=1)
        output = b2.rayCastOutput()
        closest_fraction = None
        closest_point = None
        for i in self.boundary.fixtures:
            if i.RayCast(output, input, 0):
                hit_point = input.p1 + output.fraction * (input.p2 - input.p1)
                if closest_point is None or closest_fraction > output.fraction:
                    closest_point = hit_point
                    closest_fraction = output.fraction
        return closest_point, closest_fraction


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app = App()
    app.play()
