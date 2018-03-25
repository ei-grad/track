#!/usr/bin/env python

import logging
import math
import os

import numpy as np
from shapely.geometry import LineString, Point

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
    def __init__(self):
        super(App, self).__init__()

        self.screen = None

        self.track_image = load_svg('track.svg')
        self.car_image = load_svg('car.svg')

        self.size = self.weight, self.height = self.track_image.get_size()
        self.ppm = 2.7
        self.target_fps = 60
        self.time_step = 1. / self.target_fps
        self.clock = pygame.time.Clock()

        paths, attributes = svg2paths('track.svg')
        self.svg_paths = {j['id']: i for i, j in zip(paths, attributes)}
        track_start = self.svg_paths['track'].start
        self.start_coord = Point(track_start.real, track_start.imag)
        self.lbound_linestring = traced_path(self.svg_paths['lbound'])
        self.rbound_linestring = traced_path(self.svg_paths['rbound'])

        # self.checkpoints = [
        #     b2.vec2(i.start.real / self.ppm,
        #             (self.size[1] - i.start.imag) / self.ppm)
        #     for i in self.svg_paths['track']
        # ]
        self.checkpoints = [
            b2.vec2(i.real / self.ppm,
                    (self.size[1] - i.imag) / self.ppm)
            for i in [self.svg_paths['track'].point(i) for i in np.linspace(0, 1)]
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
        # TODO: implement ground
        # for i in boundary.fixtures:
        #     i.sensor = True

        self.car = None
        self.init_car()

        # TODO: make front-wheel / rear drive switch
        # self.car.tires[0].max_drive_force = self.car.tires[1].max_drive_force = 0
        # self.car.tires[2].max_drive_force = self.car.tires[3].max_drive_force = 0

        self.key_map = {
            pygame.K_w: 'up',
            pygame.K_s: 'down',
            pygame.K_a: 'left',
            pygame.K_d: 'right',
        }
        self.pressed_keys = set()

    def BeginContact(self, contact):
        self.car.body.linearVelocity *= 0.7

    def EndContact(self, contact):
        pass

    def init_screen(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)

    def init_car(self):

        if self.car is not None:
            self.car.destroy()

        self.car = TDCar(
            self.world,
            position=b2.vec2(self.start_coord.x / self.ppm,
                             (self.size[1] - self.start_coord.y) / self.ppm),
            tire_kwargs=dict(
                dimensions=(0.2, 0.8),
                max_forward_speed=200.0,
                max_backward_speed=-80.,
                max_drive_force=300.,
            ),
            density=0.08,
        )
        self.car.next_checkpoint = 1
        self.car.laps = 0

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

        self.draw_ray(*self.get_ray(0))
        self.draw_ray(*self.get_ray(1))

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

        font = pygame.font.Font(None, 30)
        fps = font.render("FPS:%d %.2f %.2f %.2f %.2f" % ((self.clock.get_fps(),) + self.get_state()),
                          True, pygame.Color('white'))
        self.screen.blit(fps, (50, 50))

    def draw_ray(self, p0, p1):
        pygame.draw.aaline(
            self.screen, (0, 0, 255, 255),
            self.b2_to_pygame_point(p0),
            self.b2_to_pygame_point(p0+(p1-p0)),
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

    def execute(self, action):
        self.update(set([['up', 'down', 'left', 'right'][action]]))
        d = self.checkpoints[self.car.next_checkpoint] - self.car.body.worldCenter
        if d.length < self.checkpoint_radius:
            self.car.next_checkpoint = (self.car.next_checkpoint + 1) % len(self.checkpoints)
            if self.car.next_checkpoint == 0:
                self.car.laps += 1
            return 1
        return 0

    def get_ray(self, num):
        if num:
            b = self.car.tires[3].body
            x = math.radians(20)
        else:
            b = self.car.tires[2].body
            x = math.radians(-20)
        v = self.car.body.GetWorldVector((100. * math.sin(x), 100. * math.cos(x)))
        return b.worldCenter, b.worldCenter + v

    def get_state(self):
        ret = []
        for i in (0, 1):
            p0, p1 = self.get_ray(i)
            p2, frac = self.get_boundary_intersection(p0, p1)
            if p2 is not None:
                ret.append(frac)
            else:
                ret.append(1.0)
        b = self.car.body
        max_speed = self.car.tires[0].max_forward_speed
        ret.append(b.GetWorldVector((0, 1)).dot(b.linearVelocity) / max_speed)
        ret.append(b.GetWorldVector((1, 0)).dot(b.linearVelocity) / max_speed)
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
