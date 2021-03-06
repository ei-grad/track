#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Based on Chris Campbell's tutorial from iforce2d.net:
http://www.iforce2d.net/b2dtut/top-down-car
"""

import math

from Box2D import b2


class TDTire(object):

    def __init__(self, car, max_forward_speed=100.0,
                 max_backward_speed=-20, max_drive_force=150,
                 turn_torque=15, max_lateral_impulse=3,
                 dimensions=(0.5, 1.25), density=1.0,
                 position=(0, 0), angle=0):

        world = car.body.world

        self.current_traction = 1
        self.turn_torque = turn_torque
        self.max_forward_speed = max_forward_speed
        self.max_backward_speed = max_backward_speed
        self.max_drive_force = max_drive_force
        self.max_lateral_impulse = max_lateral_impulse
        self.ground_areas = []

        self.body = world.CreateDynamicBody(position=position, angle=angle)
        self.body.CreatePolygonFixture(box=dimensions, density=density)
        self.body.userData = {'obj': self}

    @property
    def forward_velocity(self):
        body = self.body
        current_normal = body.GetWorldVector((0, 1))
        return current_normal.dot(body.linearVelocity) * current_normal

    @property
    def lateral_velocity(self):
        body = self.body

        right_normal = body.GetWorldVector((1, 0))
        return right_normal.dot(body.linearVelocity) * right_normal

    def update_friction(self):
        impulse = -self.lateral_velocity * self.body.mass
        if impulse.length > self.max_lateral_impulse:
            impulse *= self.max_lateral_impulse / impulse.length

        self.body.ApplyLinearImpulse(self.current_traction * impulse,
                                     self.body.worldCenter, True)

        aimp = self.current_traction * \
            self.body.inertia * -self.body.angularVelocity
        self.body.ApplyAngularImpulse(aimp, True)

        current_forward_normal = self.forward_velocity
        current_forward_speed = current_forward_normal.Normalize()

        drag_force_magnitude = -2 * current_forward_speed
        self.body.ApplyForce(self.current_traction * drag_force_magnitude * current_forward_normal,
                             self.body.worldCenter, True)

    def update_drive(self, keys):

        if 'up' in keys:
            desired_speed = self.max_forward_speed
        elif 'down' in keys:
            desired_speed = self.max_backward_speed
        else:
            return

        # find the current speed in the forward direction
        current_forward_normal = self.body.GetWorldVector((0, 1))
        current_speed = current_forward_normal.dot(self.body.linearVelocity)

        # apply necessary force
        ds = abs(desired_speed - current_speed)
        if ds > abs(desired_speed):
            ds = abs(desired_speed)
        force = self.max_drive_force * ds / desired_speed

        self.body.ApplyForce(self.current_traction * force * current_forward_normal,
                             self.body.worldCenter, True)

    def update_turn(self, keys):
        if 'left' in keys:
            desired_torque = self.turn_torque
        elif 'right' in keys:
            desired_torque = -self.turn_torque
        else:
            return

        self.body.ApplyTorque(desired_torque, True)

    def add_ground_area(self, ud):
        if ud not in self.ground_areas:
            self.ground_areas.append(ud)
            self.update_traction()

    def remove_ground_area(self, ud):
        if ud in self.ground_areas:
            self.ground_areas.remove(ud)
            self.update_traction()

    def update_traction(self):
        if not self.ground_areas:
            self.current_traction = 1
        else:
            self.current_traction = 0
            mods = [ga.friction_modifier for ga in self.ground_areas]

            max_mod = max(mods)
            if max_mod > self.current_traction:
                self.current_traction = max_mod


class TDCar(object):
    vertices = [
        (1.5, -2.0),
        (3.2, -1.5),
        (3.2, 11.0),
        (1.0, 12.5),
        (-1.0, 12.5),
        (-3.2, 11.0),
        (-3.2, -1.5),
        (-1.5, -2.0),
    ]

    tire_anchors = [
        (-3.0, 0.5),
        (3.0, 0.5),
        (-3.0, 9.0),
        (3.0, 9.0),
    ]

    def __init__(self, world, vertices=None,
                 tire_anchors=None, density=0.1,
                 position=(0, 0), angle=0.,
                 tire_kwargs=None,
                 rays=[]):

        self.world = world

        if vertices is None:
            vertices = TDCar.vertices

        self.body = world.CreateDynamicBody(position=position, angle=angle)
        self.body.CreatePolygonFixture(vertices=vertices, density=density, friction=0.9)
        self.body.userData = {'obj': self}

        self.max_tires_angle = math.radians(40.)

        if tire_kwargs is None:
            tire_kwargs = {}
        self.tires = [
            TDTire(self,
                   position=self.body.transform * i,
                   angle=angle,
                   **tire_kwargs)
            for i in self.tire_anchors
        ]
        for tire in self.tires:
            for fixture in tire.body.fixtures:
                fixture.sensor = True

        if tire_anchors is None:
            anchors = TDCar.tire_anchors

        joints = self.joints = []
        for tire, anchor in zip(self.tires, anchors):
            j = world.CreateRevoluteJoint(
                bodyA=self.body,
                bodyB=tire.body,
                localAnchorA=anchor,
                # center of tire
                localAnchorB=(0, 0),
                enableMotor=False,
                maxMotorTorque=1000,
                enableLimit=True,
                lowerAngle=0,
                upperAngle=0,
            )
            joints.append(j)

        self._rays = self.build_rays(rays)

        self.next_checkpoint = 0
        self.laps = 0

    def build_rays(self, rays):
        ret = []
        for i in rays:
            input = b2.rayCastInput(p1=self.body.transform * i,
                                    p2=self.body.worldCenter,
                                    maxFraction=1)
            output = b2.rayCastOutput()
            self.body.fixtures[0].RayCast(output, input, 0)
            ret.append((self.body.localCenter + i * (1. - output.fraction), i))
        return ret

    @property
    def rays(self):
        return [
            (self.body.transform * a,
             self.body.transform * b)
            for a, b in self._rays
        ]

    def update(self, keys, hz):

        for edge in self.body.contacts:
            if edge.contact.touching:
                self.body.linearVelocity *= (1. - edge.contact.friction)

        for tire in self.tires:
            tire.update_friction()

        for tire in self.tires:
            tire.update_drive(keys)

        # control steering
        lock_angle = self.max_tires_angle
        # from lock to lock in 0.5 sec
        turn_speed_per_sec = math.radians(160.)
        turn_per_timestep = turn_speed_per_sec / hz
        desired_angle = 0.0

        if 'left' in keys:
            desired_angle = lock_angle
        elif 'right' in keys:
            desired_angle = -lock_angle

        front_left_joint, front_right_joint = self.joints[2:4]
        angle_now = front_left_joint.angle
        angle_to_turn = desired_angle - angle_now

        # TODO fix b2Clamp for non-b2Vec2 types
        if angle_to_turn < -turn_per_timestep:
            angle_to_turn = -turn_per_timestep
        elif angle_to_turn > turn_per_timestep:
            angle_to_turn = turn_per_timestep

        new_angle = angle_now + angle_to_turn
        # Rotate the tires by locking the limits:
        front_left_joint.SetLimits(new_angle, new_angle)
        front_right_joint.SetLimits(new_angle, new_angle)

    def destroy(self):
        for i in self.tires:
            self.world.DestroyBody(i.body)
        self.world.DestroyBody(self.body)
