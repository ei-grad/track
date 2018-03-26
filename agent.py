#!/usr/bin/env python

from os import getenv
import json
import logging
import random
import math

import numpy as np

from tensorforce.agents import Agent

from main import App


app = App()
app.init_screen()
app.render()

config = json.load(open(getenv('CONFIG', 'ppo.json')))
max_episodes = config.pop('max_episodes', None)
max_timesteps = config.pop('max_timesteps', None)
max_episode_timesteps = config.pop('max_episode_timesteps')
network_spec = config.pop('network')

agent = Agent.from_spec(
    spec=config,
    kwargs=dict(
        states=dict(type='float', shape=(len(app.get_state()),)),
        actions={
            'accel': dict(type='int', num_actions=3),
            'turn': dict(type='int', num_actions=3),
        },
        network=network_spec
    )
)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

episode = 0

MAX_FRAMES_WITHOUT_REWARD = 200.

while True:
    episode += 1
    # XXX: make random switch
    app.checkpoints = app.checkpoints[::-1]
    n = int(random.random() * len(app.checkpoints))
    app.checkpoints = app.checkpoints[n:] + app.checkpoints[:n]
    p = app.checkpoints[1] - app.checkpoints[0]
    angle = math.atan2(p.y, p.x) - math.pi / 2.
    angle += np.random.normal()
    app.init_car(angle)
    frames_without_reward = 0
    terminal = False
    frames = 0
    episode_total_reward = 0.
    while not terminal:
        frames += 1
        state = app.get_state()
        actions = agent.act(state)
        reward = app.execute(actions)
        if reward:
            reward = 1. - frames_without_reward / MAX_FRAMES_WITHOUT_REWARD
            frames_without_reward = 0
            if app.car.laps == 3:
                terminal = True
        else:
            frames_without_reward += 1
        app.clock.tick(0)
        if episode % 20 == 0 or frames % 20 == 0:
            app.render()
        # if frames_without_reward > MAX_FRAMES_WITHOUT_REWARD or app.car.body.contacts:
        has_contact = any(i.contact.touching for i in app.car.body.contacts)
        if frames_without_reward > MAX_FRAMES_WITHOUT_REWARD or has_contact:
            terminal = True
            reward = -1.
        episode_total_reward += reward
        agent.observe(reward=reward, terminal=terminal)
        if terminal:
            logging.info(
                'Episode#%d: frames=%04d checkpoint=%d lap=%d reward=%.2f',
                episode, frames,
                app.car.next_checkpoint, app.car.laps,
                episode_total_reward,
            )
