#!/usr/bin/env python

import json
import logging

from tensorforce.agents import Agent

from main import App


config = json.load(open('ppo.json'))
max_episodes = config.pop('max_episodes', None)
max_timesteps = config.pop('max_timesteps', None)
max_episode_timesteps = config.pop('max_episode_timesteps')
network_spec = config.pop('network')

agent = Agent.from_spec(
    spec=config,
    kwargs=dict(
        states=dict(type='float', shape=(4,)),
        actions=dict(type='int', num_actions=4),
        network=network_spec
    )
)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

app = App()
app.init_screen()
app.render()

episode = 0

while True:
    episode += 1
    render = episode % 20 == 0
    app.init_car()
    no_reward_frames = 0
    terminal = False
    frames = 0
    while not terminal:
        frames += 1
        state = app.get_state()
        action = agent.act(state)
        reward = app.execute(action)
        if reward:
            no_reward_frames = 0
            logging.info('Episode#%d: checkpoint #%d on frame %d', episode,
                         app.car.next_checkpoint - 1, frames)
            if app.car.laps > 0:
                terminal = True
                logging.info('Episode#%d: finish on frame %d', episode, frames)
        else:
            no_reward_frames += 1
        if render:
            if frames % 5 == 0:
                logging.info('state: %r', state)
            app.clock.tick(app.target_fps)
            app.render()
        if no_reward_frames > 500 or app.car.body.contacts:
            terminal = True
            reward = -1
        agent.observe(reward=reward, terminal=terminal)
