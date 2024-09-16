# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
from isaacgym import gymapi, gymtorch, gymutil
import torch
import math

from isaacgymenvs.tasks.base.vec_task import Env
from isaacgymenvs.tasks.cont_mc.env_curve import generate_curve

class ContinuousMountainCar(Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, **kwargs):
        self.cfg = cfg
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 2
        self.cfg["env"]["numActions"] = 1

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        self.curve, self.deriv, max_x, max_height_x = generate_curve(4, seed=15)
        
        self.min_position = 0
        self.max_position = max_x
        self.max_speed = 200
        self.goal_position = torch.tensor(max_height_x)
        self.goal_velocity = 2
        self.power = 0.015
        self.dt = 0.5
        self.friction = 0.005
        
        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        
        self.render_mode = "human"
        self.allocate_buffers()

    def allocate_buffers(self):
        self.observation = torch.empty((self.num_envs, self.num_obs), device=self.device)
        self.max_episode_height = torch.zeros(self.num_envs, 1, device=self.device)
        
        self.episode_discounted_reward = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_episode_discounted_reward = torch.zeros(self.num_envs, 1, device=self.device)
        self.episode_progress = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_action = None
        
    def reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.observation[env_ids, 0:1] = torch.rand((len(env_ids), 1), device=self.device) * (self.max_position - self.min_position) + self.min_position
        self.observation[env_ids, 1:2] = torch.zeros((len(env_ids), 1), device=self.device)
        self.episode_progress[env_ids] = 0
        
        self.prev_episode_discounted_reward[env_ids] = self.episode_discounted_reward[env_ids]
        self.episode_discounted_reward[env_ids] = 0
        
        self.max_episode_height[env_ids] = self.curve(self.observation[env_ids, 0:1])
        
    def step(self, actions):
        actions = actions.to(self.device)
        if not self.headless:
            self.render(self.render_mode)
        self.last_action = actions
        
        position = self.observation[:, 0:1]
        old_position = position.clone()
        velocity = self.observation[:, 1:2]
        force = torch.clamp(actions, -1.0, 1.0)
        
        slope = self.deriv(position)
        velocity += (force * self.power - 0.1 * slope / torch.sqrt(1 + slope ** 2) - self.friction * (torch.sign(velocity)).float() / torch.sqrt(1 + slope ** 2)) * self.dt
        velocity = torch.clamp(velocity, -self.max_speed, self.max_speed)
        position += velocity * self.dt / torch.sqrt(1 + slope ** 2)
        eps = 0.01
        position = torch.clamp(position, self.min_position + eps, self.max_position - eps)
        velocity = torch.where((position == self.min_position + eps) & (velocity < 0), -velocity, velocity)
        velocity = torch.where((position == self.max_position - eps) & (velocity > 0), -velocity, velocity)
        
        
        tmp_min, tmp_max = torch.minimum(old_position, position), torch.maximum(old_position, position)
        terminated = (tmp_min < self.goal_position) & (self.goal_position < tmp_max) & (torch.abs(velocity) < self.goal_velocity)
        
        reward = torch.zeros(self.num_envs, 1, device=self.device)
        curr_height = self.curve(position)
        # reward += torch.sign(force*velocity) * 0.1
        #reward += torch.clamp(curr_height - self.max_episode_height, min=0.0)
        self.max_episode_height = torch.maximum(self.max_episode_height, curr_height)
        reward += torch.where(terminated, torch.ones_like(reward) * 100.0, torch.zeros_like(reward))
        reward -= 0.01 * torch.abs(actions)
        
        self.observation[:, 0:1] = position
        self.observation[:, 1:2] = velocity
        
        self.episode_discounted_reward += (0.997 ** self.episode_progress) * reward
        self.episode_progress += 1
        
        terminated = (self.episode_progress >= self.max_episode_length) | terminated
        
        reset_idxs = torch.nonzero(terminated.ravel(), as_tuple=False).ravel()
        self.reset(reset_idxs)        
        
        return { "obs" : self.observation.clone().to(self.rl_device) } ,  reward.squeeze().to(self.rl_device), terminated.squeeze().to(self.rl_device), { "discounted_reward" : self.prev_episode_discounted_reward.clone().to(self.rl_device) }      
        
    
    def reset(self, env_ids=None):
        self.reset_idx(env_ids)
        return { "obs" : self.observation.clone() }

    def render(self, mode="human"):
        if mode is None:
            assert False, (
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        num_envs = self.observation.shape[0]  # Assume self.states stores states for all environments
        grid_size = int(np.ceil(np.sqrt(num_envs)))
        window_width = self.screen_width * grid_size
        window_height = self.screen_height * grid_size

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((window_width, window_height))
            else:  # mode == "rgb_array":
                self.screen = pygame.Surface((window_width, window_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20
        clearance = 10

        for idx in range(num_envs):
            row = idx // grid_size
            col = idx % grid_size

            x_offset = col * self.screen_width
            y_offset = row * self.screen_height

            env_surf = pygame.Surface((self.screen_width, self.screen_height))
            env_surf.fill((255, 255, 255))
            
            # add border
            
            pygame.draw.rect(env_surf, (0, 0, 0), (0, 0, self.screen_width, self.screen_height), 1)

            pos = self.observation[idx, 0]

            xs = torch.linspace(self.min_position, self.max_position, 1000)
            ys = self.curve(xs)
            # print(xs, ys)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            pygame.draw.aalines(env_surf, points=xys, closed=False, color=(0, 0, 0))

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            coords = []
            for c in [(l, b), (l, t), (r, t), (r, b)]:
                c = pygame.math.Vector2(c).rotate_rad(math.atan(self.deriv(pos)))
                coords.append(
                    (
                        c[0] + (pos - self.min_position) * scale,
                        c[1] + clearance + self.curve(pos) * scale,
                    )
                )

            gfxdraw.aapolygon(env_surf, coords, (0, 0, 0))
            gfxdraw.filled_polygon(env_surf, coords, (0, 0, 0))

            for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
                c = pygame.math.Vector2(c).rotate_rad(math.atan(self.deriv(pos)))
                wheel = (
                    int(c[0] + (pos - self.min_position) * scale),
                    int(c[1] + clearance + self.curve(pos) * scale),
                )

                gfxdraw.aacircle(
                    env_surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
                )
                gfxdraw.filled_circle(
                    env_surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
                )

            flagx = int((self.goal_position - self.min_position) * scale)
            flagy1 = int(self.curve(self.goal_position) * scale)
            flagy2 = flagy1 + 50
            gfxdraw.vline(env_surf, flagx, flagy1, flagy2, (0, 0, 0))

            gfxdraw.aapolygon(
                env_surf,
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
                (204, 204, 0),
            )
            gfxdraw.filled_polygon(
                env_surf,
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
                (204, 204, 0),
            )

            if self.last_action is not None:
                action = self.last_action[idx,0]
                if action < 0:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
                pygame.draw.line(
                    env_surf,
                    color,
                    (self.screen_width // 2, self.screen_height // 2),
                    (
                        self.screen_width // 2 + int(action * 100),
                        self.screen_height // 2,
                    ),
                    5,
                )
                # print action on screen
                font = pygame.font.Font(None, 36)
                text = font.render(f"Action: {action:.2f}", True, (255,0,0) if action < 0 else (0,255,0))
                text = pygame.transform.flip(text, False, True)
                env_surf.blit(text, (10, 10))

            env_surf = pygame.transform.flip(env_surf, False, True)
            self.screen.blit(env_surf, (x_offset, y_offset))

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

        

#####################################################################
###=========================jit functions=========================###
#####################################################################

if __name__ == '__main__':
    cfg = {
        "env" : {
            "numEnvs" : 4
        },
        "sim" : {
            "use_gpu_pipeline" : False
        }
        
    }
    env = ContinuousMountainCar(cfg, rl_device="cpu", sim_device="cpu", graphics_device_id=0, headless=False)
    
    s = env.reset()
    for _ in range(2000):
        env.render()
        a = torch.where(s["obs"][:, 1:2] > 0, torch.ones_like(s["obs"][:, 0:1]), -torch.ones_like(s["obs"][:, 0:1]))
        a = torch.where(s["obs"][:, 1:2] == 0, torch.rand_like(s["obs"][:, 1:2]), a)
        a = torch.rand_like(s["obs"][:, 1:2])*2 - 1
        #a = torch.zeros_like(s["obs"][:, 1:2])
        s, _, _, _ = env.step(a)
        import time; time.sleep(0.1)
        #print(s)
