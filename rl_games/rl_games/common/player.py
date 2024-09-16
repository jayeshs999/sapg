from collections import defaultdict
import os
import pickle
import shutil
import threading
import time
import gym
import numpy as np
from rl_games.common.custom_utils import create_sinusoidal_encoding
import torch
import copy
from os.path import basename
from typing import Optional
from rl_games.common import vecenv
from rl_games.common import env_configurations
from rl_games.algos_torch import model_builder, torch_ext


class BasePlayer(object):

    def __init__(self, params):
        self.config = config = params['config']
        self.load_networks(params)
        self.env_name = self.config['env_name']
        self.player_config = self.config.get('player', {})
        self.env_config = self.config.get('env_config', {})
        self.env_config = self.player_config.get('env_config', self.env_config)
        self.env_info = self.config.get('env_info')
        self.clip_actions = config.get('clip_actions', True)
        self.seed = self.env_config.pop('seed', None)

        if self.env_info is None:
            use_vecenv = self.player_config.get('use_vecenv', False)
            if use_vecenv:
                print('[BasePlayer] Creating vecenv: ', self.env_name)
                self.env = vecenv.create_vec_env(
                    self.env_name, self.config['num_actors'], **self.env_config)
                self.env_info = self.env.get_env_info()
            else:
                print('[BasePlayer] Creating regular env: ', self.env_name)
                self.env = self.create_env()
                self.env_info = env_configurations.get_env_info(self.env)
        else:
            self.env = config.get('vec_env')

        self.num_agents = self.env_info.get('agents', 1)
        self.value_size = self.env_info.get('value_size', 1)
        self.action_space = self.env_info['action_space']

        self.observation_space = self.env_info['observation_space']
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.player_config = self.config.get('player', {})
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.has_central_value = self.config.get(
            'central_value_config') is not None
        self.device_name = self.config.get('device_name', 'cuda')
        self.render_env = self.player_config.get('render', False)
        self.games_num = self.player_config.get('games_num', 2000)

        if 'deterministic' in self.player_config:
            self.is_deterministic = self.player_config['deterministic']
        else:
            self.is_deterministic = self.player_config.get(
                'deterministic', True)

        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        self.max_steps = 5400 if self.player_config.get('save_obs', False) else 27000
        self.device = torch.device(self.device_name)

        self.evaluation = self.player_config.get("evaluation", False)
        self.update_checkpoint_freq = self.player_config.get("update_checkpoint_freq", 100)
        # if we run player as evaluation worker this will take care of loading new checkpoints
        self.dir_to_monitor = self.player_config.get("dir_to_monitor")
        # path to the newest checkpoint
        self.checkpoint_to_load: Optional[str] = None

        self.expl_type = config.get('expl_type', 'none')

        if self.expl_type != 'none':
            if self.expl_type.startswith('mixed_expl'):
                embedding_genvec = torch.linspace(50.0, 0.0, self.env.num_envs).to(self.device_name)
                if 'disjoint' in self.expl_type or 'learn_param' in self.expl_type:
                    self.intr_reward_coef_embd = embedding_genvec.reshape(-1,1)
                else:
                    self.intr_reward_coef_embd = create_sinusoidal_encoding(embedding_genvec, config.get('expl_reward_coef_embd_size', 32), n=100).to(self.device_name)
            elif self.expl_type.startswith('simple'):
                self.intr_reward_coef_embd = None
        else:
            self.intr_reward_coef_embd = None

        if self.evaluation and self.dir_to_monitor is not None:
            self.checkpoint_mutex = threading.Lock()
            self.eval_checkpoint_dir = os.path.join(self.dir_to_monitor, "eval_checkpoints")
            os.makedirs(self.eval_checkpoint_dir, exist_ok=True)

            patterns = ["*.pth"]
            from watchdog.observers import Observer
            from watchdog.events import PatternMatchingEventHandler
            self.file_events = PatternMatchingEventHandler(patterns)
            self.file_events.on_created = self.on_file_created
            self.file_events.on_modified = self.on_file_modified

            self.file_observer = Observer()
            self.file_observer.schedule(self.file_events, self.dir_to_monitor, recursive=False)
            self.file_observer.start()

    def wait_for_checkpoint(self):
        if self.dir_to_monitor is None:
            return

        attempt = 0
        while True:
            attempt += 1
            with self.checkpoint_mutex:
                if self.checkpoint_to_load is not None:
                    if attempt % 10 == 0:
                        print(f"Evaluation: waiting for new checkpoint in {self.dir_to_monitor}...")
                    break
            time.sleep(1.0)

        print(f"Checkpoint {self.checkpoint_to_load} is available!")

    def maybe_load_new_checkpoint(self):
        # lock mutex while loading new checkpoint
        with self.checkpoint_mutex:
            if self.checkpoint_to_load is not None:
                print(f"Evaluation: loading new checkpoint {self.checkpoint_to_load}...")
                # try if we can load anything from the pth file, this will quickly fail if the file is corrupted
                # without triggering the retry loop in "safe_filesystem_op()"
                load_error = False
                try:
                    torch.load(self.checkpoint_to_load)
                except Exception as e:
                    print(f"Evaluation: checkpoint file is likely corrupted {self.checkpoint_to_load}: {e}")
                    load_error = True

                if not load_error:
                    try:
                        self.restore(self.checkpoint_to_load)
                    except Exception as e:
                        print(f"Evaluation: failed to load new checkpoint {self.checkpoint_to_load}: {e}")

                # whether we succeeded or not, forget about this checkpoint
                self.checkpoint_to_load = None

    def process_new_eval_checkpoint(self, path):
        with self.checkpoint_mutex:
            # print(f"New checkpoint {path} available for evaluation")
            # copy file to eval_checkpoints dir using shutil
            # since we're running the evaluation worker in a separate process,
            # there is a chance that the file is changed/corrupted while we're copying it
            # not sure what we can do about this. In practice it never happened so far though
            try:
                eval_checkpoint_path = os.path.join(self.eval_checkpoint_dir, basename(path))
                shutil.copyfile(path, eval_checkpoint_path)
            except Exception as e:
                print(f"Failed to copy {path} to {eval_checkpoint_path}: {e}")
                return

            self.checkpoint_to_load = eval_checkpoint_path

    def on_file_created(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    def on_file_modified(self, event):
        self.process_new_eval_checkpoint(event.src_path)

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        if self.config.get('expl_type').startswith('mixed_expl') and 'disjoint' in self.config.get('expl_type'):
            params['model']['name'] = 'multi_' + params['model']['name']
        self.config['network'] = builder.load(params)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs) if self.intr_reward_coef_embd is None else torch.cat([self.obs_to_torch(obs), self.intr_reward_coef_embd], dim=1), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs) if self.intr_reward_coef_embd is None else torch.cat([self.obs_to_torch(obs), self.intr_reward_coef_embd], dim=1), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if 'obs' in obs:
                obs = obs['obs']
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert (obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs.to(self.device)

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs) if self.intr_reward_coef_embd is None else torch.cat([self.obs_to_torch(obs), self.intr_reward_coef_embd], dim=1)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(
                weights['running_mean_std'])

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_deterministic=False):
        raise NotImplementedError('step')

    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [s.to(self.device) for s in rnn_states]

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        rewards_tracker = torch.zeros(self.env.num_envs, dtype=torch.float32).cpu()
        success_tracker = torch.zeros(self.env.num_envs, dtype=torch.float32).cpu()
        episode_length_tracker = torch.zeros(self.env.num_envs, dtype=torch.float32).cpu()
        episode_tracker = torch.zeros(self.env.num_envs, dtype=torch.float32).cpu()

        rigid_body_state_tracker = defaultdict(list)
        
        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        obs_buffer = []
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            self.env.update_rigid_body_state_dict(rigid_body_state_tracker)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if n % 500 == 0:
                    print(f"Step {n} of {self.max_steps}")
                obs_buffer.append(obses)
                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)
                self.env.update_rigid_body_state_dict(rigid_body_state_tracker)
                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    #if games_played//8 != (games_played-done_count)//8:
                    #    self.update_checkpoint(games_played//8 * 4)
                        
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                          all_done_indices, :] * 0.0

                    rewards_tracker[done_indices] += cr[done_indices]
                    episode_length_tracker[done_indices] += steps[done_indices]
                    episode_tracker[done_indices] += 1
                    if 'successes' in info:
                        success_tracker[done_indices] += info['successes'].cpu()[done_indices]

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        print(f'reward: {(rewards_tracker/episode_tracker).numpy()}')
                        print(f'steps: {(episode_length_tracker/episode_tracker).numpy()}')
                        if 'successes' in info:
                            print(f'successes: {(success_tracker/episode_tracker).numpy()}')
                        if print_game_res:
                            print(f'w: {game_res}')
                        print(f'num_episodes: {episode_tracker.numpy()}')

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break
            obs_buffer.append(obses)
        
        obs_buffer = torch.stack(obs_buffer)
        if self.intr_reward_coef_embd is not None:
            assert len(obs_buffer.shape) == 3
            obs_buffer = obs_buffer[:,:,:-self.intr_reward_coef_embd.shape[1]]
        
        pickle.dump(rigid_body_state_tracker, open(f'{self.loaded_checkpoint}.rigid_body_state.pkl', 'wb'))
        
        if self.player_config.get('save_obs', False):
            with open(f'{self.loaded_checkpoint}.obs.pkl', 'wb') as f:
                pickle.dump(obs_buffer.detach().cpu().numpy(), f)

        print(f'reward: {(rewards_tracker/episode_tracker).numpy()}')
        print(f'steps: {(episode_length_tracker/episode_tracker).numpy()}')
        if 'successes' in info:
            print(f'successes: {(success_tracker/episode_tracker).numpy()}')
        print(f'num_episodes: {episode_tracker.numpy()}')

    
    def update_checkpoint(self, idx):
        chkp_list = os.path.dirname(self.loaded_checkpoint)
        def extract_episode_number(checkpoint_path):
            import re
            match = re.search(r'_ep_(\d+)_', checkpoint_path)
            if match:
                return match.group(1)
            return "100000"
        sorted_list = sorted([os.path.join(chkp_list, f) for f in os.listdir(chkp_list) if "_ep_" in f], key=lambda x: int(extract_episode_number(x)))
        self.restore(sorted_list[idx])
        print(f"Loaded checkpoint {sorted_list[idx]}")

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if 'obs' in obses:
                obses = obses['obs']
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if 'observation' in obses:
                first_key = 'observation'
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size
