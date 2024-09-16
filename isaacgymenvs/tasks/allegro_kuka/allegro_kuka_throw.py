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

from typing import List

import torch
from isaacgym import gymapi
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import to_torch, torch_rand_float
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import tolerance_successes_objective


class AllegroKukaThrow(AllegroKukaBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.bucket_asset = self.bucket_pose = None
        self.bucket_object_indices = []

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _object_keypoint_offsets(self):
        """Throw task uses only a single object keypoint since we do not care about object orientation."""
        return [[0, 0, 0]]

    def _load_additional_assets(self, object_asset_root, arm_pose):
        """
        returns: tuple (num_rigid_bodies, num_shapes)
        """
        bucket_asset_options = gymapi.AssetOptions()
        bucket_asset_options.disable_gravity = False
        bucket_asset_options.fix_base_link = True
        bucket_asset_options.collapse_fixed_joints = True
        bucket_asset_options.vhacd_enabled = True
        bucket_asset_options.vhacd_params = gymapi.VhacdParams()
        bucket_asset_options.vhacd_params.resolution = 500000
        bucket_asset_options.vhacd_params.max_num_vertices_per_ch = 32
        bucket_asset_options.vhacd_params.min_volume_per_ch = 0.001
        self.bucket_asset = self.gym.load_asset(
            self.sim, object_asset_root, self.asset_files_dict["bucket"], bucket_asset_options
        )

        self.bucket_pose = gymapi.Transform()
        self.bucket_pose.p = gymapi.Vec3()
        self.bucket_pose.p.x = arm_pose.p.x - 0.6
        self.bucket_pose.p.y = arm_pose.p.y - 1
        self.bucket_pose.p.z = arm_pose.p.z + 0.45

        bucket_rb_count = self.gym.get_asset_rigid_body_count(self.bucket_asset)
        bucket_shapes_count = self.gym.get_asset_rigid_shape_count(self.bucket_asset)
        print(f"Bucket rb {bucket_rb_count}, shapes {bucket_shapes_count}")

        return bucket_rb_count, bucket_shapes_count

    def _create_additional_objects(self, env_ptr, env_idx, object_asset_idx):
        bucket_handle = self.gym.create_actor(
            env_ptr, self.bucket_asset, self.bucket_pose, "bucket_object", env_idx, 0, 0
        )
        bucket_object_idx = self.gym.get_actor_index(env_ptr, bucket_handle, gymapi.DOMAIN_SIM)
        for name in self.gym.get_actor_rigid_body_names(env_ptr, bucket_handle):
            self.rigid_body_name_to_idx['bucket/' + name] = self.gym.find_actor_rigid_body_index(env_ptr, bucket_handle, name, gymapi.DOMAIN_ENV)
        self.bucket_object_indices.append(bucket_object_idx)

    def _after_envs_created(self):
        self.bucket_object_indices = to_torch(self.bucket_object_indices, dtype=torch.long, device=self.device)

    def _reset_target(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True) -> None:
        # whether we place the bucket to the left or to the right of the table
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            left_right_random = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            x_pos = torch.where(
                left_right_random > 0, 0.5 * torch.ones_like(left_right_random), -0.5 * torch.ones_like(left_right_random)
            )
            x_pos += torch.sign(left_right_random) * torch_rand_float(0, 0.4, (len(env_ids), 1), device=self.device)
            # y_pos = torch_rand_float(-0.6, 0.4, (len(env_ids), 1), device=self.device)
            y_pos = torch_rand_float(-1.0, 0.7, (len(env_ids), 1), device=self.device)
            z_pos = torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device)
            self.root_state_tensor[self.bucket_object_indices[env_ids], 0:1] = x_pos
            self.root_state_tensor[self.bucket_object_indices[env_ids], 1:2] = y_pos
            self.root_state_tensor[self.bucket_object_indices[env_ids], 2:3] = z_pos

            self.goal_states[env_ids, 0:1] = x_pos
            self.goal_states[env_ids, 1:2] = y_pos
            self.goal_states[env_ids, 2:3] = z_pos + 0.05

        if len(env_ids) > 0 and reset_buf_idxs is not None and tensor_reset:
            rs_ofs = self.root_state_resets.shape[1]
            self.root_state_tensor[self.bucket_object_indices[env_ids], :] = self.root_state_resets[
                reset_buf_idxs[env_ids].cpu(), self.bucket_object_indices[env_ids].cpu() % rs_ofs, :
            ].to(self.device)
            self.goal_states[env_ids, 0:3] = self.root_state_tensor[self.bucket_object_indices[env_ids], 0:3]
        # we also reset the object to its initial position
        self.reset_object_pose(env_ids, reset_buf_idxs, tensor_reset)

        self.deferred_set_actor_root_state_tensor_indexed([self.bucket_object_indices[env_ids]])

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        return [self.bucket_object_indices[env_ids]]

    def _true_objective(self) -> Tensor:
        true_objective = tolerance_successes_objective(
            self.success_tolerance, self.initial_tolerance, self.target_tolerance, self.successes
        )
        return true_objective

    def update_rigid_body_state_dict(self, state_ddict, env_idx=0):
        """
        state_ddict: defaultdict of the form Dict[str, List[Tensor]]
        """
        isaacgym_to_blender_name = {
            'allegro/iiwa7_link_0': 'link_0',
            'allegro/iiwa7_link_1': 'link_1',
            'allegro/iiwa7_link_2': 'link_2',
            'allegro/iiwa7_link_3': 'link_3',
            'allegro/iiwa7_link_4': 'link_4',
            'allegro/iiwa7_link_5': 'link_5',
            'allegro/iiwa7_link_6': 'link_6',
            'allegro/iiwa7_link_7': 'link_7',
            #'allegro/iiwa7_link_ee': 'link_ee',
            'allegro/allegro_mount': 'allegro_mount',
            'allegro/palm_link': 'base_link',
            'allegro/index_link_0': 'primary_base',
            'allegro/index_link_1': 'primary_proximal',
            'allegro/index_link_2': 'primary_medial',
            'allegro/index_link_3': 'touch_sensor_base',
            'allegro/middle_link_0': 'primary_base.001',
            'allegro/middle_link_1': 'primary_proximal.001',
            'allegro/middle_link_2': 'primary_medial.001',
            'allegro/middle_link_3': 'touch_sensor_base.001',
            'allegro/ring_link_0': 'primary_base.002',
            'allegro/ring_link_1': 'primary_proximal.002',
            'allegro/ring_link_2': 'primary_medial.002',
            'allegro/ring_link_3': 'touch_sensor_base.002',
            'allegro/thumb_link_0': 'thumb_base',
            'allegro/thumb_link_1': 'thumb_proximal',
            'allegro/thumb_link_2': 'thumb_medial',
            'allegro/thumb_link_3': 'touch_sensor_thumb_base',
            'object/object': 'cube_multicolor',
            'table/box': 'cube', # 0.475 0.4 0.3
            'bucket/bucket': 'bucket'
        }
        
        for key, value in isaacgym_to_blender_name.items():
            rigid_body_idx = self.rigid_body_name_to_idx[key]
            pose = self.rigid_body_states[env_idx, rigid_body_idx,0:7].cpu().numpy()
            import transforms3d
            
            pos = pose[0:3]
            quat = pose[3:7]
            rot = transforms3d.euler.quat2euler([quat[3], quat[0], quat[1], quat[2]])
            
            state_ddict[value].append((pos, rot))