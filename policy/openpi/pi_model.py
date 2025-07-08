#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import numpy as np
try:
    import jax
    from openpi.models import model as _model
    from openpi.policies import aloha_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader
except:
    print("no openpi package loading")

import cv2
from PIL import Image

try:
    from openpi.models import model as _model
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader
except:
    print("no openpi package loading")

class PI0:
    def __init__(self, task_name,train_config_name,model_name,checkpoint_id):
        self.train_config_name = train_config_name
        self.task_name = task_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        # import ipdb; ipdb.set_trace()
        self.policy = _policy_config.create_trained_policy(config, f"/gemini/space/users/zhangyang/openpi_checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}")
        print("loading model success!")
        self.img_size = (224,224)
        self.observation_window = None
        self.random_set_language()

    # set img_size
    def set_img_size(self,img_size):
        self.img_size = img_size
    
    # set language randomly
    def random_set_language(self):
        json_Path =f"data/instructions/{self.task_name}.json"
        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict['instructions']
        instruction = np.random.choice(instructions)
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")
    
    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = img_arr[0], img_arr[1], img_arr[2], state
        img_front = np.transpose(img_front, (2, 0, 1))  # 如果按照apply_pi里面传入的input_rgb_arr，则图像是bgr的，以下right和left同理
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert (self.observation_window is not None), "update observation_window first!"
        # self.policy.__class__ 是 policy/openpi/src/openpi/policies/policy.py 的 Policy类
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.datasets.transforms import AbsoluteActionTransform
from torchvision.transforms import v2

import torch
class Lerobot_PI0:
    def __init__(self, task_name, pretrained_checkpoint_path):
        self.task_name = task_name
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        print(f"loading pretrained checkpoint from {pretrained_checkpoint_path}...")
        self.policy = PI0Policy.from_pretrained(pretrained_checkpoint_path, local_files_only=True).eval()
        print("loading model success!")
        self.img_size = (224,224)
        self.observation_window = None

        self.action_mask = torch.ones(14, dtype=torch.bool)
        self.action_mask[13] = False
        self.action_mask[6] = False
        self.delta2abs = AbsoluteActionTransform(
            action_mask=self.action_mask,
        )

        self.random_set_language()

    # set img_size
    def set_img_size(self,img_size):
        self.img_size = img_size
    
    # set language randomly
    def random_set_language(self):
        json_Path =f"data/instructions/{self.task_name}.json"
        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict['instructions']
        instruction = np.random.choice(instructions)
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")
    
    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = img_arr[0], img_arr[1], img_arr[2], state
        img_front = np.transpose(img_front, (2, 0, 1))  # 如果按照apply_pi里面传入的input_rgb_arr，则图像是bgr的，以下right和left同理
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        img_front = torch.from_numpy(img_front.astype(np.float32) / 255.)
        img_left = torch.from_numpy(img_left.astype(np.float32) / 255.)
        img_right = torch.from_numpy(img_right.astype(np.float32) / 255.)

        # import ipdb; ipdb.set_trace()

        # img_front = self.image_transform(img_front)
        # img_left = self.image_transform(img_left)
        # img_right = self.image_transform(img_right)

        self.observation_window = {
            "observation.state": torch.from_numpy(state).to(torch.float32).unsqueeze(0),
            "observation.images.cam_high": img_front.unsqueeze(0),
            "observation.images.cam_left_wrist": img_left.unsqueeze(0),
            "observation.images.cam_right_wrist": img_right.unsqueeze(0),
            "task": [self.instruction],
        }

    def get_action(self):
        assert (self.observation_window is not None), "update observation_window first!"

        inputs = {k: v.to(self.policy.config.device) if isinstance(v, torch.Tensor) else v for k, v in self.observation_window.items()}
        # self.policy.__class__ 是 policy/openpi/src/openpi/policies/policy.py 的 Policy类
        unnormalized_actions = self.policy.select_action(inputs)

        outputs = {
            'observation.state': inputs['observation.state'],
            'action': unnormalized_actions,
        }
        outputs = self.delta2abs(outputs)

        return outputs['action'][0].cpu().numpy()

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
