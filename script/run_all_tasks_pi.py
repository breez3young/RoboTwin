import subprocess
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

TASKS_LIST = [
    "block_hammer_beat",
    "block_handover",
    "blocks_stack_easy",
    "blocks_stack_hard",
    "bottle_adjust",
    "container_place",
    "diverse_bottles_pick",
    "dual_bottles_pick_easy",
    "dual_bottles_pick_hard",
    "dual_shoes_place",
    "empty_cup_place",
    "mug_hanging_easy",
    "mug_hanging_hard",
    "pick_apple_messy",
    "put_apple_cabinet",
    "shoe_place",
    "tool_adjust",
]

num_gpus = torch.cuda.device_count()
tasks_per_gpu = 2
max_workers = num_gpus * tasks_per_gpu

def run_task(task_name, gpu_id):
    cmd = [
        "python", "./script/eval_policy_pi.py",
        task_name, "D435", "lerobot_new_transforms", "pi0_new_transforms", "40000", "1",
        "--eval-video-log", "--use_lerobot_pi0"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return subprocess.run(cmd, env=env)

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for idx, task in enumerate(TASKS_LIST):
        gpu_id = (idx // tasks_per_gpu) % num_gpus
        futures.append(executor.submit(run_task, task, gpu_id))
    for future in as_completed(futures):
        print("任务完成", future.result())