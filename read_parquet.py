import pandas as pd
import os
import numpy as np
from PIL import Image

parquet_path = "/gemini/space/huggingface_cache/rhodes_lerobot/RoboTwin/all_tasks_50ep/data/chunk-000/episode_000300.parquet"

# 读取parquet文件
df = pd.read_parquet(parquet_path)

print(f"文件: {parquet_path}")
print(f"字段名: {list(df.columns)}")
print(f"数据条数: {len(df)}")
print(f"前5行数据:")
print(df.head())

# 可视化 observation.images.cam_high 列的前5个图片
img_key = 'observation.images.cam_high'
vis_dir = 'vis_imgs'
os.makedirs(vis_dir, exist_ok=True)

for i in range(5):
    img_data = df.iloc[i][img_key]['bytes']
    # 如果是numpy数组直接保存，如果是bytes则先解码
    if isinstance(img_data, bytes):
        # 假设是jpg/png编码的bytes
        from io import BytesIO
        img = Image.open(BytesIO(img_data))
    elif isinstance(img_data, np.ndarray):
        img = Image.fromarray(img_data)
    else:
        raise ValueError(f"未知图片数据类型: {type(img_data)}")

    img.save(os.path.join(vis_dir, f"cam_high_{i}.png"))
print(f"前5张图片已保存到 {vis_dir}/ 目录下") 