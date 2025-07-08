#!/bin/zsh

while true; do
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    if [ $? -eq 0 ]; then
        echo "git fetch origin 成功！"
        break
    else
        echo "git fetch origin 失败，5秒后重试..."
        sleep 1
    fi
done
