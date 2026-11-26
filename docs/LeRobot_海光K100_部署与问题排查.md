# LeRobot 在海光 K100 / OpenBayes 平台上的部署、问题排查与最终运行指南

> 适用环境：OpenBayes 海光平台、8 × K100_AI（单卡 64 GiB）、Ubuntu 22.04、DTK 26.04、Python 3.10.13、海光定制 PyTorch 2.7.1、LeRobot v0.4.4。  
> 本文基于实际部署过程整理，重点记录：**遇到的问题、报错现象、问题原因、解决指令、最终可复现运行命令**。  
> 目标：在不破坏海光 PyTorch / DTK 环境的前提下，使 LeRobot 在 8 张 K100 上完成 ACT 多卡训练。

---

## 1. 最终部署架构

最终环境采用如下结构：

```text
OpenBayes 基础镜像
│
├── DTK 26.04
├── 海光 PyTorch 2.7.1
│   ├── torch
│   └── torchvision 0.22.0
│
└── /openbayes/home/lerobot-dcu/
    ├── .venv/                      # LeRobot 独立 Python 环境
    ├── lerobot/                    # LeRobot v0.4.4 源码
    ├── outputs/                    # 训练输出
    └── logs/                       # 日志

离线数据
│
├── /output/datasets/aloha_sim_insertion_human/
│   ├── meta/
│   │   └── info.json
│   ├── data/
│   └── videos/
│
└── /output/torch_cache/hub/checkpoints/
    └── resnet18-f37072fd.pth
```

设计原则：

1. **海光 PyTorch 不重新安装**，直接复用基础镜像中的 `torch 2.7.1`。
2. LeRobot 及其纯 Python 依赖放入独立 `.venv`。
3. `.venv` 的 Python 包优先于系统 Python 包。
4. 海光 `torch / torchvision` 仍从 `/usr/local/lib/python3.10/dist-packages` 加载。
5. 多卡训练不用系统 `accelerate`，统一使用：

```bash
/output/lerobot-dcu/.venv/bin/python -m torch.distributed.run
```

6. OpenBayes 正式环境无法访问 Hugging Face，因此数据集与 ResNet18 权重提前离线准备。

---

# 2. 为什么使用 LeRobot v0.4.4

当前海光镜像使用：

```text
Python 3.10.13
PyTorch 2.7.1
```

因此部署时没有直接使用最新 LeRobot，而是固定：

```text
LeRobot v0.4.4
```

主要原因是：

- v0.4.4 支持 Python 3.10。
- v0.4.4 的 PyTorch 版本范围能够覆盖 PyTorch 2.7.1。
- 直接使用较新的 LeRobot / `uv sync` 容易解析官方 NVIDIA CUDA PyTorch。
- 海光环境最重要的是**保住现有 DTK 定制 PyTorch**。

---

# 3. 初始环境检查

首先检查系统中的海光 PyTorch。

```bash
python -V

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch path:", torch.__file__)
print("available:", torch.cuda.is_available())
print("devices:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY

hy-smi
```

部署过程中实际验证结果为：

```text
Python 3.10.13
torch 2.7.1
torch.cuda.is_available() = True
torch.cuda.device_count() = 8
device0 = K100_AI
```

因此可以确认：

```text
PyTorch
   ↓
DTK
   ↓
K100_AI
```

基础链路已经可用。

---

# 4. 创建 LeRobot 工作目录

```bash
cd /output

mkdir -p lerobot-dcu
cd lerobot-dcu
```

---

# 5. 使用 uv 创建虚拟环境

由于后续 Python 项目统一使用 `uv` 管理，因此虚拟环境仍由 `uv` 创建。

关键点是需要让虚拟环境能够看到系统中的海光 PyTorch：

```bash
uv venv \
    --python "$(which python)" \
    --system-site-packages \
    .venv
```

激活：

```bash
source /output/lerobot-dcu/.venv/bin/activate
```

检查：

```bash
which python
python -V
```

应为：

```text
/output/lerobot-dcu/.venv/bin/python
Python 3.10.13
```

---

# 6. 获取 LeRobot v0.4.4

```bash
cd /output/lerobot-dcu

git clone \
    --branch v0.4.4 \
    --depth 1 \
    https://github.com/huggingface/lerobot.git

cd lerobot
```

检查：

```bash
git describe --tags
```

应看到：

```text
v0.4.4
```

---

# 7. 问题一：uv 准备覆盖海光 PyTorch

## 7.1 现象

执行：

```bash
uv pip install -e . --dry-run
```

发现解析结果包含：

```text
torch==2.10.0
torchvision==0.25.0
triton==3.6.0

nvidia-cublas-cu12
nvidia-cudnn-cu12
nvidia-nccl-cu12
nvidia-cufft-cu12
...
```

## 7.2 原因

虽然 `.venv` 使用了：

```text
--system-site-packages
```

Python 本身可以看到系统海光 PyTorch，但是 `uv` 的依赖解析器仍可能重新解析 `torch`，进而选择官方 PyPI / NVIDIA CUDA 版本。

如果直接执行：

```bash
uv pip install -e .
```

则可能将：

```text
海光 PyTorch 2.7.1
```

替换成：

```text
官方 PyTorch + NVIDIA CUDA 12.x
```

这会直接破坏 K100 环境。

## 7.3 处理方式

**禁止执行会重新解析 Torch 的安装。**

先查看 dry-run：

```bash
uv pip install -e . --dry-run
```

只要出现：

```text
torch-2.x
nvidia-*
triton
```

就不要继续。

LeRobot 的安装过程改为：

```bash
python -m pip install -e .
```

同时后续对关键包尽量使用：

```bash
--no-deps
```

避免重新解析 PyTorch 依赖。

---

# 8. 问题二：系统包与 LeRobot 虚拟环境发生大量依赖冲突

## 8.1 现象

安装过程中出现类似：

```text
cupy requires numpy<1.29
megatron-core requires numpy<2.0
vllm requires numpy==1.25
transformers 5.3.0 requires newer huggingface-hub
...
```

## 8.2 原因

基础镜像本身带有：

```text
vLLM
Megatron
CuPy
Transformers
OpenBayes CLI
Litellm
Langfuse
...
```

而 `.venv` 又使用：

```text
--system-site-packages
```

因此虚拟环境能“看见”系统的整套 Python 软件栈。

这些软件并不是 LeRobot 训练必须依赖，因此 pip 会报告大量“环境级”冲突。

## 8.3 原则

不要为了 LeRobot 去修复整个 OpenBayes 基础镜像。

实际目标只有：

```text
LeRobot 自己的 Python 依赖 → .venv
海光 torch / torchvision   → 系统环境
```

而不是让 `.venv` 同时满足：

```text
vLLM + Megatron + Litellm + LeRobot + OpenBayes CLI
```

所有依赖。

---

# 9. 问题三：Hugging Face Hub 安装成功但仍加载系统版本

## 9.1 现象

安装：

```bash
python -m pip install --upgrade --force-reinstall \
    "huggingface-hub==0.35.3"
```

虽然 pip 显示：

```text
Successfully installed huggingface-hub-0.35.3
```

但执行：

```bash
python - <<'PY'
import huggingface_hub
print(huggingface_hub.__version__)
print(huggingface_hub.__file__)
PY
```

却得到：

```text
1.5.0
/usr/local/lib/python3.10/dist-packages/huggingface_hub/__init__.py
```

## 9.2 原因

OpenBayes 设置了：

```bash
PYTHONPATH=/usr/local/:/usr/local/:/usr/local/lib/python3.10/dist-packages:
```

实际 `sys.path` 顺序中：

```text
/usr/local/lib/python3.10/dist-packages
```

位于：

```text
/output/lerobot-dcu/.venv/lib/python3.10/site-packages
```

之前。

因此系统包会覆盖虚拟环境中的包。

## 9.3 错误尝试：直接 unset PYTHONPATH

曾执行：

```bash
unset PYTHONPATH
```

结果：

```text
ModuleNotFoundError: No module named 'torch'
```

## 9.4 为什么不能直接 unset

海光 PyTorch 本身位于：

```text
/usr/local/lib/python3.10/dist-packages/torch
```

OpenBayes 正是通过 `PYTHONPATH` 将该目录暴露给 Python。

所以：

```bash
unset PYTHONPATH
```

虽然解决了系统 Python 包抢优先级的问题，却同时让海光 Torch 消失。

## 9.5 最终解决方案：重新排列 PYTHONPATH

使用：

```bash
export PYTHONPATH=/output/lerobot-dcu/lerobot/src:/output/lerobot-dcu/.venv/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages
```

顺序变成：

```text
1. LeRobot source
2. LeRobot .venv
3. OpenBayes 系统 Python 包
```

这样：

```text
huggingface_hub / transformers / datasets 等
    ↓
优先从 .venv 获取

torch / torchvision
    ↓
若 .venv 没有，则从系统海光环境获取
```

---

# 10. 问题四：NumPy 2.x 与海光 PyTorch ABI 不兼容

## 10.1 现象

导入 Torch 时出现：

```text
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

随后还有：

```text
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

## 10.2 原因

海光 PyTorch 2.7.1 是在 NumPy 1.x ABI 下构建的。

但 LeRobot 环境中安装了：

```text
numpy 2.2.6
```

调整 `PYTHONPATH` 后，`.venv` 的 NumPy 2.2.6 优先加载，因此：

```text
海光 torch 2.7.1
        +
NumPy 2.2.6
```

产生 ABI 冲突。

## 10.3 解决

将 NumPy 固定回：

```text
1.25.0
```

执行：

```bash
python -m pip uninstall -y numpy

python -m pip install \
    --no-deps \
    "numpy==1.25.0"
```

检查：

```bash
python - <<'PY'
import numpy
import torch

print("numpy:", numpy.__version__, numpy.__file__)
print("torch:", torch.__version__, torch.__file__)
print("DCU:", torch.cuda.is_available())
PY
```

预期：

```text
numpy 1.25.0
torch 2.7.1
DCU True
```

进一步测试：

```bash
python - <<'PY'
import numpy as np
import torch

a = np.array([1, 2, 3], dtype=np.float32)

b = torch.from_numpy(a)
print("numpy -> torch:", b)

c = b.numpy()
print("torch -> numpy:", c)

d = torch.randn(1024, 1024, device="cuda")
print("device:", d.device)
print("mean:", d.mean().item())

print("NUMPY / TORCH / DCU SUCCESS")
PY
```

---

# 11. 问题五：rerun-sdk 要求 NumPy 2.x

## 11.1 现象

pip 报告：

```text
rerun-sdk 0.26.2 requires numpy>=2
```

但海光 PyTorch 又需要 NumPy 1.x ABI。

## 11.2 原因

这是两个不同组件的依赖要求发生冲突：

```text
海光 PyTorch 2.7.1 → NumPy 1.x
rerun-sdk 0.26.2  → NumPy >= 2
```

## 11.3 解决原则

当前任务是：

```text
K100 上训练 LeRobot
```

而不是运行 Rerun 可视化。

因此训练环境优先保证：

```text
PyTorch / NumPy / K100
```

稳定。

可将 `rerun-sdk` 从训练环境移除：

```bash
python -m pip uninstall -y rerun-sdk
```

如源码 `pyproject.toml` 中仍包含该依赖，也可在海光适配分支中去掉：

```bash
sed -i '/"rerun-sdk>=0.24.0,<0.27.0",/d' pyproject.toml
```

如以后必须使用 Rerun，建议另建独立环境，不要和当前 K100 训练环境混用。

---

# 12. 问题六：TorchCodec 在海光平台不可用

## 12.1 现象

运行时出现：

```text
'torchcodec' is not available in your platform,
falling back to 'pyav' as a default decoder
```

## 12.2 原因

TorchCodec 涉及 PyTorch 二进制与后端兼容性，在海光定制 PyTorch 环境中不保证可用。

## 12.3 处理

不使用 TorchCodec，直接使用 PyAV。

训练命令中明确：

```bash
--dataset.video_backend=pyav
```

该警告不是失败，可以忽略。

---

# 13. 问题七：system Transformers 5.3.0 与 Hugging Face Hub 0.35.3 冲突

## 13.1 现象

运行训练时出现：

```text
ImportError:
cannot import name 'is_offline_mode'
from 'huggingface_hub'
```

调用链中显示：

```text
/usr/local/lib/python3.10/dist-packages/transformers
```

即系统 Transformers 5.3.0 被加载。

## 13.2 原因

LeRobot v0.4.4 的 import 链会导入：

```text
lerobot.processor
  ↓
tokenizer_processor
  ↓
transformers
```

即使当前训练的是 ACT，启动过程中仍会 import Transformers。

系统中的：

```text
transformers 5.3.0
```

和 `.venv` 中：

```text
huggingface_hub 0.35.3
```

接口不兼容。

## 13.3 解决

在 LeRobot `.venv` 中固定：

```text
transformers 4.57.1
tokenizers 0.22.1
huggingface-hub 0.35.3
```

执行：

```bash
export PYTHONPATH=/output/lerobot-dcu/lerobot/src:/output/lerobot-dcu/.venv/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages

/output/lerobot-dcu/.venv/bin/python -m pip install \
    --no-deps \
    --upgrade \
    --force-reinstall \
    "huggingface-hub==0.35.3" \
    "transformers==4.57.1" \
    "tokenizers==0.22.1"
```

检查：

```bash
/output/lerobot-dcu/.venv/bin/python - <<'PY'
import huggingface_hub
import transformers
import tokenizers

print("hub:", huggingface_hub.__version__, huggingface_hub.__file__)
print("transformers:", transformers.__version__, transformers.__file__)
print("tokenizers:", tokenizers.__version__, tokenizers.__file__)

from transformers import AutoProcessor, AutoTokenizer

print("TRANSFORMERS IMPORT SUCCESS")
PY
```

应确保 Transformers 路径为：

```text
/output/lerobot-dcu/.venv/lib/python3.10/site-packages/...
```

而不是：

```text
/usr/local/lib/python3.10/dist-packages/...
```

---

# 14. 问题八：调用系统 accelerate 导致依赖再次串环境

## 14.1 现象

直接运行：

```bash
accelerate launch ...
```

报错调用：

```text
/usr/local/bin/accelerate
```

随后导入：

```text
/usr/local/lib/python3.10/dist-packages/transformers
```

并再次发生 Transformers / Hugging Face Hub 冲突。

## 14.2 原因

Shell 中虽然已经显示：

```text
(.venv)
```

但裸命令：

```bash
accelerate
```

实际仍可能指向系统 CLI：

```text
/usr/local/bin/accelerate
```

因此运行的是系统软件栈，而不是 LeRobot `.venv`。

## 14.3 解决

不再使用系统 Accelerate。

多卡统一使用：

```bash
/output/lerobot-dcu/.venv/bin/python \
    -m torch.distributed.run
```

这样启动器本身就是 `.venv` Python。

---

# 15. 问题九：torchrun 启动的 worker 使用了错误 Python

## 15.1 现象

使用：

```bash
torchrun ...
```

后出现：

```text
binary: /usr/bin/python3
```

8 个 rank 同时：

```text
ModuleNotFoundError: No module named 'lerobot'
```

## 15.2 原因

实际调用的是：

```text
/usr/local/bin/torchrun
```

该入口绑定系统 Python，因此子进程使用：

```text
/usr/bin/python3
```

而不是：

```text
/output/lerobot-dcu/.venv/bin/python
```

## 15.3 解决

禁止使用裸：

```bash
torchrun
```

改成：

```bash
/output/lerobot-dcu/.venv/bin/python \
    -m torch.distributed.run
```

---

# 16. 问题十：8 个 rank 能启动，但找不到 lerobot

## 16.1 现象

8 卡 launcher 成功启动，但 worker 报：

```text
ModuleNotFoundError: No module named 'lerobot'
```

## 16.2 原因

LeRobot 是 editable/source 安装，源码位于：

```text
/output/lerobot-dcu/lerobot/src
```

多进程环境下如果没有把该路径显式加入 `PYTHONPATH`，worker 不一定能解析到 LeRobot 源码。

## 16.3 解决

明确：

```bash
export PYTHONPATH=/output/lerobot-dcu/lerobot/src:/output/lerobot-dcu/.venv/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages
```

训练入口不用 console script：

```text
/output/lerobot-dcu/.venv/bin/lerobot-train
```

而直接使用模块：

```bash
-m lerobot.scripts.lerobot_train
```

这样每个 rank 都由：

```text
/output/lerobot-dcu/.venv/bin/python
```

直接执行 LeRobot 模块。

---

# 17. 8 卡 K100 rank 验证

在真正训练前，建立：

```bash
cat > /tmp/check_8rank.py <<'PY'
import os
import sys
import torch
import lerobot

rank = os.environ.get("RANK")
local_rank = os.environ.get("LOCAL_RANK")

torch.cuda.set_device(int(local_rank))

print(
    f"rank={rank}, "
    f"local_rank={local_rank}, "
    f"python={sys.executable}, "
    f"gpu={torch.cuda.current_device()}, "
    f"name={torch.cuda.get_device_name()}, "
    f"lerobot={lerobot.__file__}",
    flush=True,
)
PY
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/output/lerobot-dcu/.venv/bin/python \
-m torch.distributed.run \
    --standalone \
    --nproc-per-node=8 \
    /tmp/check_8rank.py
```

实际成功结果应包含：

```text
rank=0 local_rank=0 ... gpu=0 name=K100_AI
rank=1 local_rank=1 ... gpu=1 name=K100_AI
rank=2 local_rank=2 ... gpu=2 name=K100_AI
...
rank=7 local_rank=7 ... gpu=7 name=K100_AI
```

并且每个 rank 的：

```text
python=
```

均应为：

```text
/output/lerobot-dcu/.venv/bin/python
```

这一步已经验证：

```text
8 × K100
    ↓
8 个 rank
    ↓
正确 Python
    ↓
正确 GPU 绑定
```

全部成功。

---

# 18. 问题十一：训练要求 policy.repo_id

## 18.1 现象

训练进入配置校验后报：

```text
ValueError:
'policy.repo_id' argument missing.
Please specify it to push the model to the hub.
```

## 18.2 原因

LeRobot v0.4.4 的 policy 默认：

```text
push_to_hub = True
```

因此如果没有：

```text
policy.repo_id
```

配置校验会失败。

当前 OpenBayes 又不能访问 Hugging Face，因此本来也不应上传模型。

## 18.3 解决

训练命令加入：

```bash
--policy.push_to_hub=false
```

这样只保存本地 checkpoint。

---

# 19. 问题十二：8 卡 DDP 已成功，但数据集不存在且服务器无外网

## 19.1 现象

训练日志已经出现：

```text
ProcessGroupNCCL ... size: 8
```

说明 8 卡进程组建立成功。

之后进入：

```text
Creating dataset
```

然后报：

```text
FileNotFoundError:
.../aloha_sim_insertion_human/meta/info.json
```

接着 LeRobot 尝试访问：

```text
https://huggingface.co/api/datasets/lerobot/aloha_sim_insertion_human/refs
```

最终：

```text
Network is unreachable
```

## 19.2 原因

OpenBayes 正式运行环境没有 Hugging Face 外网访问能力。

当本地数据集不存在时，LeRobot 会尝试从 Hugging Face 获取 metadata / dataset。

因此：

```text
本地数据不存在
       ↓
LeRobot 尝试 Hugging Face
       ↓
OpenBayes 无外网
       ↓
训练失败
```

## 19.3 解决

数据集必须提前离线下载并上传。

最终服务器路径固定为：

```text
/output/datasets/aloha_sim_insertion_human
```

训练时显式指定：

```bash
--dataset.root=/output/datasets/aloha_sim_insertion_human
```

---

# 20. Windows 上离线下载 ALOHA 数据集

本地电脑为 Windows，因此在 Windows PowerShell 中完成数据准备。

## 20.1 创建目录

```powershell
mkdir D:\lerobot_data
cd D:\lerobot_data
```

## 20.2 用 uv 创建下载环境

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 禁止脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

安装：

```powershell
uv pip install "huggingface_hub==0.35.3"
```

## 20.3 创建下载脚本

```powershell
@'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="lerobot/aloha_sim_insertion_human",
    repo_type="dataset",
    revision="v3.0",
    local_dir=r"D:\lerobot_data\aloha_sim_insertion_human",
)

print("Downloaded to:", path)
'@ | Set-Content download_dataset.py
```

运行：

```powershell
python .\download_dataset.py
```

## 20.4 验证数据

```powershell
Test-Path D:\lerobot_data\aloha_sim_insertion_human\meta\info.json
```

应返回：

```text
True
```

目录应至少包含：

```text
D:\lerobot_data\aloha_sim_insertion_human\
├── meta\
│   └── info.json
├── data\
└── videos\
```

---

# 21. Windows 上离线下载 ResNet18 预训练权重

ACT 默认视觉 backbone：

```text
ResNet18_Weights.IMAGENET1K_V1
```

如果服务器本地不存在权重，Torchvision 可能尝试联网下载。

因此也需要提前准备。

PowerShell：

```powershell
mkdir D:\lerobot_data\torch_cache\hub\checkpoints -Force
```

下载：

```powershell
Invoke-WebRequest `
  -Uri "https://download.pytorch.org/models/resnet18-f37072fd.pth" `
  -OutFile "D:\lerobot_data\torch_cache\hub\checkpoints\resnet18-f37072fd.pth"
```

检查：

```powershell
Get-Item D:\lerobot_data\torch_cache\hub\checkpoints\resnet18-f37072fd.pth
```

---

# 22. 上传到 OpenBayes

可使用 FileZilla。

Windows：

```text
D:\lerobot_data\aloha_sim_insertion_human
```

上传至：

```text
/output/datasets/
```

最终得到：

```text
/output/datasets/aloha_sim_insertion_human/
```

ResNet18：

```text
D:\lerobot_data\torch_cache\hub\checkpoints\resnet18-f37072fd.pth
```

上传到：

```text
/output/torch_cache/hub/checkpoints/
```

服务器检查：

```bash
ls -lh /output/datasets/aloha_sim_insertion_human/meta/info.json

ls -lh /output/torch_cache/hub/checkpoints/resnet18-f37072fd.pth
```

---

# 23. TF32 / ROCm 提示是否需要处理

训练日志中会出现：

```text
torch.backends.cuda.matmul.allow_tf32 is not supported on ROCm by default.
Please set environment variable HIPBLASLT_ALLOW_TF32=1 to enable it.
```

这是海光 / ROCm 兼容后端的提示，不是错误。

当前为了优先验证训练稳定性：

**不必设置 `HIPBLASLT_ALLOW_TF32=1`。**

等完整训练稳定后再进行性能调优。

---

# 24. NCCL / ProcessGroupNCCL 日志是否正常

8 卡训练时会看到：

```text
ProcessGroupNCCL initialization options: size: 8
```

以及各个 rank：

```text
Rank 0
Rank 1
...
Rank 7
```

这说明多卡通信已经建立。

虽然海光平台底层并非 NVIDIA GPU，但 DTK/PyTorch 兼容层仍通过 PyTorch 的 NCCL 风格接口完成多卡通信。

只要：

```text
size: 8
```

且 8 个 rank 均成功创建，就说明多卡初始化成功。

---

# 25. 当前最终版本组合

建议固定以下版本，不要再随意升级：

```text
Python                3.10.13
LeRobot               0.4.4
PyTorch               2.7.1（海光系统版）
torchvision           0.22.0（海光系统版）
NumPy                 1.25.0
huggingface-hub       0.35.3
transformers          4.57.1
tokenizers            0.22.1
video backend         PyAV
```

禁止直接执行：

```bash
uv sync
```

也不要随意执行：

```bash
uv pip install -U torch
pip install -U torch
pip install torchvision
```

否则很可能重新拉取官方 NVIDIA PyTorch。

---

# 26. 推荐制作固定激活脚本

创建：

```bash
cat > /openbayes/home/lerobot-dcu/activate_lerobot.sh <<'EOF'
#!/bin/bash

source /openbayes/home/lerobot-dcu/.venv/bin/activate

export PYTHONPATH=/openbayes/home/lerobot-dcu/lerobot/src:/output/lerobot-dcu/.venv/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages

export TORCH_HOME=/openbayes/home/torch_cache

echo "=============================================="
echo " LeRobot DCU environment activated"
echo "=============================================="
echo "Python: $(which python)"
echo "PYTHONPATH=$PYTHONPATH"
echo "TORCH_HOME=$TORCH_HOME"
EOF

chmod +x /openbayes/home/lerobot-dcu/activate_lerobot.sh
```

以后只需：

```bash
source /openbayes/home/lerobot-dcu/activate_lerobot.sh
```

---

# 27. 环境快速检查指令

```bash
source /openbayes/home/lerobot-dcu/activate_lerobot.sh

python - <<'PY'
import sys
import numpy
import torch
import torchvision
import huggingface_hub
import transformers
import lerobot

print("======================================")
print("LeRobot DCU Environment")
print("======================================")

print("python       :", sys.executable)
print("lerobot      :", lerobot.__file__)

print("numpy        :", numpy.__version__, numpy.__file__)
print("torch        :", torch.__version__, torch.__file__)
print("torchvision  :", torchvision.__version__, torchvision.__file__)

print("hf hub       :", huggingface_hub.__version__, huggingface_hub.__file__)
print("transformers :", transformers.__version__, transformers.__file__)

print("DCU available:", torch.cuda.is_available())
print("GPU count    :", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PY
```

正常应得到：

```text
torch        2.7.1
torchvision  0.22.0
numpy        1.25.0
hf hub       0.35.3
transformers 4.57.1

DCU available True
GPU count     8

GPU 0 K100_AI
...
GPU 7 K100_AI
```

---

# 28. 最终任务运行指令：部署完成后如何直接演示成功

本节是整个文档中最重要的一节。

假设以下内容已经准备好：

```text
/output/lerobot-dcu/.venv
/output/lerobot-dcu/lerobot

/output/datasets/aloha_sim_insertion_human/meta/info.json

/output/torch_cache/hub/checkpoints/resnet18-f37072fd.pth
```

那么每次重新打开 OpenBayes 实例后，只需要执行以下步骤。

---

## 28.1 第一步：进入 LeRobot 环境

如果已经创建激活脚本：

```bash
source /openbayes/home/lerobot-dcu/activate_lerobot.sh
```

如果没有激活脚本，则执行：

```bash
source /openbayes/home//lerobot-dcu/.venv/bin/activate

export PYTHONPATH=/openbayes/home/lerobot-dcu/lerobot/src:/openbayes/home/lerobot-dcu/.venv/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages

export TORCH_HOME=/openbayes/home/torch_cache
```

---

## 28.2 第二步：确认离线数据完整

```bash
test -f /output/datasets/aloha_sim_insertion_human/meta/info.json \
    && echo "[OK] LeRobot dataset exists" \
    || echo "[ERROR] dataset missing"

test -f /output/torch_cache/hub/checkpoints/resnet18-f37072fd.pth \
    && echo "[OK] ResNet18 weights exist" \
    || echo "[ERROR] ResNet18 weights missing"
```

应看到：

```text
[OK] LeRobot dataset exists
[OK] ResNet18 weights exist
```

---

## 28.3 第三步：确认 8 张 K100

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("DCU:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

应看到：

```text
GPU count: 8

0 K100_AI
1 K100_AI
2 K100_AI
3 K100_AI
4 K100_AI
5 K100_AI
6 K100_AI
7 K100_AI
```

---

## 28.4 第四步：8 卡 rank 演示

如果 `/tmp/check_8rank.py` 不存在：

```bash
cat > /tmp/check_8rank.py <<'PY'
import os
import sys
import torch
import lerobot

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)

print(
    f"rank={rank} "
    f"local_rank={local_rank} "
    f"python={sys.executable} "
    f"gpu={torch.cuda.current_device()} "
    f"name={torch.cuda.get_device_name()} "
    f"lerobot={lerobot.__file__}",
    flush=True,
)
PY
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/output/lerobot-dcu/.venv/bin/python \
-m torch.distributed.run \
    --standalone \
    --nproc-per-node=8 \
    /tmp/check_8rank.py
```

出现：

```text
rank=0 ... gpu=0 name=K100_AI
...
rank=7 ... gpu=7 name=K100_AI
```

即可证明 8 卡 launcher 正常。

---

## 28.5 第五步：启动最终 8 卡 ACT 训练

为了避免旧输出目录冲突，先生成新运行目录：

```bash
RUN_NAME="act_aloha_8xk100_$(date +%Y%m%d_%H%M%S)"

echo $RUN_NAME
```

然后启动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/output/lerobot-dcu/.venv/bin/python \
-m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --dataset.root=/output/datasets/aloha_sim_insertion_human \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=/output/lerobot-dcu/outputs/${RUN_NAME} \
    --job_name=${RUN_NAME} \
    --batch_size=8 \
    --steps=20000 \
    --num_workers=4 \
    --log_freq=50 \
    --save_checkpoint=true \
    --save_freq=2500 \
    --wandb.enable=false
```

当前配置：

```text
GPU 数量            8
每卡 batch size     8
global batch size   64
training steps      20000
checkpoint          每 2500 step 保存
WandB               关闭
Hugging Face push   关闭
video decoder       PyAV
```

---

## 28.6 第六步：另开终端观察 8 张 K100

```bash
watch -n 1 hy-smi
```

成功时应看到：

- 8 张 K100 均存在训练进程；
- 8 张卡显存均有占用；
- 8 张卡利用率持续变化；
- 不再只有 GPU0 工作。

可进一步查看训练进程：

```bash
ps -ef | grep -E "lerobot_train|torch.distributed" | grep -v grep
```

---

## 28.7 演示成功判据

只要同时满足以下条件，就可以认为：

> **LeRobot 已在海光 K100 平台上完成部署并成功进入 8 卡训练。**

判据：

```text
[1] torch.cuda.is_available() == True
[2] torch.cuda.device_count() == 8
[3] GPU 0~7 均识别为 K100_AI
[4] 8 个 DDP rank 均成功建立
[5] ProcessGroupNCCL size == 8
[6] LeRobot 成功创建 dataset
[7] ACT 成功初始化
[8] 出现正常 step / loss 训练日志
[9] hy-smi 中 8 张 K100 均有持续负载
[10] checkpoint 能写入 output_dir
```

---

# 29. 最终部署过程中问题汇总

| 序号 | 问题 | 根因 | 最终解决方案 |
|---|---|---|---|
| 1 | uv 要安装 torch 2.10 + NVIDIA CUDA | uv resolver 重新解析 Torch | 禁止 uv 覆盖 Torch，复用系统海光 PyTorch |
| 2 | 大量依赖冲突 | `--system-site-packages` 暴露整个基础镜像 | 只保证 LeRobot 必要依赖，不修整套系统 |
| 3 | Hub 0.35.3 安装后仍加载 1.5.0 | 系统 `PYTHONPATH` 优先级更高 | 重排 `PYTHONPATH` |
| 4 | unset PYTHONPATH 后 Torch 消失 | 海光 Torch 本身在 dist-packages | 不删除路径，只调整顺序 |
| 5 | NumPy 2.2.6 报 ABI 错误 | 海光 Torch 由 NumPy 1.x ABI 构建 | NumPy 固定 1.25.0 |
| 6 | rerun-sdk 要 NumPy >=2 | Rerun 与海光 Torch ABI 冲突 | 训练环境移除 Rerun |
| 7 | torchcodec 不可用 | 二进制后端不兼容 | 使用 PyAV |
| 8 | Transformers 导入失败 | 系统 Transformers 5.3 与 Hub 0.35.3 不兼容 | `.venv` 固定 Transformers 4.57.1 |
| 9 | accelerate 启动失败 | 调用了 `/usr/local/bin/accelerate` | 不用系统 accelerate |
| 10 | torchrun worker 用 `/usr/bin/python3` | 调用了系统 torchrun | 使用 `.venv/bin/python -m torch.distributed.run` |
| 11 | worker 找不到 lerobot | source 路径未传入多进程环境 | 将 `lerobot/src` 加入 PYTHONPATH |
| 12 | policy.repo_id missing | 默认 push_to_hub=True | `--policy.push_to_hub=false` |
| 13 | dataset info.json 不存在 | 数据未预先下载 | Windows 下载并上传到 `/output/datasets` |
| 14 | Hugging Face Network unreachable | OpenBayes 无外网 | 完全离线运行 |
| 15 | ResNet18 可能联网下载 | ACT 默认使用 ImageNet 预训练 backbone | 提前下载权重并设置 `TORCH_HOME` |
| 16 | TF32 ROCm warning | 海光 / ROCm 后端默认未开启 TF32 | 当前忽略，稳定后再优化 |

---
![alt text](image/image.jpg)

![alt text](image/image-1.png)