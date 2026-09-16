# SRDTrans 中文使用说明

SRDTrans 用于荧光图像时间序列的自监督去噪。它直接使用低信噪比数据训练，不需要配对的干净图像。

本仓库的基本流程是：

```text
单通道 TIFF 时间序列
        ↓
放入 datasets/<数据集名称>/
        ↓
train.py 自监督训练
        ↓
模型保存在 pth/<数据集名称_时间>/
        ↓
test.py 分块去噪并拼接
        ↓
结果保存在 results/
```

## 1. 当前电脑上的运行环境

本机已经存在可用的 Conda 环境 `srdtrans`：

- Python 3.6.15
- PyTorch 1.10.2 + CUDA 11.3
- torchvision 0.11.3
- NumPy 1.19.5
- tifffile 2020.9.3
- scikit-image 0.17.2
- einops 0.4.1
- timm 0.6.12
- PyYAML 6.0.1
- matplotlib 3.3.4

在 PowerShell 中进入项目并激活环境：

```powershell
cd "E:\Wokspace\CAPilot workspace\代码\Denoise\SRDTrans"
conda activate srdtrans
```

检查 PyTorch 和显卡：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

正常情况下应看到 `True` 和显卡名称。本仓库的训练、推理代码都直接调用 `.cuda()`，因此当前版本不能只用 CPU 运行。

如果需要在另一台电脑复现当前环境，可以安装已经验证过的版本组合：

```powershell
conda create -n srdtrans python=3.6
conda activate srdtrans
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install numpy==1.19.5 tifffile==2020.9.3 einops==0.4.1 timm==0.6.12 tqdm==4.64.1 scikit-image==0.17.2 matplotlib==3.3.4 pyyaml==6.0.1
```

> 原项目发布较早，不建议直接使用全部最新版依赖。若新显卡无法使用上述旧版 CUDA 构建，再根据当前驱动选择新版 PyTorch，并重新验证训练和推理。

## 2. 准备输入数据

在 `datasets` 下新建一个数据文件夹，例如：

```text
datasets/
└── my_data/
    ├── stack_01.tif
    └── stack_02.tif
```

输入数据需要满足以下要求：

- 每个文件都是单通道 TIFF stack，数组维度为 `[T, Y, X]`。
- 多通道 Hyperstack 要先按通道拆分，每个通道保存成独立的 TIFF stack。
- TIFF 文件直接放在数据文件夹中，不要再嵌套子文件夹。
- 文件夹内只放待处理的 TIFF，不要混入说明文件或其他格式。
- 每个维度不能小于对应补丁尺寸，即 `T >= patch_t`，`X、Y >= patch_x`。
- 默认保持原始位深，例如 `uint16`；一般不要预先归一化成 0–1。

`--datasets_folder` 只填写 `datasets` 下的文件夹名称，例如 `my_data`，不要填写完整路径。

## 3. 先做一次小规模训练测试

当前电脑只有一张 RTX 4050 Laptop GPU（6 GB），必须使用 `--GPU 0`，不能照抄原 README 中的 `--GPU 0,1`。

建议先用较小补丁和 1 个 epoch 验证数据格式及完整流程：

```bash
python train.py \
  --datasets_path datasets \
  --datasets_folder my_data \
  --GPU 0 \
  --n_epochs 1 \
  --train_datasets_size 100 \
  --patch_x 64 \
  --patch_t 64
```

如果命令开始输出 TIFF shape、epoch、batch 和 loss，说明训练流程已经正常启动。

训练结束后会生成类似目录：

```text
pth/
└── my_data_202609151230/
    ├── para.yaml
    └── E_01_Iter_XXXX.pth
```

注意：训练脚本中的 `--pth_path` 当前没有真正控制保存位置，模型固定保存在项目根目录的 `pth/` 下。

## 4. 正式训练

确认小规模测试正常后，可增加训练轮数和抽取补丁数：

```bash
python train.py \
  --datasets_path datasets \
  --datasets_folder my_data \
  --GPU 0 \
  --n_epochs 20 \
  --train_datasets_size 6000 \
  --patch_x 64 \
  --patch_t 64
```

参数说明：

| 参数                      | 含义                      | 建议                            |
| ------------------------- | ------------------------- | ------------------------------- |
| `--datasets_path`       | 数据集根目录              | 保持`datasets`                |
| `--datasets_folder`     | 根目录下的数据文件夹名    | 例如`my_data`                 |
| `--GPU`                 | 使用的 GPU 编号           | 本机使用`0`                   |
| `--n_epochs`            | 训练轮数                  | 测试用 1，正式训练可从 20 开始  |
| `--train_datasets_size` | 期望抽取的训练补丁数量    | 默认 6000；实际数量可能略有差异 |
| `--patch_x`             | X、Y 方向补丁大小         | 6 GB 显存先用 64                |
| `--patch_t`             | 时间方向补丁长度          | 6 GB 显存先用 64                |
| `--overlap_factor`      | 相邻补丁重叠比例          | 默认 0.5，一般不改              |
| `--select_img_num`      | 每个 stack 最多使用的帧数 | 数据太长时可用于截断            |
| `--scale_factor`        | 输入强度缩放因子          | 保持 1                          |

`patch_x` 和 `patch_t` 建议使用 8 的倍数。原论文仓库示例使用 `160 × 160 × 160`，但对 6 GB 显存比较激进；建议先从 64 开始，确认显存余量后再逐步增加到 96 或 128。

训练会把所有 TIFF stack 读入内存。数据量很大时，应先选取代表性片段训练，避免系统内存不足。

## 5. 使用自己训练的模型推理

先查看训练产生的模型文件夹名称：

```powershell
Get-ChildItem .\pth -Directory
```

假设模型文件夹为 `my_data_202609151230`，执行：

```bash
python test.py \
  --datasets_path datasets \
  --datasets_folder my_data \
  --denoise_model my_data_202609151230 \
  --GPU 0 \
  --patch_x 64 \
  --patch_t 64
```

其中：

- `--denoise_model` 填写 `pth/` 下的文件夹名，不是 `.pth` 文件名。
- 推理时 `patch_t` 必须与训练该模型时的 `patch_t` 一致。
- 建议先让 `patch_x` 也与训练时一致；显存不足时可尝试减小 `patch_x`。
- 如果模型文件夹内有多个 `.pth`，程序只会使用按文件名排序后的最后一个，通常就是最后一个 epoch。

结果目录类似：

```text
results/
└── DataFolderIs_my_data_<时间>_ModelFolderIs_my_data_202609151230/
    ├── para.yaml
    └── E_20_Iter_XXXX/
        ├── stack_01_E_20_Iter_XXXX_output.tif
        └── stack_02_E_20_Iter_XXXX_output.tif
```

输出会尽量沿用输入数据类型：`uint16` 输入输出为 `uint16`，`int16` 输入输出为 `int16`，其他类型最终保存为 `int32`。

## 6. 使用预训练模型

从原项目 README 的 Model Zoo 下载模型后，整理成下面的结构：

```text
pth/
└── cad_03hz/
    └── model.pth
```

然后运行：

```bash
python test.py \
  --datasets_path datasets \
  --datasets_folder my_data \
  --denoise_model cad_03hz \
  --GPU 0 \
  --patch_x 64 \
  --patch_t 160
```

预训练模型的 `patch_t` 要使用发布该模型时的训练值。原 README 的预训练模型示例为 `patch_t=160`。在本机 6 GB 显存上，可以先将空间补丁 `patch_x` 降到 64；已验证 `patch_t=160、patch_x=64` 的单次 GPU 前向计算可以运行。

## 7. 常见问题

### `ValueError: num_samples should be a positive integer`

常见原因：

- TIFF 不是 `[T, Y, X]` 的单通道 stack；
- 数据维度小于补丁尺寸；
- 把多通道 Hyperstack 直接交给程序。

先用 ImageJ/Fiji 检查维度，并按通道拆分后重新保存。

### CUDA out of memory

依次尝试：

1. 保证使用 `--GPU 0`；
2. 将 `patch_x` 从 128 或 160 降到 64；
3. 训练自有模型时，将 `patch_t` 也降到 64；
4. 推理已有模型时不要随意修改 `patch_t`，它必须与模型训练值一致。

### 找不到数据或模型

检查目录名与参数是否对应：

```text
--datasets_path datasets
--datasets_folder my_data       -> datasets/my_data/
--denoise_model model_folder    -> pth/model_folder/
```

文件夹名称中不要带 `datasets/` 或 `pth/` 前缀。

### 出现 `torch.meshgrid` 警告

这是旧版实现与 PyTorch 的兼容性提示，不影响当前推理结果，可以暂时忽略。

### 输出非常亮、非常暗或强度异常

先确认训练和推理使用相同的原始数据强度范围，并保持 `--scale_factor 1`。该参数在现有推理代码中的强度处理并不适合随意修改。

## 8. 推荐的实际使用顺序

1. 用 Fiji 确认数据为单通道 `[T, Y, X]` TIFF stack。
2. 将少量代表性数据复制到 `datasets/my_data/`。
3. 用 `64 × 64 × 64`、1 epoch 做冒烟测试。
4. 打开输出模型目录，确认 `.pth` 和 `para.yaml` 已生成。
5. 用同一批数据执行一次推理，检查输出 TIFF 的 shape、位深和信号强度。
6. 再开始正式训练，并用独立数据评价去噪效果，避免只看训练数据。
