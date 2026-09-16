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
结果默认保存在输入文件同目录，格式与输入一致
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

默认结果保存到每个输入文件所在目录。例如：

```text
datasets/my_data/
├── stack_01.tif
├── stack_01_denosied_PFC.tif
├── stack_01_denosied_PFC.yaml
├── stack_02.tif
├── stack_02_denosied_PFC.tif
└── stack_02_denosied_PFC.yaml
```

输出会尽量沿用输入数据类型：`uint16` 输入输出为 `uint16`，`int16` 输入输出为 `int16`，其他类型最终保存为 `int32`。

`--output_path "D:\降噪结果"` 可指定输出目录，程序直接在该目录生成结果，不再创建时间或模型子目录。不填写或填写空值时，结果与源文件同目录。输出文件保留输入扩展名，并添加 `_denosied_<模型名>` 后缀（按约定使用此拼写）。模型名取 `--denoise_model` 指定的模型文件夹名称；例如 `--denoise_model cad_03hz` 生成 `stack_01_denosied_cad_03hz.tif`。同一个输入使用不同模型时分别保存结果；重复使用同一模型会替换同名结果。目录批处理会跳过已有的 `*_denosied` 和 `*_denosied_<模型名>` 图像文件。

### 直接读取 H5 或外部文件

`test.py` 支持 `.tif`、`.tiff`、`.h5`、`.hdf5`，`--datasets_path` 可直接填写单个文件或图像目录。此时可以省略 `--datasets_folder`；原来的“根目录 + 子文件夹”参数写法也继续支持。`train.py` 仍只支持 TIFF。

H5 读写使用 `h5py`，Zstd 压缩还需要 `hdf5plugin`。本机旧的 Python 3.6 环境可使用 `python -m pip install h5py==3.1.0 hdf5plugin==3.3.1`；现代 Python 环境可使用 `uv pip install h5py hdf5plugin`。本程序在读取 H5 时自动注册压缩过滤器；其他程序读取 Zstd H5 时也需要支持该过滤器，例如 Python 中先执行 `import hdf5plugin`。

例如，直接处理 CAPilot 导出的 H5：

```powershell
python test.py --datasets_path "E:\Wokspace\CAPilot workspace\data\3_信号提取\movie_export_f1-1000.h5" --denoise_model "你的模型文件夹名" --GPU 0 --patch_x 64 --patch_t 64
```

H5 自动优先选择 `/images`、`/data`、`/mov`；没有这些名称时，选择唯一的三维数值数据集。其他情况需通过 `--h5_dataset "/group/movie"` 指定。默认存储顺序为 `(T, Y, X)`，其他顺序可通过 `--h5_axis_order yxt` 等参数指定，读取后统一转换为 `(T, Y, X)`。程序不根据维度大小猜测轴顺序。

`--test_datasize 64` 可限制实际读取的 H5 帧数，必须不少于 `patch_t`。推理仍将选取的图像数据及拼接结果存入内存，并非全程流式处理。TIFF 输入输出 TIFF，H5 输入输出 H5。例如使用 `--denoise_model cad_03hz` 处理 `movie_export_f1-1000.h5`，默认在同目录生成 `movie_export_f1-1000_denosied_cad_03hz.h5`，源文件保持不变。TIFF 输出默认使用 Deflate 无损压缩。H5 输出在 `/images` 保存降噪图像，维度统一为 `(T, Y, X)`，默认使用 Zstd level 3 无损压缩，并设置 `shuffle=True`；输出不复制源 H5 的其他数据集或属性。每个结果旁还保存同名 `.yaml` 推理参数。

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
  --patch_t 128
```

预训练模型的 `patch_t` 必须与实际权重匹配。本地 `cad_03hz/Pretrained_0.3Hz_50mWpower_1000frames.pth` 的时间位置编码长度为 8，当前模型在时间方向缩小 16 倍，因此应使用 `patch_t=128`。若使用 160，加载时会报位置编码 `[1, 8, 128]` 与 `[1, 10, 128]` 不匹配。在本机 6 GB 显存上，空间补丁可设为 `patch_x=64`。

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
