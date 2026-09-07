# 面向 WAAM 缺陷图像增广的 Hybrid CVAE-CGAN（非官方参考实现）

**语言 / Language:** [English](README.md) | 中文

> **同步声明**：英文版 `README.md` 为规范版本（normative）。本中文版与其保持同步更新；若两者出现不一致，以英文版为准。

**维护状态：已完成 / 按现状提供。** 本仓库是一份已完成的参考实现，不再主动开发。Issue 可能无人回复；需要新功能请 [fork](CONTRIBUTING.md)，不要等待。已发布协议中的致命缺陷仍可能被修复。

这是 Yang 等人 hybrid CVAE-CGAN 协议的非官方、已完成参考实现。与原作者无关联。用它在公开焊缝数据上运行该方法，并引用原论文。本仓库不再主动开发。

> Junle Yang, Lei Yuan, Haochen Mu, Fengyang He, Donghong Ding, Zengxi Pan, Huijun Li, *"Generation of WAAM Defect Images Using a Hybrid CVAE-CGAN: A Data Augmentation Strategy for Small and Imbalanced Datasets"*, Proc. 15th IEEE Int. Conf. on CYBER Technology in Automation, Control, and Intelligent Systems (CYBER 2025), Shanghai, China, 15-18 July 2025. DOI: [10.1109/CYBER67662.2025.11168313](https://doi.org/10.1109/CYBER67662.2025.11168313)

会议论文省略的架构与超参细节，取自同一第一作者的期刊扩展：

> Junle Yang, Lei Yuan, Fengyang He, Zening Wu, Donghong Ding, Zengxi Pan, Huijun Li, *"Physics-guided generative data augmentation for vision-based signal processing under class-imbalanced conditions in directed energy deposition monitoring system"*, Mechanical Systems and Signal Processing, vol. 250, article 114138, 2026. **开放获取（CC BY 4.0）**：DOI: [10.1016/j.ymssp.2026.114138](https://doi.org/10.1016/j.ymssp.2026.114138)

该方法是**单个联合训练的 hybrid 模型**：同一个解码器同时是 CVAE 解码器 `D(z, y)` 与 CGAN 生成器 `G(z, y)`。训练最小化

```
L_G = MSE_recon  +  beta * KL  +  lambda * VGG19_perceptual  +  gamma * adversarial
```

更长的操作说明（期刊扩展开关、自有数据迁移、仓库结构、完整局限）见 [`docs/USAGE.md`](docs/USAGE.md)。与论文的实测偏离见 [`docs/CALIBRATION.md`](docs/CALIBRATION.md)。

## 本仓库是什么、不是什么

本仓库是对 Yang 等人所述 hybrid CVAE-CGAN 训练与增广协议的**从零复现**。贡献在于别人不必再重写训练循环，也不必重新踩标定坑。与论文的每一处偏离都记录在 [`docs/CALIBRATION.md`](docs/CALIBRATION.md)。

它**不是**原作者的代码、通用图像生成库、持续运营的社区脚手架，或即插即用的工业工具。绝对数字与此处使用的公开替代数据（LoHi-WELD）以及从零训练的 ResNet-18 分类器绑定，不可与论文的专有结果比较。

## 范围一览

引用任何数字前请先读下面五层。

**论文方法（已实现）。** 单阶段 hybrid CVAE-CGAN：同一解码器同时是 `D(z, y)` 与 `G(z, y)`；联合 MSE + KL + VGG19 感知 + 对抗损失；Sub-Pixel 解码器；balance-to-max filling-rate 扫描。

**本仓库适配。** PyTorch 而非 TensorFlow/Keras。KL 权重按本仓库均值归一化的 [0, 1] 损失换算（latent 32 时 `0.015`，latent 128 时 `0.059`，对应论文 `beta = 30`）。对抗项是 hinge + spectral normalisation（期刊扩展），不是会议论文的 BCE。编码器卷积体是 4x4 stride-2 DCGAN 块，不是论文的 3x3 残差。下游分类器是单帧上从零训练的 ResNet-18。默认 FID 使用 [pytorch-fid](https://github.com/mseitzer/pytorch-fid)（官方 TensorFlow Inception 权重）。本仓库已发表的 FID 数字用的是旧的 torchvision 路径（`--fid_backend legacy`）。

**数据不等价。** 论文训练于专有可见光熔池图像（1,898 张、9 类）。本仓库使用公开 LoHi-WELD 焊缝裁剪，外加模拟的小而失衡 `--subset`。模态匹配；缺陷语义、相机几何与尺度不匹配。论文报告的数字因数据专有而无法被外部验证；本仓库的每一个数字都可基于公开数据端到端复现。

**不可与论文直接比较。** 准确率、F1、FID 绝对值、DR/MDR。LoHi-WELD 没有非缺陷类，因此 DR/MDR 在此不可定义。

**未完整复现。** 期刊 physics-guided 损失；21 帧时序增广与 LSTM/GRU 分类器；论文残差编码器体；FFT 去噪已实现但默认关闭。

## 免责声明

- 本仓库是**仅基于已发表论文的独立复现**，**与原作者及其机构无隶属、背书或任何关联**。
- 原作者的 WAAM 数据集为专有数据，本项目**未使用、未访问、未索取**。
- 所有实验均在**公开可获取、许可开放的数据集**上进行。各数据集自身许可适用。
- 本仓库不复制论文中的任何图、表或文字。
- 本项目复现的是*方法*，**不**声称复现论文的实验结果或报告数字。

## 快速开始

在 12 GB GPU 上，论文规模子集上推荐的 70-epoch 生成器大约需要**一小时**。五比例分类器扫描通常要几小时。冒烟测试只需几分钟。见 [`docs/USAGE.md`](docs/USAGE.md#compute-order-of-magnitude) 与 [`docs/CALIBRATION.md`](docs/CALIBRATION.md) 第 4 节计时表。

测试通过的 Python 版本为 3.11（CI 所用版本）；其他版本未测试。

```bash
pip install -r requirements.txt

# 合成数据自检——任何代码改动后都应先跑这个
python src/smoke_test.py

# 训练联合模型（推荐的 LoHi-WELD 配置）
python src/train_joint.py --data_root data/lohi \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --val_per_class 200 \
  --epochs 70 --batch_size 8 --img_size 224 --channels 3 --latent_dim 128 \
  --kl_weight 0.059 --d_norm group --weighted_sampler \
  --out_dir runs/joint_lohi_recommended

# 生成 balance-to-max 池
python src/generate.py --ckpt runs/joint_lohi_recommended/joint.pt \
  --counts "pore=560,deposit=450,discontinuity=300,stain=0" --out_root generated

# 下游扫描：每个 filling rate 一个从零训练的分类器
python src/train_classifier.py --data_root data/lohi --gen_root generated \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --ratios "0.0,0.25,0.5,0.75,1.0" --channels 3 --epochs 100 --out_dir runs/sweep

# 论文风格图
python src/make_paper_figures.py --data_root data/lohi \
  --ckpt runs/joint_lohi/joint.pt --history runs/joint_lohi/history.csv \
  --sweep runs/sweep/sweep_metrics.csv --cm_dir runs/sweep \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --channels 3 \
  --out_dir results

# 独立 FID（默认后端：pytorch-fid）
python src/eval_fid.py --real_root data/lohi --fake_root generated --channels 3
# 本仓库已发表的 FID 数字用的是：
#   python src/eval_fid.py ... --fid_backend legacy
```

会议论文默认（latent 32、BatchNorm、不加权采样）见 [`CHANGELOG.md`](CHANGELOG.md) 的 v0.2.0。期刊扩展开关、迁移到你自己的类文件夹图像、完整局限列表：[`docs/USAGE.md`](docs/USAGE.md)。

## 数据

当前实验使用 **[LoHi-WELD](https://github.com/SylvioBlock/LoHi-Weld)**（Block et al., IEEE Access 2024）：可见光焊缝图像、四个缺陷类，允许免费研究与商用、需引用。下载后只制备 `high_resolution_welds`：

```bash
python src/prepare_yolo_crops.py \
  --input_root <archive>/weld-dataset/high_resolution_welds \
  --out_root data/lohi --img_size 224 --channels 3 --min_side 16 \
  --classes "pore,deposit,discontinuity,stain"
```

这会生成 8,012 个裁剪块（deposit 1,193、discontinuity 2,975、pore 304、stain 3,540；失衡约 11.6×）。主实验使用其中的小 `--subset`，而非全部 8,012 张。224x224 画布高估了真实分辨率：源标注框中位约 47 px，因此几乎所有裁剪都是上采样（测量见 [`docs/CALIBRATION.md`](docs/CALIBRATION.md) 第 4 节）。

本仓库不打包任何数据集。`data/`、`generated/`、`runs/` 与 `*.pt` 按设计被 git-ignore。出处与许可见 [`DATA_SOURCES.md`](DATA_SOURCES.md)。RIAWELC 仅为历史脚注；其数字见 [`CHANGELOG.md`](CHANGELOG.md)（v0.1.0），请勿引用。

## 结果 - LoHi-WELD（当前）

本节按发布原样保留 v0.2.0 的单 seed 表格；表格后的三 seed 注记（v0.3.0 加入）是 r=1.0 结论的当前读法。

单 seed（42）。生成器：在论文规模子集（pore 40 / deposit 150 / discontinuity 300 / stain 600）上联合训练 CVAE-CGAN，70 epoch 中于第 25 epoch 早停（最佳 epoch 15）。诊断：FID 从 371 降至 216 后在 ~200 平台（仅作诊断，不可与任何已发表 FID 比较）；重建 MSE 0.049–0.060，对照常数均值基线 0.062；样图网格显示清晰的类形态。

**Filling-rate 扫描**——每个 ratio 一个从零训练的 ResNet-18，各 100 epoch，全部在同一个 held-out 纯真实测试集（1,603 张）上评估：

| ratio | accuracy | macro-F1 | weighted-F1 | deposit | discontinuity | pore | stain |
|---|---|---|---|---|---|---|---|
| 0.00（仅真实） | 0.8222 | 0.6936 | 0.8177 | 0.6521 | 0.9145 | 0.3778 | 0.8301 |
| **0.25** | **0.8353** | **0.7264** | **0.8306** | 0.6727 | 0.9203 | 0.4731 | 0.8393 |
| 0.50 | 0.8178 | 0.6767 | 0.8078 | 0.6108 | 0.9097 | 0.3590 | 0.8273 |
| 0.75 | 0.8141 | 0.7029 | 0.8064 | 0.5957 | 0.8977 | **0.4902** | 0.8281 |
| 1.00（balance-to-max） | 0.7392 | 0.6418 | 0.7403 | 0.6199 | 0.7932 | 0.3871 | 0.7669 |

两个按实测报告的发现：

1. **增广帮助少数类。** pore（304 张真实裁剪）在 r=0.25 获 +0.095 F1、r=0.75 获 +0.112；macro-F1 在 r=0.25 达峰，较仅真实基线 +0.033。
2. **论文的 balance-to-max 建议在这张单 seed 表上不迁移。** 此处 r=1.0 是最差比例。请把“填到 N_max”当作在你自己数据上待检验的假设，而非默认。

> **结论 2 已被部分取代，上方表格按发布原样保留。**
> 在三个彼此独立的生成器/生成池/分类器 seed 上，推荐配置
> （`--d_norm group --weighted_sampler`、`--latent_dim 128`、`--kl_weight 0.059`、
> 跑满 70 epoch）在 r=1.0 的 macro-F1 为 **0.7185 ± 0.0019**，相对
> 0.6785 ± 0.0447 的纯真实均值高 +0.0399；pore F1 为 **0.4658 ± 0.0475**，
> 高 +0.1132。这支持 r=1.0 的*平均*正向作用，但不支持单调曲线或 r=1.0
> 是唯一最优。匹配的 BatchNorm、未加权采样对照在 r=1.0 也有正向结果，
> 因此 GroupNorm 和加权采样**不是**符号反转的必要条件。完整表格和限制见
> [`docs/CALIBRATION.md`](docs/CALIBRATION.md) 第 9 节。

`results/` 中的图。**每张图都属于某一次特定运行**：

| 图 | 由哪次运行产出 |
|---|---|
| `filling_rate_multiseed.png` | 上文三 seed GroupNorm + 加权采样扫描 |
| `filling_rate_curve.png` | 已发布的 v0.2.0 单 seed 扫描（`runs/sweep`） |
| `training_curves.png`、`reconstruction_comparison.png`、`latent_tsne.png` | 已发布的 v0.2.0 生成器（`runs/joint_lohi`） |
| `confusion_matrices.png`（r=0 vs r=1）、`class_distribution.png` | 已发布的 v0.2.0 扫描 |
| `real_vs_generated.png`、`class_*.png` | 推荐的 `runs/paper2_gn_wrs` 生成器 |

第一行与最后一行的复现命令见 `docs/CALIBRATION.md` 第 9 节。

![按 seed 聚合的填充率曲线](results/filling_rate_multiseed.png)

阴影带是三个 seed 上的 ±1 样本标准差；`n=2` 标记 r=0.25，该比例下 seed 42 那一臂因最后 epoch 的 loss 尖峰被排除。这张图支持的是 r=1.0 的结论，而不是三种配置逐比例的排序。

**已提交的 LoHi-WELD showcase。** 来自推荐生成器 `runs/paper2_gn_wrs/joint.pt` 的定性示例。特写上的按类 FID 只是仓库内部诊断量，不可与文献比较。v0.2.0 旧版本保留为 `results/v0.2.0_*.png`。

![LoHi-WELD 真实与生成样本](results/real_vs_generated.png)

| deposit | discontinuity |
|---|---|
| ![deposit 特写](results/class_deposit.png) | ![discontinuity 特写](results/class_discontinuity.png) |
| pore | stain |
| ![pore 特写](results/class_pore.png) | ![stain 特写](results/class_stain.png) |

已发表表格使用 `--selection final`。当前分类器默认是 `--selection best_val`。下游模型是单帧上从零训练的 ResNet-18，而非论文的 LSTM/GRU。完整注意事项见 [`docs/USAGE.md`](docs/USAGE.md#limitations) 与 [`docs/CALIBRATION.md`](docs/CALIBRATION.md) 第 10 节。

## 许可证

代码：[MIT](LICENSE)。默认 FID 后端依赖
[pytorch-fid](https://github.com/mseitzer/pytorch-fid)（Apache-2.0），见
[NOTICE](NOTICE)。第三方数据集保留其自身许可。`results/` 中已提交的图由 LoHi-WELD 影像派生，因此仍受该数据集引用要求约束。

## 引用方式

如果本复现对你的工作有帮助，请引用**原始论文**与**数据集**而不是本仓库。[`CITATION.cff`](CITATION.cff) 列两组：被复现论文加每个数据集的强制引用，以及实现组件引用（FID、t-SNE、InceptionV3、ResNet-18、VGG19）。

```bibtex
% --- 主方法参考（被复现的会议论文）---
@inproceedings{yang2025hybridcvaecgan,
  title     = {Generation of WAAM Defect Images Using a Hybrid CVAE-CGAN:
               A Data Augmentation Strategy for Small and Imbalanced Datasets},
  author    = {Yang, Junle and Yuan, Lei and Mu, Haochen and He, Fengyang and
               Ding, Donghong and Pan, Zengxi and Li, Huijun},
  booktitle = {Proc. 15th IEEE International Conference on CYBER Technology
               in Automation, Control, and Intelligent Systems (CYBER 2025)},
  address   = {Shanghai, China},
  month     = jul,
  year      = {2025},
  pages     = {1--6},
  doi       = {10.1109/CYBER67662.2025.11168313}
}

% --- 次要方法参考（期刊扩展；提供会议论文省略的细节）。CC BY 4.0 开放获取。---
@article{yang2026physicsguided,
  title   = {Physics-guided generative data augmentation for vision-based
             signal processing under class-imbalanced conditions in directed
             energy deposition monitoring system},
  author  = {Yang, Junle and Yuan, Lei and He, Fengyang and Wu, Zening and
             Ding, Donghong and Pan, Zengxi and Li, Huijun},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {250},
  pages   = {114138},
  year    = {2026},
  doi     = {10.1016/j.ymssp.2026.114138}
}

% --- 以下两条 RIAWELC 引用是该数据集的强制使用条款 ---
@inproceedings{totino2022riawelc,
  title     = {RIAWELC: A Novel Dataset of Radiographic Images for
               Automatic Weld Defects Classification},
  author    = {Totino, Benito and Spagnolo, Fanny and Perri, Stefania},
  booktitle = {Proc. Interdisciplinary Conference on Mechanics, Computers
               and Electrics (ICMECE 2022)},
  address   = {Barcelona, Spain},
  month     = oct,
  year      = {2022}
}

@article{perriweldingdefects,
  title   = {Welding Defects Classification Through a Convolutional
             Neural Network},
  author  = {Perri, Stefania and Spagnolo, Fanny and Frustaci, Fabio and
             Corsonello, Pasquale},
  journal = {Manufacturing Letters},
  volume  = {35},
  pages   = {29--32},
  year    = {2023},
  doi     = {10.1016/j.mfglet.2022.11.006}
}

% --- 作为主实验的可见光焊缝缺陷数据集 ---
@article{block2024lohiweld,
  title   = {LoHi-WELD: A Novel Industrial Dataset for Weld Defect Detection
             and Classification, a Deep Learning Study, and Future
             Perspectives},
  author  = {Block, Sylvio Biasuz and Dutra da Silva, Ricardo and
             Lazzaretti, Andre Eugenio and Minetto, Rodrigo},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {77442--77453},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3407019}
}
```
