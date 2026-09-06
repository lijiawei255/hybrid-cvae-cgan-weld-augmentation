# 面向 WAAM 缺陷图像增广的 Hybrid CVAE-CGAN（非官方复现）

**语言 / Language:** [English](README.md) | 中文

> **同步声明**：英文版 `README.md` 为规范版本（normative）。本中文版与其保持同步更新；若两者出现不一致，以英文版为准。

本仓库是对下列论文所提 hybrid CVAE-CGAN 框架的**非官方、方法学层面**的复现尝试：

> Junle Yang, Lei Yuan, Haochen Mu, Fengyang He, Donghong Ding, Zengxi Pan, Huijun Li, *"Generation of WAAM Defect Images Using a Hybrid CVAE-CGAN: A Data Augmentation Strategy for Small and Imbalanced Datasets"*, Proc. 15th IEEE Int. Conf. on CYBER Technology in Automation, Control, and Intelligent Systems (CYBER 2025), Shanghai, China, 15-18 July 2025. DOI: [10.1109/CYBER67662.2025.11168313](https://doi.org/10.1109/CYBER67662.2025.11168313)

会议论文省略的架构与超参细节，取自同一第一作者的期刊扩展：

> Junle Yang, Lei Yuan, Fengyang He, Zening Wu, Donghong Ding, Zengxi Pan, Huijun Li, *"Physics-guided generative data augmentation for vision-based signal processing under class-imbalanced conditions in directed energy deposition monitoring system"*, Mechanical Systems and Signal Processing, vol. 250, article 114138, 2026. **开放获取（CC BY 4.0）**：DOI: [10.1016/j.ymssp.2026.114138](https://doi.org/10.1016/j.ymssp.2026.114138)

如果本仓库对你的工作有帮助，请引用**上述原始论文**而不是本仓库——它们是方法的来源，本复现的存在正是为了把读者引向它们。

本仓库作为**公开参考**，面向任何想做类似复现的人：记录从论文复现了什么、使用了哪个公开替代数据集、以及实现与原文在哪些地方偏离。

该方法生成类条件焊缝缺陷图像，用于增广小而失衡的数据集。它是**单个联合训练的 hybrid 模型**，而非两阶段流水线：同一个解码器同时充当 CVAE 解码器 `D(z, y)` 与 CGAN 生成器 `G(z, y)`，这正是此处 "hybrid" 的含义。训练最小化如下联合目标：

```
L_G = MSE_recon  +  beta * KL  +  lambda * VGG19_perceptual  +  gamma * adversarial
```

生成器与判别器在每个 minibatch 内交替更新、各自反向传播。KL 项把隐空间塑向 `N(0, I)`，使得生成阶段用 `z ~ N(0, I)` 加类标签采样可行；感知项与对抗项负责锐化纯 VAE 会留下的模糊。

## 免责声明

- 本仓库是**仅基于已发表论文的独立复现**，**与原作者及其机构无隶属、背书或任何关联**。
- 原作者的 WAAM 数据集为专有数据，本项目**未使用、未访问、未索取**。
- 所有实验均在**公开可获取、许可开放的数据集**上进行（见 `DATA_SOURCES.md`）。各数据集自身许可适用；署名要求保留在数据下载说明中。
- 本仓库不复制论文中的任何图、表或文字。
- 本项目复现的是*方法*，**不**声称复现论文的实验结果或报告数字。

## 快速开始

```bash
pip install -r requirements.txt

# 合成数据自检——任何代码改动后都应先跑这个
python src/smoke_test.py

# 训练联合模型。--subset 以类名键入，定义小而失衡的训练集；
# held-out 测试划分先切出，因此生成器永远看不到它。
python src/train_joint.py --data_root data/lohi \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --val_per_class 200 \
  --epochs 70 --batch_size 8 --img_size 224 --channels 3 --latent_dim 32 \
  --lr 1e-3 --lr_d 1e-3 --kl_weight 0.015 --perc_weight 0.1 --adv_weight 0.1 \
  --out_dir runs/joint_lohi

# 生成 balance-to-max 池：filling rate 1.0 每类所需数量。
# 类名从 checkpoint 读取并校验，绝不假设。
python src/generate.py --ckpt runs/joint_lohi/joint.pt \
  --counts "pore=560,deposit=450,discontinuity=300,stain=0" --out_root generated

# 下游扫描：每个 filling rate 一个从零训练的分类器，
# 全部在同一个 held-out 真实测试集上评估。
python src/train_classifier.py --data_root data/lohi --gen_root generated \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --ratios "0.0,0.25,0.5,0.75,1.0" --channels 3 --epochs 100 --out_dir runs/sweep

# 论文风格图（训练曲线、filling-rate 敏感性、latent t-SNE、
# 真实/重建/生成、混淆矩阵、类别分布）
python src/make_paper_figures.py --data_root data/lohi \
  --ckpt runs/joint_lohi/joint.pt --history runs/joint_lohi/history.csv \
  --sweep runs/sweep/sweep_metrics.csv --cm_dir runs/sweep \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --channels 3 \
  --out_dir results

# 独立 FID：两棵图像目录之间
python src/eval_fid.py --real_root data/lohi --fake_root generated --channels 3
```

### 取自期刊扩展版的开关

上面的快速开始复现的是*会议*论文的配置。`train_joint.py` 另有四个开关，实现的是只出现在期刊扩展版（MSSP 2026）中的组件。四者的默认值都保持会议论文的行为，因此不显式启用就什么都不会变。

| 开关 | 默认值 | 期刊扩展版取值 | 状态 |
|---|---|---|---|
| `--d_norm` | `batch` | `group` | 已实测，推荐 |
| `--weighted_sampler` | 关闭 | 开启 | 已实测，推荐 |
| `--kl_warmup` | 关闭（beta 恒定） | `10,50` | 已实现并通过单元测试，**未经训练验证** |
| `--monitor` | `val_loss` | - | 仅在与 `--kl_warmup` 同时使用时才需要 |

- `--d_norm group` 把判别器的 BatchNorm 换成 GroupNorm。在 `--batch_size 8` 下，BatchNorm 会让一个样本的得分依赖于同批次另外七个样本；GroupNorm 在单个样本内部归一化。实测效果：判别器 hinge loss 的中位数从 0.819（判别器占优）变为 1.998，与期刊论文所述其"稳定在约 2.0 的常数值附近"一致。
- `--weighted_sampler` 用 `WeightedRandomSampler` 抽取训练批次，权重正比于 `1/N_class`。`--subset` 只限制每类数量，并**不**平衡批次：1,090 张的子集里一个 40 张的少数类只占 3.7% 的抽样，因此 batch size 为 8 时，多数 minibatch 里根本没有少数类样本。
- `--kl_warmup ZERO,RAMP` 先把 beta 保持为 0 共 `ZERO` 个 epoch，然后在 `RAMP` 个 epoch 内线性升到 `--kl_weight`。它需要期刊论文的 200-epoch 协议才有意义；在 70-epoch 的运行里只有 10 个 epoch 会跑在满 beta 下，因此本仓库刻意不训练它。
- `--monitor val_recon` 仅以重建 MSE 挑选最佳 checkpoint。在 `--kl_warmup` 下必须使用，因为总损失包含 `beta * KL`，其含义会随 beta 爬升而改变。

同时启用 `--d_norm group --weighted_sampler` 就是 `docs/CALIBRATION.md` 第 9 节实测的那个配置。复用其中数字前请先读该节的两条限定：那一臂同时改了两个变量，因此两个效果都没有被单独归因；并且它的 FID 变差而重建变好。

### 使用你自己的数据集

把 `--data_root` 指向一个"类名子文件夹"结构即可。其余均与数据集无关：

- 类别身份在所有地方都**以类名**表达（`--subset`、`--counts`），且每个类名都会对照数据集或 checkpoint 校验。未知或拼错的类名会报错，而不是静默选错类。
- `--subset` 的数量请依据你自己的每类数量设定。方法面向*小而失衡*数据，因此少数类数量要低到任务尚未被解决，多数类数量设为你希望 balance-to-max 填到的 `N_max`。
- `--channels 3` 适用于彩色数据；`--img_size` 必须能被 16 整除。
- `--fft_denoise` 启用论文的 FFT 低通预处理。默认关闭，且在本仓库使用的两个数据集上都实测接近空操作。RIAWELC 的放射图本身频带受限。LoHi-WELD 的裁剪块把 99.2–99.9% 的频谱能量放在归一化半径 0.1 以内，因此在论文的 `cutoff = 0.25` 下，滤波对缺陷边缘所在中频段的保留率为 **0.0%**，对图像的改变在 [0, 255] 尺度上仅为 MSE 3–14（上限为 65,025）。它在那里之所以无效，是因为这些裁剪块本身就是小源框的上采样，而上采样无法创造高频内容——所以滤波主要做的是让*目标*更平滑。它带来的任何表面提升都应读作"任务变简单了"，而非"方法变强了"。见 `docs/CALIBRATION.md` 第 3 节。若你自己的图像带有真实的高频传感器噪声，可以打开；同时请也给 `src/eval_fid.py` 传该参数，否则 FID 测到的是预处理不一致，而不是生成质量。
- 若你改动损失归一化，`--kl_weight` 必须重新换算。见 `models.cvae_loss` 与 `CHANGELOG.md` 中的推导。

## 结果 - v0.2.0（LoHi-WELD，当前）

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

1. **增广帮助少数类。** pore（304 张真实裁剪，稀缺类）在 r=0.25 获 +0.095 F1、r=0.75 获 +0.112；macro-F1 在 r=0.25 达峰，较仅真实基线 +0.033。这是方法的核心论断，在公开数据集上得到演示。
2. **论文的 balance-to-max 建议在此不迁移。** r=1.0 是*最差*比例（macro-F1 0.642，低于 0.694 基线），而期刊扩展报告单调改善并在 1.0 达峰。可能原因是生成器保真度：r=1.0 时三个类约一半训练集为合成图，而我们的样本比论文的糊，灌满会稀释真实信号而非强化它。请把"填到 N_max"当作在你自己数据上待检验的假设，而非默认。

> **结论 2 已被部分取代，上方表格按发布原样保留。**
> 用期刊扩展版的配置（`--d_norm group --weighted_sampler`、`--latent_dim 128`、`--kl_weight 0.059`、跑满 70 epoch）重跑扫描后，**r=1.0 处的结论被反转**：该比例成为曲线上*最好*的一点，macro-F1 0.7168，相对本臂自己的纯真实基线 0.6848 高出 +0.032；pore F1 0.5053，相对已发布的 0.3871 高出 +0.118。在干净跑完的四个比例上，曲线单调上升（0.6848 -> 0.6932 -> 0.7102 -> 0.7168），这正是期刊论文报告、而 v0.2.0 曲线不具备的形状。新扫描的 r=0.25 臂**不是一个有效测量**：其 loss 在最后一个 epoch 尖峰了三个数量级，而分类器评分的正是最后一个 epoch。
>
> 请不要把这读成"GroupNorm 修好了 balance-to-max"。新配置与 v0.2.0 同时相差五个方面，而匹配对照——`runs/probe_eq_g0.1` 中那个 latent-128 的 BatchNorm 臂——尚未扫描，因此这次反转还不能归因于两个开关中的任何一个。它也仅是单 seed，而实测的同 seed 重跑波动为 pore F1 0.034。完整数字、两张表与全部四条限定见 `docs/CALIBRATION.md` 第 9 节。

`results/` 中的图：`filling_rate_curve.png`（上述扫描）、`training_curves.png`（各损失分量与每 epoch FID）、`reconstruction_comparison.png`（真实/重建/生成）、`latent_tsne.png`（编码器隐空间投影）、`confusion_matrices.png`（r=0 vs r=1）、`class_distribution.png`，以及每类特写（`class_*.png`）与真实-生成对照网格（`real_vs_generated.png`）。

**注意事项。** 单 seed，且分类器在 GPU 上并非逐位可复现：`train_classifier.py` 为所有随机源设了种子，但从未启用确定性 CUDA 算法，因此在同 seed 下重复*未改动*的纯真实臂，pore F1 变动了 0.034、macro-F1 变动了 0.009。小于该幅度的差异请在任一方向上都视为噪声——本段此前"~0.02 macro-F1"的估计对少数类过于乐观，而少数类只在 61 张测试图上评分。它还在恒定学习率下评估**最后**一个 epoch，没有早停也没有最佳 epoch 选择，因此一次后期 loss 尖峰就足以让某一臂整体失效。两者均已在 `docs/CALIBRATION.md` 第 10 节实测并解释。另外，下游分类器是单帧图像上从零训练的 ResNet-18，而非论文在 21 帧序列上的 LSTM/GRU，因此绝对分数不可与论文比较。其余标定记录见上文局限列表。

## 结果 - v0.1.0（已被取代，请勿引用）

> **这些数字来自已删除的两阶段骨架在全量数据集上的运行，那不是本仓库现在遵循的协议。** FID 值另外**无效**：当时把 [0, 1] 图像喂给期望 [-1, 1] 的 InceptionV3（`transform_input=False`），所有激活都偏离分布。它们不可与文献 FID 比较、不可跨该修复互相比较、也不可与当前结果比较。保留它们仅因 v0.1.0 曾带着它们发布。见 `CHANGELOG.md`。

全量 RIAWELC 数据集（24,407 张，128x128，batch 64）上的单 seed 运行。这些数字描述*本复现*在替代数据集上的表现，**不**可与原论文结果比较。

| 阶段 | 设置 | 结果 |
|---|---|---|
| Stage 1 CVAE | 100 epochs, latent dim 128 | 最终 L1 重建 0.102, KL ~7.0 |
| Stage 2 CGAN | 200 epochs, G 由 CVAE 解码器初始化 | 最终 D loss 0.38, 无模式坍缩 |
| FID (InceptionV3) | 24,407 真实 vs 4,000 生成 | 37.96 |

下游增广增益实验（预训练 ResNet-18，分层 80/20 纯真实测试划分，seed 42）：条件 A = 失衡真实子集（每缺陷类保留 20%，约 7.7k 张），条件 B = A + 每缺陷类 1,000 张生成图（约 10.7k）。两条件均达 ~99.4% 测试准确率（macro-F1 0.9931 vs 0.9932）——**该数据集上无可测增广增益**：ImageNet 预训练分类器即使在失衡基线下也几乎完美解决 RIAWELC，任务饱和、没有留给增广的空间。论文的设置（1,898 张专有熔池图、最少每类 36 张）难得多；要演示增益需要更难的替代任务。该负面结果按原样报告。协议说明见 `src/train_classifier.py`；生成器曾在全部真实图上训练，故生成数据可能携带测试图信息（与论文相同的协议局限）。

每类真实 vs 生成样本（上排真实，下排生成）：

![Real vs generated samples](results/real_vs_generated.png)

每类特写（上排真实，下排生成；标题中的数字是该类 FID——越低越好；归一化非标准，因此只可类间互比，不可与文献值比较）：

| CR（裂纹）, FID 71.9 | LP（未焊透）, FID 63.5 |
|---|---|
| ![CR close-up](results/class_CR.png) | ![LP close-up](results/class_LP.png) |
| **PO（气孔）, FID 83.5** | **ND（无缺陷）, FID 14.7** |
| ![PO close-up](results/class_PO.png) | ![ND close-up](results/class_ND.png) |

## 仓库结构

```
src/models.py             编码器、解码器(=生成器)、判别器、损失
src/data.py               类文件夹加载器、FFT 去噪、无泄漏划分
src/augment.py            balance-to-max filling-rate 协议
src/prepare_yolo_crops.py 检测格式数据集 -> 类文件夹裁剪(含 ROI 步骤)
src/train_joint.py        联合 CVAE-CGAN 训练(论文的单阶段)
src/generate.py           类条件采样, 数量以类名键入
src/eval_fid.py           FID(InceptionV3 特征) 与输入归一化
src/train_classifier.py   下游 filling-rate 扫描(从零 ResNet-18)
src/make_paper_figures.py 训练曲线、敏感性曲线、t-SNE、混淆矩阵
src/make_comparison.py    真实-生成对照网格, 输出到 results/
src/make_class_figures.py 每类特写(含每类 FID)
src/smoke_test.py         单元检查 + 合成数据端到端运行
```

## 数据

当前实验使用 **[LoHi-WELD](https://github.com/SylvioBlock/LoHi-Weld)**（Block et al., IEEE Access 2024）：3,022 张**可见光**焊缝图像、四个缺陷类，允许免费研究与商用、需引用。与被复现论文的熔池图像一样是可见光；其高分辨率焊缝裁出 8,012 个缺陷补丁、类失衡 **11.6×**——接近会议论文的 14.7×。

它以检测数据形式发布（图像 + YOLO 标签对），因此 `src/prepare_yolo_crops.py` 把每个标注框裁进类文件夹；该裁剪同时就是论文的 ROI 提取预处理步骤。精确制备命令、每类数量与许可条款见 `DATA_SOURCES.md`。

**RIAWELC**（X 光放射图）曾被 v0.1.0 与生成器标定记录使用，现在仅为历史脚注——当前没有任何数字源自它。其引用被保留，因为那些已公开的结果用过它。见 `DATA_SOURCES.md`。

本仓库不打包任何数据集，也永远不应提交——`data/`、`generated/`、`runs/` 与 `*.pt` 按设计被 git-ignore。下载链接、使用条款与许可干净的替代见 `DATA_SOURCES.md`。

## 局限与诚实说明

引用本仓库任何数字前请先读此节。

- **数据不同、模态相同。** 论文在专有可见光熔池图像上训练（1,898 张、9 类、最高 14.7× 失衡）。本仓库使用公开 LoHi-WELD 可见光焊缝，因此*模态*现在匹配，但缺陷语义、相机几何与尺度仍不同。此处任何结果都不复现或验证论文的数字。
- **"小而失衡"条件是模拟的。** LoHi-WELD 的 8,012 裁剪、11.6× 失衡比论文设置更大更温和，因此主实验抽取一个小而失衡的子集（`--subset "pore=40,deposit=150,discontinuity=300,stain=600"`，15× 失衡，balance-to-max 目标 600）以重建论文条件。这些数量是你应依据自己数据集设定的参数，不是方法的属性。
- **下游分类器不是论文的分类器。** 两篇论文都在 21 帧时序序列上评估 LSTM/GRU 模型。静态焊缝图像没有时间轴，无法构造序列；故以单帧上从零训练的 ResNet-18 替代，尊重论文对轻量非预训练模型的偏好。准确率与 F1 因此不可与论文比较——只有协议可比。
- **FID 在此是趋势指标，不是绝对分数。** 数据集、分辨率与参考集都不同于任何已发表 FID。只在本仓库内比曲线，绝不跨仓库比。
- **四个超参是标定的，不是照抄的。** KL 权重（0.015 vs 论文 30）、判别器学习率、FFT 去噪步（关闭）、分辨率（224 vs 400）各自因实测原因与会议论文不同。对抗目标遵循**期刊扩展**（hinge + spectral normalisation）而非会议论文的 BCE minimax，因为无界 BCE 项在该数据规模下摧毁重建。证据与复现命令：[`docs/CALIBRATION.md`](docs/CALIBRATION.md)。
- **单 seed。** 所有报告数字均为 seed 42 的单次运行。该规模下 GAN 训练有噪声；小于几个点的差异请视为噪声内。
- **构造上无泄漏。** 生成器永远看不到分类器的 held-out 测试图。这比严格必要更严，是对论文 "real, unseen images" 的诚实读法；也意味着报告的增益（若有）未被测试集信息 inflate。

### 范围：本仓库刻意不复现的部分

- **期刊扩展的物理引导生成不在范围内。** 它相对会议论文的贡献是对生成熔池信号的热力学/边界约束。静态焊缝图像不含可条件化的热/时序信号，因此这些约束在此不可实现、也未被尝试。本仓库只借用期刊的 GAN 目标（hinge + spectral normalisation）与少数超参细节。
- **时序/序列增广不在范围内。** 期刊增广并评估 21 帧序列；本仓库生成并评估单帧，因为两个替代数据集都是静态图像。
- **报告的运行不含论文的 FFT 去噪步。** 它已实现（`--fft_denoise`）但关闭：在 RIAWELC 上实测无必要、在 LoHi-WELD 上未测试。因此 v0.2.0 的数字描述的是*不含*该预处理步的流水线。
- **结构与目标来自不同论文。** 训练结构（单阶段联合模型、Sub-Pixel 解码器、70 epoch、lr 1e-3）遵循会议论文；对抗目标（hinge + spectral normalisation）遵循期刊扩展，因为 BCE 在该数据规模下被证明不稳定。
- **编码器卷积体是已记录的偏离。** 论文用 3×3 残差块 + max pooling（会议）或 GroupNorm（期刊），我们用 plain 4×4 stride-2 块 + BatchNorm；只有聚合方式（1×1 缩减 + 池化）对齐。判别器与会议拓扑一致、仅差省略的 dropout。卷积拓扑不是两篇论文声称的贡献、它们也未消融它，因此任何质量差异不能归因于它；模块签名使替换块对需要架构完全一致的人是 drop-in 改动。
- **论文的 DR/MDR 指标未报告。** 它们是缺陷 vs 正常的比率；LoHi-WELD 的四类全是缺陷类型、没有非缺陷类，因此这些比率在此不可定义。每类 recall 是最接近的类比。

## 复用本工作

想生成你自己的训练数据？准备类文件夹图像（或按 `DATA_SOURCES.md` 拉取 LoHi-WELD），运行 `src/train_joint.py`，再用 `src/generate.py` 以 `src/augment.py` 的每类数量采样。你训练的模型生成的图像归你使用；代码为 MIT 许可。欢迎链回本仓库，但不强制。

## 许可证

代码：[MIT](LICENSE)。第三方数据集保留其自身许可。

## 引用方式

如果本复现对你的工作有帮助，请引用**原始论文**与**数据集**而不是本仓库。[`CITATION.cff`](CITATION.cff) 列两组：被复现论文加每个数据集的强制引用（请引这些），以及第二组实现组件引用——FID、t-SNE、InceptionV3、ResNet-18、VGG19——为本仓库分析与骨干所用组件的署名。

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
  note    = {In press, Elsevier. Recorded as "in press" by the RIAWELC
             repository, so no year, volume or pages are asserted here}
}

% --- 作为主实验的可见光焊缝缺陷数据集 ---
@article{block2024lohiweld,
  title   = {LoHi-WELD: A Novel Industrial Dataset for Weld Defect Detection
             and Classification, a Deep Learning Study, and Future
             Perspectives},
  author  = {Block, Sylvio Biasuz and Dutra da Silva, Ricardo and
             Lazzaretti, Andre Eugenio and Minetto, Rodrigo},
  journal = {IEEE Access},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3407019}
}
```
