# 实验总结：九个 baseline 官方实现与配置核对

本次只补齐源码、训练设置和运行验收，没有启动完整训练、学习率搜索或新增种子。统一 MMSP 24→24、label_len=12，保留 in_domain 与 zeroshot_v1 两套站点、时间和训练 scaler 协议。官方设置并非对所有数据集唯一；本项目明确选取下面的作者脚本作为参考，数据窗口、通道数和硬件设置按 MMSP 协议适配。

## 核对与修正

| 模型 | 原状态 / 本次处理 | 当前官方参考设置 |
|---|---|---|
| DLinear | 原为 FusionSF 风格：坐标拼接、kernel13、常量初始化、AdamW。补入作者原模型，功率单通道、kernel25、正常 Linear 初始化 | ETTh1 单变量脚本：seed2021、batch32、Adam lr0.005、MSE、10epochs、patience3、type1 |
| PatchTST | 原为 TSLib/FusionSF 实现，d512/LayerNorm。补入作者 supervised 模型，RevIN、BatchNorm、残差注意力、可学习位置编码 | ETTh1：seed2021、batch128、d16/ff128、3layers/4heads、dropout0.3、patch16/stride8、Adam lr1e-4、MSE、100epochs、patience100、type3 |
| FusionSF | 保留官方三模态核心和 merged experiment 设置；修正 NWP guide 分支为官方17通道（经纬度+15气象变量） | fusionSF.sh：seed42、batch16、dim64/depth12/heads8、dropout0.4、ctx mask0.99、TS/context VQ；AdamW lr0.0016、wd0.05、betas0.9/0.95、L1+VQ、cosine warmup5、100epochs、MAE patience20 |
| Cross-Unet | 保留作者交叉 U-Net 核心；按 use_cross_unet.py 最终覆盖值运行 | seed2021、batch128、d256/ff512、3layers/4heads、seg12（MMSP24h适配）、Adam lr1e-4、MSE、100epochs、patience10、type3 |
| TimeXer | 保留官方 MS 外生变量核心；把旧100epochs/plateau调度改为参考脚本默认训练设置 | ETTh1 MS：seed2021、batch32、d512/ff512、2layers/8heads、Adam lr1e-4、MSE、10epochs、patience3、type1；patch16→12适配24小时窗口 |
| Chronos-2 | 已为官方冻结 Chronos2Pipeline；保留纯功率及原生 NWP 协变量接口 | chronos-forecasting2.2.2、官方本地chronos-2、median0.5、无梯度训练；不是 ChronosX |
| ChronosX | 新增指定论文的官方 chronosx 分支核心；不是 Chronos-2 加自定义层 | Chronos-T5-small、IIB+OIB、hidden256/1layer、冻结骨干、只训练 injection_block；token CE、AdamW lr0.01/wd0、linear、5000更新步、batch32、累积2、clip1、每100步按验证CE保存最优、20采样 |
| Time-LLM | 新增作者时间序列重编程、词原型、统计提示、冻结LLM、RevIN和预测头 | ETTh1短预测脚本：Llama-7B/32层、seed2021、d32/ff128、patch16/stride8、batch24、Adam lr0.01、MSE、100epochs、patience10、type1、bf16 |
| Time-VLM | 新增作者 full-shot CLIP 分支：序列生成图像、文本、局部/全局检索记忆、交叉注意力、门控融合 | ETTh1：CLIP ViT-B/32冻结、seed2024、d128、image56、periodicity24、norm0.4、bank100/top5、batch32、Adam lr0.001、MSE、10epochs、patience5、type1、GPU fp16 |

PatchTST、Cross-Unet 和 Time-LLM 作者训练代码创建 OneCycleLR 后，在这里选用的 epoch 调度下没有按 batch 更新它；构造器实际把首轮 lr 设为 base/25、Adam beta1 设为0.95。本项目复现这些实际生效的值，随后按作者 type1/type3 公式调度，未把它们替换为完整 OneCycle 训练。

DLinear/PatchTST 的旧模型及参数保留在 `--baseline_config legacy_fusionsf`，历史 checkpoint 不受覆盖。FusionSF 旧15通道 checkpoint 可加 `--guide_channels 15`。默认 run_id 带配置名称，防止将新官方模型写入旧结果目录。官方配置没有自动运行任何 sweep；ChronosX 官方实验曾搜索不同学习率，本次只接入其 API 默认0.01，不将它称为 MMSP 最优学习率。

## 数据与运行适配

九个模型都接受统一五项批次，经 `Dataset_MMSP` / `data_provider(args, flag)` 读取，无模型专用 Dataset。DLinear、PatchTST、Time-LLM、Time-VLM 使用功率历史；Cross-Unet/TimeXer 使用历史 NWP；ChronosX 使用历史和未来 NWP；FusionSF 使用历史卫星和未来 NWP。Chronos-2 按 nwp_mode 明确选择。各模型所用模态并不相同，后续性能比较必须注明。

推理时 decoder 的未来功率恒为零，ChronosX 的训练标签只传给 token_loss；未来真值不进入其 generate。ChronosX 按作者做协变量绝对均值归一化、缺失指示拼接及 EOS 行补齐。MMSP 在统一数据层已有 NWP 插值，因此其缺失指示通常为零。训练缺失增强最大0.2，在模型训练适配层执行，未建立新 Dataset。

作者部分 loader 会丢弃验证/测试末批、验证时 shuffle；本项目验证与测试保持顺序并覆盖全部窗口，训练 drop_last 按各参考实现设置。除 Time-LLM 官方 eval batch8 和 ChronosX eval batch8 外，验证/测试默认使用各自训练 batch。统计采用完整窗口的 MAE/RMSE/MSE/R²，验证早停按表中指标。原 FusionSF 全量 scaler 拟合不迁入，统一保留训练站点/时段 fit 的已验收协议。

运行仅用一张明确选定的空闲 GPU0，CPU/BLAS/PyTorch绑定最多4核，DataLoader workers0。Time-LLM 官方8卡训练适配为单卡，未声称有效 batch 或硬件与作者8卡实验完全一致。Time-VLM CPU精度回退fp32；GPU保持fp16。Transformers适配4.57.6以同时兼容已有Chronos-2，不降级其他项目环境。VQ库实际安装与项目固定依赖对齐为1.2.0；作者requirements未锁定该库版本，不能声称它是唯一官方版本。权重从项目本地读取，入口不自动下载。

Time-VLM 保留作者 forward 更新记忆库的行为；它依赖批次顺序与历史记忆，评估不 shuffle。本项目将原 Python 记忆库改为可保存、迁移设备的 buffers，连同指针保存到 checkpoint。作者将生成图像转为uint8再送入处理器，该路径不能将梯度传回生成图像的卷积；本次保留此官方行为，不声称这些卷积已经得到训练。作者 CLIP 分支按字符截断提示到77，迁入保留该行为；processor 补充 padding/truncation 以处理批量输入。仅迁入指定官方 CLIP 路径，未扩展 BLIP2/ViLT/自定义VLM实验。

## 结果边界

本次是代码与训练接口验收，没有新完整测试集 MAE/RMSE/R²；新配置对旧配置的绝对与相对性能变化均未测量，最佳新配置尚未确定。真实批次前向/梯度损失只说明可计算，不属于性能指标，不参与排名。旧 DLinear MAE0.0775414/RMSE0.1239254、PatchTST MAE0.0638855/RMSE0.1155920 均属于旧 FusionSF 配置，不能标作新作者实现结果。上述两项旧配置中 PatchTST 较好，但不能据此评价本次九模型。

## 保存位置与来源

- 模型适配：`models/{DLinear,PatchTST,FusionSF,CrossUnet,TimeXer,Chronos2,ChronosX,TimeLLM,TimeVLM}.py`；作者计算层：`layers/`。
- 默认设置：`models/official_presets.py`；各官方仓库、版本与所选脚本：`models/baseline_sources.json`；离线参考脚本副本：`scripts/official_reference/`（仅txt，不作为新运行入口）。
- 本地预训练资源：`pretrained/chronos-2/`、`chronos-t5-small/`、`clip-vit-base-patch32/`、`llama-7b/`；此前 TimesFM3 保留。
- 本次验收记录和日志：`reports/baseline_acceptance_*_20261010.json/.log`、`baseline_weights*_20261010.log`。未生成正式实验 checkpoint、全量预测表或性能图表；另以4个真实窗口验证 DLinear epoch、ChronosX 2步更新和 Time-VLM fp16 epoch，其临时验收 checkpoint 位于 `reports/baseline_training_interface_20261010/`，不是正式实验最优模型。旧结果继续保存在 `reports/{protocol}/results.csv` 和 `checkpoints/{protocol}/legacy_20261010/`。
- 官方来源：[DLinear](https://github.com/cure-lab/LTSF-Linear)、[PatchTST](https://github.com/yuqinie98/PatchTST)、[FusionSF](https://github.com/MAZiqing/FusionSF)、[Cross-Unet](https://github.com/ZjuMachine/PV-power)、[TimeXer](https://github.com/thuml/TimeXer)、[Chronos-2](https://github.com/amazon-science/chronos-forecasting)、[ChronosX](https://github.com/amazon-science/chronos-forecasting/tree/chronosx)、[Time-LLM](https://github.com/KimMeen/Time-LLM)、[Time-VLM](https://github.com/moyutian/ICML25-TimeVLM)。上游有许可证的计算层保留对应文件；Time-VLM 所核对版本未提供独立 LICENSE，NOTICE 记录来源与改动，不另行赋予授权。

已验证实际训练/验证/最优 checkpoint 路径、Time-VLM 记忆库恢复、旧 DLinear/PatchTST checkpoint 加载及 wheel 打包。FusionSF 旧15通道/VQ checkpoint 也能严格加载；同两个真实窗口在旧VQ1.31.0与项目VQ1.2.0下最大预测差为0（仅批次兼容性核对，未重新测全量指标）。继承的旧 BasicTS 包仍有依赖冲突，属于原共享 Python 环境；PV-TSFM 自身 runtime 依赖核对无冲突，不加载 BasicTS。

Llama-7B 官方默认权重未下载完成：多次下载遇到慢速连接/TLS异常，已停止，不保留后台下载队列。Time-LLM 的真实权重运行核对采用作者支持的 GPT-2 完整12层（LLM dim768）；其余重编程架构与参考训练设置不变。GPT-2 运行验证不等于 Llama-7B 验证，正式论文配置的 Llama 部分仍须本地权重补齐后再验收。

最终验收：九模型 × 两协议共18个真实批次核对全部通过，输出 `[2,24,1]`；可训练模型梯度有限，Chronos-2骨干冻结。Time-LLM 通过的是官方支持的GPT-2 12层，默认Llama-7B权重和运行仍未齐全。详情在 `baseline_acceptance_20261010.json`。结论：九模型的代码与训练设置已接入统一入口，正式预测效果需用户选定下一项实验后再测量。
