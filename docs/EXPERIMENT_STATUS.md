# 实验状态与待办

更新时间：2026-10-13。PV-TSFM用于统一实验与可信论文Metric；历史探索提供研究线索。协议定义见 [DATA_PROTOCOLS.md](DATA_PROTOCOLS.md)，运行授权见根目录AGENTS.md。

## 实验总表

MMSP，seed42，24→24；收录本框架已验收的正式测试Metric、用户批准且经一致性验收的历史结果复用，以及本次明确批准展示的DLinear/PatchTST zeroshot历史登记成绩。`MMSP zeroshot`对应`zeroshot_v1`。总表CSV及以下展示均保留六位小数；原始指标精度保留在metrics.json与分协议结果索引；“—”表示暂无已验收成绩，TimeXer两个协议已完成完整产物验收。不同协议不直接排名，第二列input标明实际输入，未正式确认的版本写“待确认”。DLinear/PatchTST采用用户批准的FusionSF仓库旧实现/旧训练配置，不是原作者官方版本；其余版本及来源见下文。

| models | input | MMSP in_domain MAE | MMSP in_domain RMSE | MMSP in_domain R^2 | MMSP zeroshot MAE | MMSP zeroshot RMSE | MMSP zeroshot R^2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DLinear | 历史功率+经纬度 | 0.077541 | 0.123925 | 0.748166 | 0.078117 | 0.124453 | 0.746018 |
| PatchTST | 历史功率+经纬度 | 0.063885 | 0.115592 | 0.780897 | 0.064574 | 0.115093 | 0.782784 |
| FusionSF | 历史功率+历史卫星+未来NWP+坐标/时间编码 | 0.040876 | 0.091265 | 0.863416 | 0.041598 | 0.089822 | 0.867700 |
| Cross-Unet | 历史功率+历史/未来NWP+相关历史功率块 | 0.042289 | 0.083879 | 0.884629 | 0.044612 | 0.084163 | 0.883846 |
| Chronos-2 | 历史功率+历史/未来NWP | 0.060910 | 0.117771 | 0.772558 | 0.061052 | 0.117897 | 0.772070 |
| ChronosX | 历史功率+历史/未来NWP | 0.047528 | 0.108813 | 0.805843 | 0.048782 | 0.107841 | 0.809296 |
| TimeXer | 历史功率+历史NWP | 0.070145 | 0.117773 | 0.772551 | 0.072329 | 0.120635 | 0.761361 |
| Time-VLM | 历史功率+衍生图像/固定文本 | 0.068934 | 0.119808 | 0.764624 | 0.068747 | 0.119382 | 0.766292 |
| STAEformer | 历史功率+小时/星期 | — | — | — | — | — | — |

**DLinear/PatchTST的zeroshot三列为历史登记成绩，原Checkpoint、日志和预测等产物缺失，未完成当前完整验收；CSV同样适用此说明。** 来源：[历史登记](history_audit_20261010/experiment_registry.csv)的R0004/R0005；原始精度与来源记录保存在`reports/DLinear_PatchTST_migration_seed42_20261011/runs.json`，不登记到分协议正式测试索引。

**2026-10-16调参验收：本表TimeXer in_domain使用宽度128/128，Cross-Unet in_domain使用lr0.0002，均按验证RMSE预选；zeroshot保持原配置。完整来源见文末“有限调参最优配置正式测试”。**

[下载总表CSV](../reports/experiment_summary.csv)。指标来源：新运行的`outputs/{protocol}/{model_id}/metrics.json`或下方已验收历史证据路径，验收依据为各实验`runs.json`的acceptance字段；原分协议`reports/{protocol}/results.csv`继续作为正式测试索引。完成验收后同步更新此表，验证指标、历史探索与未验收测试默认不填入；上述两组历史登记成绩为用户明确授权的展示例外，不等同完整产物验收。Chronos-2固定采用history_future，是已验收历史推理结果复用，不是本轮重新推理；证据路径见下方记录。DLinear/PatchTST的in_domain已完成完整产物验收，zeroshot按用户授权从历史登记填入并明确标注产物缺失。

## Chronos-2四种输入Metric

MMSP，seed42，24→24；冻结权重、中位数预测、cross_learning=False。第二列为input，仅列指标，统一六位小数；两协议分别评价。四种历史输入版本经一致性验收后复用，主表版本由用户按输入方案固定为history_future，不按测试成绩选择。

| models | input | MMSP in_domain MAE | MMSP in_domain RMSE | MMSP in_domain R^2 | MMSP zeroshot MAE | MMSP zeroshot RMSE | MMSP zeroshot R^2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chronos-2 | 功率 | 0.066126 | 0.137008 | 0.692190 | 0.066126 | 0.137008 | 0.692190 |
| Chronos-2 | 功率+历史NWP | 0.067451 | 0.133809 | 0.706396 | 0.067627 | 0.133851 | 0.706212 |
| Chronos-2 | 功率+未来NWP | 0.063279 | 0.130402 | 0.721155 | 0.063249 | 0.130387 | 0.721222 |
| Chronos-2 | 功率+历史NWP+未来NWP | 0.060910 | 0.117771 | 0.772558 | 0.061052 | 0.117897 | 0.772070 |

[下载四种输入CSV](../reports/chronos2_input_metrics.csv)。唯一迁移验收与原始配置/日志/预测/指标路径：`reports/Chronos2_two_protocols_seed42_20261011/runs.json`；原始指标保留完整精度。

## 当前授权

- 2026-10-14最新任务：baseline统一实验完整验收后，在MMSP Site #0验证A01协变量负贡献。固定A01 Chronos-2+B131+Linear、seed42、24→24、原100/10/L1/AdamW/batch配置和60/20/20时间划分；每个NWP子集均从头训练B131及Linear，同时删除对应历史/未来NWP通道。阶段1为Full和15个leave-one-variable-out；按验证RMSE相对Full的严格改善识别负贡献变量。阶段2联合删除这些变量并重新训练，仍只按验证RMSE决定是否选择。选择决定写盘后才正式测试Full及联合删除组合；其他LOO不运行测试。阶段3使用验证预测按Site #0、24个horizon及各被删变量的训练集未来窗口均值三分位天气状态分析，不使用测试反馈。单站点任务的站点分组只有Site #0，不能推断跨站条件依赖。入口`scripts/run_site0_a01_covariate_ablation.py`，记录`reports/A01_site0_covariate_ablation_seed42_20261014/runs.json`；当前协调器只等待baseline验收，不提前启动消融。

- 2026-10-14：用户要求检查Time-VLM/TimeXer收敛、统一预测指标选模、明确输入差异，四个baseline的最大轮数/patience统一100/10。本轮采用验证RMSE；Cross-Unet复用已验收100/10的验证MSE最优结果（与RMSE等价），其余三模型两个协议从头训练，沿用seed42、24→24、原结构/输入/batch/精度及初始学习率。Time-VLM/TimeXer改用验证RMSE停滞3轮后学习率减半，ChronosX保留CE训练并以20次采样中位数的全量验证RMSE选模。无新种子、超参搜索或同输入对照自动扩展；最多两张空闲GPU、合计四CPU核。任务记录：`reports/Baseline_alignment_100_10_seed42_20261014/runs.json`。

- 2026-10-12 A04：用户批准实现并运行A04-a/b，随后补充in_domain也要做；两个协议seed42、A02预算，单GPU串行四组，验收后停止。证据为reports/Chronos2WeatherReadout_two_protocols_seed42_20261012/runs.json。

- 2026-10-12 A03：用户批准保留A02协议限制继续。三个独立版本的zeroshot_v1/seed42正式训练、测试及产物验收已完成；独立表已更新。本阶段停止，未启动多种子、容量控制或其他模型。证据为reports/Chronos2HistoryWeatherAdapter_zeroshot_seed42_20261012/runs.json。

- 2026-10-12当前：用户明确批准停止含卫星退化组，将FusionSF读出修复扩展到剩余两输入的两个协议。旧训练及队列已退出、产物保留；首种输入两个协议已正式验收，不重跑。剩余四组在GPU4完成真实批次功率监督梯度检查后，从seed42/epoch1串行重跑；正初始化+读出lr为主干1/100，ReLU/VQ/结构/协议和100/20预算不变。含卫星batch64，无卫星batch256，eval64、CPU0,7两线程、num_workers0；每组正式验收后继续，失败停止。

- 仅MMSP的in_domain与zeroshot_v1、seed42、24→24。
- 2026-10-11 06:52最新决定：用户停止TimeVLM、暂停GPU3，聚焦FusionSF利用率核对，并要求训练GPU利用率达不到99%就停止。同卡原版三模态与当前无卫星变体的短时训练步骤采样均未达99%，两项训练和两条队列已退出，ChronosX及其余FusionSF组未启动；已有权重/日志保留，不自动恢复。详情见“FusionSF原版对照与99%停止条件”。此前TimeVLM/ChronosX200/20及FusionSF100/20预算、seed42、24→24保持记录，其他已完成模型不改。
- 2026-10-11用户接受DLinear/PatchTST与原作者官方版本不同，批准将经验收的旧版本in_domain结果填表，并在主表第二列标明input；随后明确批准将zeroshot历史登记成绩填入主表，标注原产物缺失、未完成当前完整验收，不启动补跑。
- 2026-10-11用户批准Chronos-2固定history_future主版本的历史结果迁移验收，以及四种输入Metric表；仅复用已有完整推理证据，不启动训练、重新推理或调参。
- 最新恢复Site1探索协议，当前任务仅完成数据协议核对；协议检查通过，暂不修复FusionSF全零预测，不影响现有训练。尚未授权新Site1完整训练或官方架构变更。
- 用户明确要求估计Cross-Unet耗时后运行TimeXer；Cross-Unet已完整训练、测试并验收，TimeXer两个协议seed42已完成训练、正式测试与完整产物验收；本次明确授权的Chronos-2 Adapter两协议、两个版本已完成训练、测试与完整产物验收，当前无本任务训练进程。最多两张GPU、合计四个CPU核，DataLoader=0；框架、协议及官方核心结构不改。
- 10–20分钟随机间隔只读检查日志、存活、NaN/Inf及疑似不变/全零验证；不自动停止、调参或重跑，不发邮件。
- 顺序FusionSF→Cross-Unet→TimeXer；每个模型完成两个协议并验收后停止，等待下一模型审批。其他baseline和Ours不自动推进。

## Time-VLM复现（已完成）

运行名：`TimeVLM_official_clip_seed42_sl24_pl24_power`。in_domain使用GPU0/CPU0,7，zeroshot_v1使用GPU3/CPU9,10；每进程2线程、num_workers=0。两个协议已完成训练、正式测试与完整产物验收，主表已同步。

官方CLIP full-shot，MMSP历史功率单通道、24→24、seed42；batch/eval32、Adam lr0.001、MSE、type1、FP16、最多10轮、验证MSE选权重、patience5；dim128、patch16/stride8/padding8、记忆100/top5、图像56/periodicity24/norm_const0.4、CLIP冻结。本地依赖与权重齐全，CPU单批次及GPU混合精度前后向、末批次、记忆库保存恢复检查通过；单批次检查不是完整性能复现。官方固定版本已在线核对，既有移植差异与具体配置保存在runs.json。

保留官方行为：记忆在每次forward更新，包括顺序验证/测试，仅访问历史输入；文本按77字符截断后是相同固定前缀，统计信息和MMSP描述未进入CLIP；图像uint8转换使图像生成模块无梯度。记忆及指针随checkpoint保存；正式测试从最低验证MSE权重及其记忆状态开始。本次不修改这些行为，不使用NWP或MMSP卫星，后续对比须注明输入差异。原始配置、协议、完整预测、指标、训练日志和权重按既有目录保留；纯功率输入不拟合NWP/卫星scaler。


完整产物验收通过：全部25,450×24标签与原始CSV、已验收测试参考一致，站点/时间索引、配置/协议匹配；预测/权重有限、各站非全零；统一全量Metric重算与指标、日志及唯一分协议索引一致。各协议从保存权重及记忆状态顺序重放首64个样本进行身份检查，不重做完整推理。纯功率输入不拟合NWP/卫星scaler。

in_domain完成10轮，最低验证MSE权重第9轮；测试MAE/RMSE/R²=0.071056/0.122202/0.755121。同协议对照TimeXer：MAE -0.001486（-2.05%）、RMSE +0.002004（+1.67%）、R² -0.007966（-1.04%）。
zeroshot_v1完成10轮，最低验证MSE权重第9轮；测试MAE/RMSE/R²=0.069875/0.121659/0.757294。同协议对照TimeXer：MAE -0.001576（-2.21%）、RMSE +0.003355（+2.84%）、R² -0.013201（-1.71%）。

Time-VLM只消费历史功率及衍生图像/固定文本，TimeXer还消费历史NWP，差异不能单独归因于架构。仅seed42；官方文本截断、图像无梯度及顺序记忆更新行为保留，未开展消融或修复。两协议验收后按用户新授权串行执行FusionSF三个输入版本。唯一验收证据为本轮runs.json的acceptance；未生成新图表或重复Markdown报告。

## Chronos-2 Adapter迁移与复现（已完成）

用户明确批准保持PV-TSFM统一输入，在in_domain及zeroshot_v1重新训练seed42、24→24。旧FusionSF C/D/E缓存的全部功率输入、目标、站点及时间索引与当前zeroshot一致，窗口76,570/25,450/25,450；但旧天气是原始物理量、按变量名排序，当前为训练集StandardScaler、Dataset_MMSP顺序。因此旧Checkpoint不用于新训练初始化，旧特征和Metric不替代当前结果，两个协议均重新生成冻结特征并训练。

| 版本 | 历史名 | 结构 | seed42历史验证MAE | 累计学习参数 |
| --- | --- | --- | ---: | ---: |
| b131_linear_v1 | E11 | 冻结Chronos-2 + B131，随后冻结B131并训练Linear(1536,24)输出残差 | 0.039306 | 240540 |
| b131_mlp63_v1 | E12 | 同一B131前级 + Linear(1536,63)-GELU-Linear(63,24)输出残差 | 0.039179 | 302019 |

从已核对纯Chronos-2 Adapter候选按seed42验证MAE选这两项，不按测试成绩选优。B系列含另一个FusionSF骨干，不归入此纯Chronos-2 Adapter候选；E13混合预测也不作为单个Adapter。E12三seed历史测试未稳定优于E11，当前只检验授权的seed42。结构命名v1固定网络，结构变动递增vN；输入/协议/seed单独记录，不把它们伪装成结构版本。B131_v1是必要第一阶段依赖，按每协议训练一次并供两个版本共用，非新增baseline。

保留原两阶段训练设置：B131 lr0.0016，读出lr0.000016；AdamW betas0.9/0.999、wd0.05、L1、batch512/eval1024，FP32，每阶段max100、patience10、验证MAE选权重，无调度、裁剪、夜间mask或后处理。缓存只是冻结骨干的计算复用，窗口、输入、标签与scaler由Dataset_MMSP提供；不得复用旧raw-NWP缓存。

统一入口：`run.py --model Chronos2Adapter --adapter_version {b131_linear_v1,b131_mlp63_v1}`；`scripts/run_chronos2_adapters.py`仅在一个协议内串行运行已批准的两个版本。记录：`reports/Chronos2Adapter_two_protocols_seed42_20261011/runs.json`及分协议runs.json/日志；权重、缓存和正式预测分别在checkpoints/outputs对应协议的Chronos2Adapter目录。旧Checkpoint的小批缓存forward核对最大差4.77e-7，这是代码迁移检查，不是当前协议完整复现。

当前：四组完整训练、正式测试及产物验收均完成，训练进程已退出。以下默认是当前统一标准化NWP输入下的正式测试Metric，保留六位小数；门控行已按2026-10-16授权替换为正式测试成绩。研究版本独立登记，不覆盖主表的官方Chronos-2或未确定的Ours。

| models | input | MMSP in_domain MAE | MMSP in_domain RMSE | MMSP in_domain R^2 | MMSP zeroshot MAE | MMSP zeroshot RMSE | MMSP zeroshot R^2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A01 (Chronos-2 + B131 + Linear; E11) | 历史功率+历史/未来NWP | 0.044017 | 0.095937 | 0.849073 | 0.044076 | 0.094072 | 0.854885 |
| Chronos-2 + Adapter (b131_mlp63_v1) | 历史功率+历史/未来NWP | 0.044072 | 0.096432 | 0.847512 | 0.044001 | 0.094183 | 0.854542 |
| A02 (A01 + Weather Adapter) | 历史功率+历史/未来NWP | 0.043909 | 0.095870 | 0.849284 | 0.043622 | 0.092942 | 0.858351 |
| A03-a (History-Aware Weather Adapter) | 历史功率+历史/未来NWP（显式历史统计） | — | — | — | 0.043618 | 0.093301 | 0.857255 |
| A03-b (History-Aware Weather Adapter) | 历史功率+历史/未来NWP（显式历史相关性） | — | — | — | 0.043629 | 0.092974 | 0.858251 |
| A03-c (History-Aware Weather Adapter) | 历史功率+历史/未来NWP（学习型历史关系） | — | — | — | 0.043590 | 0.093110 | 0.857836 |
| A04-a (Weather-Conditioned Chronos Readout) | 历史功率+历史/未来NWP（天气Query读取预测Token） | 0.043910 | 0.095813 | 0.849462 | 0.043794 | 0.093361 | 0.857069 |
| A04-b (Weather-Conditioned Chronos Readout) | 历史功率+历史/未来NWP（天气Query读取历史上下文及预测Token） | 0.043962 | 0.096162 | 0.848364 | 0.043710 | 0.093440 | 0.856828 |
| T3 (Native Token Adapter) | 历史功率+历史/未来NWP（全变量Token+Cross-Unet风格Adapter） | 0.043781 | 0.095936 | 0.849077 | 0.043994 | 0.094272 | 0.854268 |
| T2 (Native Token Adapter) | 历史功率+历史/未来NWP（全变量Token+MLP） | 0.043913 | 0.096004 | 0.848863 | 0.043847 | 0.094418 | 0.853814 |
| T1 (Native Token Adapter) | 历史功率+历史/未来NWP（PV未来Token+MLP） | 0.043741 | 0.095720 | 0.849755 | 0.043653 | 0.092837 | 0.858670 |
| 固定门控（正式测试） | 历史功率+历史/未来NWP（全局固定软门控） | 0.043534 | 0.095902 | 0.849182 | 0.043327 | 0.094323 | 0.854109 |
| 动态门控（正式测试） | 历史功率+历史/未来NWP（逐样本逐小时动态软门控） | 0.044847 | 0.100817 | 0.833328 | 0.044264 | 0.095437 | 0.850642 |


门控行已按2026-10-16授权登记正式测试成绩；验证成绩保留在原validation_review.json，不能与测试成绩混用。测试证据为reports/A01_gate_two_protocols_seed42_20261015/{in_domain,zeroshot_v1}/{fixed,dynamic}/test_acceptance.json（仅已完成版本）。

[下载Adapter指标CSV](../reports/chronos2_adapter_metrics.csv)。原始精度与完整验收在`reports/Chronos2Adapter_two_protocols_seed42_20261011/runs.json`的acceptance字段；分协议`reports/{protocol}/results.csv`按model_id区分版本，不能仅按通用model=Chronos2Adapter合并。权重/训练记录在`checkpoints/{protocol}/Chronos2Adapter_{版本}_seed42_sl24_pl24_history_future/`，日志/预测/标签/索引/正式Metric在对应`outputs/`目录；冻结特征是可复现计算证据，保留在outputs对应协议下的Chronos2Adapter_features_v1目录。没有生成新图表或重复报告。

每协议B131第一阶段只训练一次：in_domain完成19轮/最低验证MAE第9轮，zeroshot完成16轮/第6轮。第二阶段in_domain Linear完成50轮/最佳40轮、MLP63完成59轮/最佳49轮；zeroshot Linear完成56轮/最佳46轮、MLP63完成73轮/最佳63轮。全部按最低验证MAE权重正式测试，非按测试指标选择权重。

完整验收：各25,450×24预测/标签有限、各站非全零，全部标签/站点/时间索引与当前协议一致；训练集NWP scaler与协议一致。全部76570/25450/25450冻结缓存目标、站点/时间索引逐项核对；两读出版本的B131权重与同协议第一阶段完全相同。全量指标重算与metrics.json、日志、唯一正式索引一致；6个样本加载保存权重核对GPU预测，CPU最大差4.18e-7，这是权重身份检查，无新增完整推理。现有验收状态为代码/产物检查通过，用户最终验收待确认。

相同输入、相同协议对照冻结Chronos-2：in_domain Linear MAE -0.016892（-27.73%）、RMSE -0.021834（-18.54%）、R² +0.076516（+9.90%）；MLP63分别 -0.016838（-27.64%）、-0.021339（-18.12%）、+0.074955（+9.70%）。zeroshot Linear分别 -0.016976（-27.81%）、-0.023825（-20.21%）、+0.082814（+10.73%）；MLP63分别 -0.017052（-27.93%）、-0.023714（-20.11%）、+0.082472（+10.68%）。两个版本均优于本项目的同输入冻结基线；只做seed42，不能据此声称多种子稳定性。当前标准化NWP与历史raw-NWP不同，不直接比较新旧成绩。

测试描述：in_domain Linear三个指标均略好；zeroshot MLP63的MAE低0.000076，Linear的RMSE/R²更好，不能宣称一个版本全面最好，也不据此改选Ours。zeroshot MLP63首次启动被GPU占用检查拦截，尚未训练；核对卡已空闲后仅重试该组，保留b131_mlp63_v1.startup_failed.log及startup_failure记录，未重跑已完成的其他组。没有NaN/Inf或全零预测异常。四组完成后停止，不自动扩展种子、调参或模型。


## 已验收TimeXer

运行名：`TimeXer_official_seed42_sl24_pl24_power_nwp_full`。

| 协议 | GPU | CPU | 训练PID | 状态 |
| --- | --- | --- | --- | --- |
| in_domain | 3 | 0,7 | 3313923（已退出） | 训练/测试与完整产物验收完成，7轮/最佳4轮 |
| zeroshot_v1 | 0 | 9,10 | 3313957（已退出） | 训练/测试与完整产物验收完成，7轮/最佳4轮 |

官方MS分支：dim/FF512、2层/8头、factor3、dropout0.1、patch12适配24小时、use_norm=True；实际输入为历史功率+15维历史NWP，不消费未来NWP。MMSP24→24和seed42为统一任务设定。batch/eval32、FP32、Adam lr0.0001/betas0.9/0.999、MSE、官方type1调度、最多10轮、验证MSE选权重、patience3、训练drop_last=False，验证/测试全部窗口。

- 本轮配置、训练日志和验收记录：`reports/TimeXer_two_protocols_seed42_20261011/`；唯一状态记录为runs.json。训练及旧检查进程已完成，监控快照仅留本机。

2026-10-11完整产物验收通过：各25,450×24标签与原始CSV一致，完整站点/时间索引、协议及训练集NWP scaler一致；统一全量Metric重算与保存指标/唯一索引相同；两协议均按最低验证MSE选择第4轮权重，第7轮早停。5个跨站样本加载Checkpoint核对保存预测，CPU/GPU最大差2.39e-7；这是权重身份检查，完整结果来自原正式测试，无重训或完整重推理。预测与权重有限、各站非全零，负预测保留原口径。

同协议对照Cross-Unet：in_domain MAE +0.029164（+67.23%）、RMSE +0.035824（+42.46%）、R² -0.120176（-13.61%）；zeroshot_v1 MAE +0.026839（+60.16%）、RMSE +0.034141（+40.57%）、R² -0.113350（-12.82%）；TimeXer不使用未来NWP，不能将差异单独归因于架构。唯一验收依据为本轮runs.json的acceptance字段，主表已同步六位小数。
- 权重及训练配置/协议/scaler：`checkpoints/{protocol}/{运行名}/`。
- 日志及正式测试后配置/协议/scaler、预测、标签、站点/时间索引和指标：`outputs/{protocol}/{运行名}/`。
- 正式测试索引：`reports/{protocol}/results.csv`。启动、首批或首轮通过不等于完整性能复现。

## 已验收Cross-Unet

进一步直接核查相邻PV-power目录及其原始bfd6bd5提交、GitHub main：Decoder每层均将corr传给self-attention，矩阵乘法进入残差后再到cross-attention/Linear预测；官方Adam包含全部模型参数，预测loss正常backward/step。两个源码版本分别在CPU训练模式做有界检查，固定并detach Encoder输出、匹配dropout随机数，仅替换Decoder矩阵仍改变预测；corr梯度L1=1.035978，四个P-Corr投影权重梯度均非零，Encoder没有梯度。证据追加到既有pcorr_usage_check_20261012.json的PV_power_official_decoder_check。本次是源码与合成小批计算路径核验，不是新增训练或性能消融；未修改PV-power目录、模型、权重或Metric，临时四源码片段/缓存已删除。

2026-10-12按用户要求核查P-Corr实际使用：两个协议的已训练权重各取4个真实测试样本，hook确认同一[B,16,16]矩阵进入3个Encoder和4个Decoder层，执行corr×通道表征并加入残差；Bottleneck自身不直接使用corr。诊断列表的detach不切断计算路径，四个P-Corr投影权重均有有限非零梯度，原Checkpoint的Adam步数和非零二阶动量确认其实际参与训练。zero-shot将矩阵置零/跨样本交换时，预测最大变化分别0.527921/0.010840；原矩阵恢复后输出逐位一致。小批CPU重放与已保存预测最大差≤1.23e-6，未重训或全量复评，模型及Checkpoint未改。证据：[pcorr_usage_check_20261012.json](../reports/CrossUnet_two_protocols_seed42_20261010/pcorr_usage_check_20261012.json)，入口记录在同目录runs.json的pcorr_usage_audit_20261012。

更正A03先前审查中“softmax计算了却返回归一化前矩阵”的错误判断：本地版及固定官方bfd6bd5版本均返回展平256项softmax后的矩阵，总和为1，并非逐行softmax。A03的review.json已保留更正记录；A03实现和已完成的实验结果未改。此检查证明P-Corr确实被使用，不等于证明它带来性能增益或因果解释。

运行名：`CrossUnet_official_seed42_sl24_pl24_power_nwp_full`；seed42、24→24、官方核心结构。batch128、Adam lr0.0001、type3、最多100轮、验证MSE patience10；详细架构/实际输入/训练配置保存于对应config.json与runs.json。

训练/权重：in_domain完成42轮、最优32轮；zeroshot_v1完成43轮、最优33轮。正式测试指标见上方总表。

约3小时8分钟完成，两协议均在最优后10轮无改善早停。2026-10-11验收通过：各25,450×24原始标签、站点/时间索引与协议逐项一致；预测/权重有限，各站非全零；独立重算MAE/RMSE/全局R²与保存指标、日志、唯一结果索引一致；权重对应最低验证MSE，保存配置/协议/scaler匹配。未重训或重新推理。

同协议对照FusionSF：in_domain MAE+0.002501610(+6.12%)、RMSE−0.006890709(−7.55%)、R²+0.019846256(+2.30%)；zeroshot_v1 MAE+0.003013659(+7.24%)、RMSE−0.005659215(−6.30%)、R²+0.016145861(+1.86%)。测试标签与索引完全一致；实际输入不同，不能隔离架构贡献，不跨协议排名。Cross-Unet输出存在负值，沿用冻结的原始预测评价口径，无裁剪；比例与范围记于验收JSON。

唯一机器验收记录：`reports/CrossUnet_two_protocols_seed42_20261010/runs.json`的acceptance字段；对应启动配置与training.log在同目录。权重/协议/scaler及完整预测/标签/索引/指标保留在checkpoints/outputs对应协议和运行名下；没有新图表或重复Markdown报告。

## 已验收FusionSF

运行名：`FusionSF_official_script3_guide17_seed42_sl24_pl24_all_full`；seed42、24→24、官方三模态guide17；batch16、FP32、AdamW lr0.0016/wd0.05/betas0.9/0.95、L1+VQ、cosine+5轮warmup、最多100轮、验证MAE patience20。完整配置保留在对应config.json。

训练/权重：in_domain完成30轮、最优10轮；zeroshot_v1完成29轮、最优9轮。正式测试指标见上方总表。

2026-10-10按用户要求验收通过：两协议各25,450×24标签与原始CSV逐元素一致、完整站点/时间索引对应、配置/协议/scaler匹配、预测与权重有限、各站预测非全零；独立重算全局MAE/RMSE/R²与保存文件、日志和唯一索引行一致。两个协议均在验证最优后20轮无改善早停。第2–6轮曾匹配全零验证参考、第7轮恢复；原因未确认。只做seed42，无同配置已验收旧基线，绝对/相对改进不可计算，不跨协议排名。

唯一验收说明：`reports/FusionSF_two_protocols_seed42_20261010/acceptance.md`；机器证据在同目录runs.json的acceptance字段，配置及日志亦在该目录。权重和输出按上述运行名保留在checkpoints/outputs对应协议下；没有新图表，没有验收重训或重新推理。

## 已验收Chronos-2历史结果复用

用户2026-10-11批准主表采用history_future，同时维护none/history/future/history_future四输入指标表。模型使用本项目pretrained/chronos-2，权重冻结，无本地训练Checkpoint；实际为24小时功率、24小时历史及未来15维NWP，无卫星。原始完整推理在BasicTS完成，batch64、seed42、中位数0.5、cross_learning=False，不裁剪预测；旧cfg中的optimizer/epochs等继承字段未参与训练。

两协议四输入共8组各25,450窗口验收通过：全部功率输入、目标、站点与时间索引符合当前数据协议；NWP特征顺序、训练站点/时段scaler一致；数据文件和预训练权重与历史路径逐字节一致；预测有限，按统一utils/metrics.py重算MAE/RMSE/全局R²与原始metrics.json及完整推理日志一致，指标变化为0。当前模型入口的协变量构造及quantile/API设置与保存配置及快照适配器核对一致。

唯一记录为`reports/Chronos2_two_protocols_seed42_20261011/runs.json`的acceptance字段；原始配置、协议/scaler、metadata、完整日志、inputs/prediction/targets/site_ids/forecast_timestamps与metrics保留在`outputs/{protocol}/legacy_20261010/Chronos-2/{输入版本}/{原运行ID}/`，不重复复制、不改原始文件。正式分协议索引仅登记history_future；四输入细表引用相同证据。此次没有通过TSLib入口重新推理，不计新增实验或seed。

纯功率版两协议输入相同且模型不训练，因此指标相同；含NWP版本分别使用各协议训练站点scaler。未来NWP仍按现有约定假定预测起点可用，CSV缺少发布时间。

## 已验收DLinear/PatchTST旧版本复用

用户2026-10-11明确接受官方版本差异并批准填表。两个模型均为FusionSF仓库实现、历史功率+经纬度三通道接口，seed42、24→24；主表不标为原作者官方复现。DLinear使用moving_avg13、共享线性权重初始化、AdamW lr0.0016/batch64；PatchTST使用LayerNorm版编码器、dim512/FF2048、3层/8头、dropout0.05、AdamW lr0.0002/batch16。两者均使用旧MSE训练、验证MAE选权重、ReduceLROnPlateau调度、最多50轮；DLinear完成19轮、最佳9轮，PatchTST完成50轮、最佳34轮。

in_domain各25,450窗口通过完整产物验收：全部保存功率输入/标签、站点/时间索引与当前原始数据/协议一致；预测及权重有限；统一MAE/RMSE/全局R²重算与metrics.json和完整训练日志一致，迁移前后变化为0；日志确认正式评估最低验证MAE权重。唯一验收记录：`reports/DLinear_PatchTST_migration_seed42_20261011/runs.json`，包含原配置、协议、训练日志、最佳Checkpoint及完整预测路径。原始证据仍在`checkpoints/in_domain/legacy_20261010/{模型}/power_coordinates/{原运行ID}/`，没有复制或改写，没有新增训练或推理。

zeroshot确已在BasicTS完成，历史登记R0004/R0005保留完整精度Metric、配置及协议记录；原Checkpoint、日志和预测等随此前内部BasicTS归档清理被删除，目前未找到完整副本。用户本次明确批准将这两组成绩填入主表，状态为历史登记成绩、原产物缺失、未完成当前完整验收；分协议正式测试索引不新增这两项。填表与历史原值的变化为0（展示四舍五入至六位小数），没有新增训练或推理。原作者官方入口和legacy_fusionsf入口继续区分，未改框架、公共接口或数据协议。

## 待办与验收

Site1探索：窗口与归一化核对通过，协议代码无需修改。旧lr0.0016完成4轮、lr0.00016完成3轮，均在5轮warmup内被中止；十站点同结构实验第2–6轮也出现全零验证平台，第7轮恢复，不能据此保证Site1会恢复。Site1后续是否进行保留官方结构的训练诊断/充分训练，须先明确任务范围；不直接替换ReLU、不盲目再降学习率。核对证据保存在已有Site1重跑summary.json，测试MAE/RMSE/R²仍缺失。

1. TimeXer两个协议已完成完整产物验收并登记主表；Adapter四组也已完成训练、正式测试与产物验收，独立指标表已同步。
2. Time-VLM两协议已完成训练、正式测试与完整产物验收并填表；Adapter用户最终验收状态保持待确认，不自动重跑。Time-VLM完成后停止，其他模型、Site1、多种子与调参需用户明确布置。
3. 比较保持同协议，输入不同必须注明；正式实验缺失留空，历史原产物缺失例外继续标注，不自动补跑。

## 清理与历史资料

用户明确要求删除内部BasicTS归档，接受相关原始材料丢失，后续需要时重新实验。本次已删除archives、无运行依赖的vendor/BasicTS、构建缓存、协议/接口验收临时产物、重复过程报告和已移除Time-LLM的GPT-2权重；完整删除清单/统计记录于本轮runs.json的cleanup字段，没有另外新建报告。

此前清理未修改框架/公共接口/数据协议、未启动实验、未删除当时两条FusionSF训练的必要证据。原始MMSP数据和仍被模型使用的预训练权重保留；后续实验仅按明确授权启动；TimeXer及Adapter均已完成，Time-VLM按2026-10-11最新任务启动。

现有legacy正式结果、对应权重/预测/必要日志与Site1失败实验保留；无法确认无用的科研证据不删除。历史指标登记保留单一CSV，不保留重复Markdown汇总；删除的内部归档/验收产物标为缺失。历史成绩默认不进入新正式结果索引；用户2026-10-11明确批准、已核对完整证据的Chronos-2及DLinear/PatchTST旧版本复用单独记录来源和版本，不计作新的独立推理、训练或seed。独立FusionSF目录继续保留历史探索。

项目转发需包含数据、预训练权重及必要实验产物；它们不随Git同步。当前清理未提交或push。

## FusionSF三种输入实验

2026-10-12最新授权：A02两协议完成并验收后在GPU4继续以下六组，取消99%停止条件，允许通过batch调整优化真实吞吐；每种先in_domain再zeroshot_v1，seed42、24→24。当前队列状态：`completed_artifact_acceptance_passed`。固定GPU4、CPU0,7两线程、num_workers0。

用户随后明确批准“修复重跑”：首组旧尝试epoch23结束，25450×24正式预测全零，验收失败；旧Metric/预测/权重/配置/日志及诊断保留在原目录及runs.json的failed_attempts，未进入正式结果表。功率+未来NWP两协议采用readout_repair_v1独立目录，从seed42/epoch1重新训练，不复用退化权重或优化器。

修复仅改变读出初始化和读出学习率：原Linear权重保留随机方向，行L2范数缩至0.05/sqrt(128)，bias0.1，使初始输出在[0.05,0.15]；主干AdamW lr0.0016，读出组lr0.000016（主干1/100）。结构/ReLU/VQ/L1+VQ、FP32、max100/patience20、warmup5-cosine100和协议保持。首种输入两协议已正式验收；用户随后批准停止含卫星退化组、保留产物，将同一修复扩展到剩余两种输入的两个协议，经真实批次功率监督梯度检查后从epoch1重跑。短检查不是正式训练或性能复现；修复配置不能称官方默认配置或单纯输入消融。

必要检查：`reports/FusionSF_inputs_two_protocols_seed42_20261011/positive_readout_preflight_20261012.json`；仅正初始化而不降低读出学习率的失败诊断日志保留。

剩余两种输入检查：`reports/FusionSF_inputs_two_protocols_seed42_20261011/positive_readout_remaining_preflight_20261012.json`；停止前含卫星最佳epoch2权重的六样本ReLU前负值及零监督梯度证据保留，未进行该旧尝试的全量正式测试。

- `power_future_nwp`：batch256/eval64。
- `power_satellite_history_future_nwp`：batch64/eval64。
- `power_history_future_nwp`：batch256/eval64。

真实吞吐证据：`reports/FusionSF_inputs_two_protocols_seed42_20261011/batch_throughput_20261012.json`；训练期GPU采样含验证阶段，每30秒一次，不将峰值称为持续满载。

历史NWP与对应时刻功率拼成16通道，未来NWP保留原17通道引导（15天气+经纬度）；无卫星版本去除卫星分支及context VQ。保留当前script配置、ReLU输出、TS VQ、L1+VQ、AdamW lr0.0016、warmup5/cosine100、max100/patience20、验证MAE选权重。新输入属于研究变体，独立填表，不替换原FusionSF主表或复用历史0.040592。训练/测试/验收失败立即停止。

| models | input | MMSP in_domain MAE | MMSP in_domain RMSE | MMSP in_domain R^2 | MMSP zeroshot MAE | MMSP zeroshot RMSE | MMSP zeroshot R^2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FusionSF (power_future_nwp; readout_repair_v1) | 历史功率+未来NWP+坐标/时间编码 | 0.040904 | 0.091621 | 0.862347 | 0.040731 | 0.085422 | 0.880344 |
| FusionSF (power_satellite_history_future_nwp; readout_repair_v1) | 历史功率+历史卫星+历史/未来NWP+坐标/时间编码 | 0.040414 | 0.088262 | 0.872255 | 0.041471 | 0.089704 | 0.868047 |
| FusionSF (power_history_future_nwp; readout_repair_v1) | 历史功率+历史/未来NWP+坐标/时间编码 | 0.042900 | 0.096197 | 0.848253 | 0.040797 | 0.089615 | 0.868309 |

[下载三种输入CSV](../reports/fusionsf_input_metrics.csv)。脚本：`scripts/run_fusionsf_inputs.py`；唯一运行/验收记录：`reports/FusionSF_inputs_two_protocols_seed42_20261011/runs.json`。权重和预测按各run的checkpoint_dir/output_dir保存，修复尝试独立命名；正式索引及独立表仅填已验收指标。全部完成后停止，无后续模型或新种子。

### 2026-10-13 验收与结果分析

六组修复实验全部通过保存产物复核：独立重算每组25,450×24测试指标，全部标签逐元素匹配原始功率CSV，站点/时间与原FusionSF一致；配置、协议、实际使用模态的训练scaler、有限权重/非退化预测、验证MAE最低权重、早停、日志及唯一正式索引均一致。无卫星版本只有NWP scaler，卫星scaler不适用。复用原GPU权重重放证据（六组最大差均0），本次未训练或推理。两张独立结果CSV及文档展示已逐值核对，原主表不替换；机器复核写入原runs.json的acceptance_review_20261013。

下表对照原已验收三模态FusionSF：in_domain MAE/RMSE/R²=0.040875803/0.091264941/0.863415558；zeroshot_v1=0.041598372/0.089822095/0.867700063。Δ=当前−基线，误差指标负值为改善；相对变化=Δ/基线×100%。

| 协议 / 版本 | 最佳轮 / 完成轮 | ΔMAE（相对变化） | ΔRMSE（相对变化） | ΔR² |
| --- | --- | --- | --- | --- |
| in_domain / power_future_nwp | 10 / 30 | +0.000028120（+0.0688%） | +0.000356347（+0.3905%） | -0.001068678 |
| zeroshot_v1 / power_future_nwp | 6 / 26 | -0.000866899（-2.0840%） | -0.004399974（-4.8985%） | +0.012644076 |
| in_domain / power_satellite_history_future_nwp | 3 / 23 | -0.000462143（-1.1306%） | -0.003002789（-3.2902%） | +0.008839915 |
| zeroshot_v1 / power_satellite_history_future_nwp | 35 / 55 | -0.000127193（-0.3058%） | -0.000117941（-0.1313%） | +0.000347204 |
| in_domain / power_history_future_nwp | 10 / 30 | +0.002023703（+4.9509%） | +0.004932488（+5.4046%） | -0.015162590 |
| zeroshot_v1 / power_history_future_nwp | 8 / 28 | -0.000801215（-1.9261%） | -0.000206800（-0.2302%） | +0.000608495 |

in_domain三项指标最佳均为power_satellite_history_future_nwp（batch64、最佳epoch3）：0.040414/0.088262/0.872255；跨站三项最佳均为power_future_nwp（batch256、最佳epoch6）：0.040731/0.085422/0.880344。该“最佳”仅是本次测试结果描述，不据此更改预先登记版本或继续调参。无卫星的同batch256比较中，加入历史NWP使域内MAE/RMSE和跨站MAE/RMSE均变差，当前未显示收益；卫星+历史NWP组合的域内优势没有转为跨站优势。不同输入涉及分支/VQ和batch差异，与原基线还涉及初始化及读出学习率变化，不能将改善单独归因于模态或架构，也不能称官方默认配置的输入因果消融。

两次旧失败仍保留：首组全量预测全零/验收失败；旧含卫星组停止且没有全量正式测试。修复六组没有缺失正式测试或全站全零退化。仅seed42，未验证稳定性；未来NWP发布时间缺失和双向插值的在线可用性限制保持。六组日志路径、checkpoint_dir/output_dir和失败产物均见原runs.json；本次未新增图表。结论：修复后有可信的完整结果，更多输入未带来一致收益；下一步最小行动是用户审阅上述比较，当前不启动新实验。

## TimeVLM续训与ChronosX队列

最新授权：TimeVLM总上限200轮/patience20续训；两个协议完成并验收后，在同一GPU0串行复现ChronosX两个协议。固定CPU9,10两线程、num_workers0；GPU3/CPU0,7继续原FusionSF六组队列100/20。其他已有实验与结果不改。

TimeVLM从保存的第9轮最佳模型/记忆/优化器/type1调度恢复，epoch10开始；恢复lr3.90625e-6，每轮减半。旧第10轮末状态、随机数和AMP scaler缺失，续训使用新随机数/AMP状态，不能称完整轨迹恢复。原10轮指标/权重/预测/日志保留，主表暂展示原已验收结果，续训验收通过再更新。预算改为200/20，架构及其他训练参数保留，baseline_config=extended_budget。

<!-- TIMEVLM_CONTINUATION_STATUS -->

2026-10-11 06:33续训首轮epoch10完成：in_domain验证MSE0.013576081914034432，低于源epoch9的0.013577664509545995；验证MAE0.06659789648635328/RMSE0.11651644482232726。仅为验证结果，完整训练和正式测试尚未完成，主表未更新；zeroshot_v1及ChronosX等待串行执行。

ChronosX固定Chronos-T5-small及IIB+OIB，冻结主干，训练2127616个注入层参数（共48281856参数）；输入历史功率+历史/未来NWP及缺失标记。官方CE/AdamW lr0.01/wd0、FP32、batch32/eval8、accumulation2、clip1、20采样中位数保留。按用户预算改为完整epoch验证CE、max200/patience20及按239400步线性调度；官方原预算5000步并未采用。EOS对齐、冻结参数、真实小批前后向与采样预测已核对；检查不等于正式性能复现。

2026-10-12用户重新授权单独复现ChronosX，允许适当调参；不恢复TimeVLM/FusionSF队列。已完成真实训练性能比较：默认batch32/FP32/accumulation2为747.93窗口/秒、平均利用率29.71%；候选batch1024/BF16/accumulation1为3106.18窗口/秒、平均79.64%/峰值100%、峰值分配显存8358MiB。候选有效batch从64变为1024，优化轨迹不等价；未进行正式训练/测试，不能称性能复现完成。15秒短时证据为`reports/ChronosX_two_protocols_seed42_20261011/gpu_profile_20261012.json`。随后用户明确批准运行，等待任意完全空闲GPU后自动启动；本次按吞吐优化，平均利用率低于99%也继续。独立入口`scripts/run_chronosx.py`已加入等待逻辑，在无计算进程、显存<=1024MiB、利用率0%的GPU出现后进行BF16 GPU检查、两个协议串行训练/测试和完整产物验收。协调日志`reports/ChronosX_two_protocols_seed42_20261011/coordinator_20261012.log`，等待与实际运行状态保存在原runs.json；每个GPU阶段开始前再次检查占用。ChronosX混合精度训练/验证已接入，验收步数按实际batch/accumulation计算，数据协议及其他模型不改。

2026-10-12 13:40等待器发现GPU4完全空闲，GPU BF16真实batch1024前后向、验证和64×20采样预测检查通过后启动in_domain（训练PID303159，协调器PID302963）；zeroshot_v1等待串行。首轮于13:41:19完成：训练loss2.974764、验证token CE2.254133（仅验证，非测试Metric），训练25.180秒、验证12.275秒、真实训练吞吐3040.95窗口/秒。实际GPU利用率采样包含训练/验证，峰值100%，详见in_domain_gpu_utilization.csv，不将峰值称为持续满载。每协议正式测试后保存完整预测/标签/索引与指标，两协议完整产物验收通过才更新总表。

<!-- CHRONOSX_STATUS -->

ChronosX两协议已完成100/10预算下的正式训练、测试和产物验收，主表已同步。

in_domain完成16轮，选模依据minimum validation rmse，最佳第6轮；MAE/RMSE/R²=0.047528/0.108813/0.805843。

zeroshot_v1完成16轮，选模依据minimum validation rmse，最佳第6轮；MAE/RMSE/R²=0.048782/0.107841/0.809296。

唯一整体验收证据：`reports/Baseline_alignment_100_10_seed42_20261014/ChronosX/runs.json`；20次采样中位数、仅seed42，未验证多种子稳定性。此队列完成后停止，无后续实验。

<!-- /CHRONOSX_STATUS -->

脚本：`scripts/run_timevlm_chronosx.py`；队列状态/日志：`reports/TimeVLM_ChronosX_queue_20261011/`。模型配置、吞吐检查、逐组日志与唯一验收分别为`reports/TimeVLM_resume200_pat20_seed42_20261011/runs.json`、`reports/ChronosX_two_protocols_seed42_20261011/runs.json`；对应模型名下checkpoints/outputs保存权重、协议/scaler、预测/标签/索引/Metric。ChronosX保存预测起点CPU/CUDA随机数，用于首批采样预测重放身份核对。全量Metric无裁剪/夜间mask。任一阶段失败停止，不自动重跑或推进其他模型。


### 两GPU耗时估计

2026-10-11 06:29估计：GPU0的TimeVLM续训约5.2分钟/轮，ChronosX约6.3分钟/轮，ChronosX每协议完整测试约15分钟；假设TimeVLM每协议续训20–40轮、ChronosX每协议30–50轮，GPU0预计10月11日17:00至12日01:00结束，两个模型都跑满200上限则约10月14日上午。GPU3现有无卫星FusionSF约9分钟/轮，卫星版本参考此前约18.4分钟/轮；六组均30–50轮时预计10月12日晚至13日晚结束，全部跑满100则约10月16日上午。早停时点未定、共享硬件速度会变，此为规划范围，不是完成承诺。精确估算、依据与假设：`reports/TimeVLM_ChronosX_queue_20261011/duration_estimate.json`。

用户要求检查GPU3低利用率。06:33采样GPU3为9%/701MiB/64.73W；训练CPU平均100%，主线程74.8%、autograd线程25.2%，磁盘读写0。CPU0单线程、num_workers0独立读取32个batch16（512样本）的cProfile用时1.575秒，NWP get_group调用1024次、累计1.076秒（约68%本次数据读取计时）；storage和Dataset_MMSP重复提取同一NWP分组。证据支持CPU侧数据准备及计算调度受限，但该数据读取剖析不能量化真实训练各阶段占比。可优化点是缓存坐标对应NWP数组；本次仅诊断，未改加载代码、训练参数或进程。证据：`reports/FusionSF_inputs_two_protocols_seed42_20261011/gpu3_utilization_check.json`。

### 暂停、等价加速与恢复（用户随后授权）

两条队列及训练曾暂停，再从保存的最佳checkpoint恢复以载入新代码：TimeVLM epoch10→11，FusionSF epoch2→3；未保存的后续模型/优化器/RNG/迭代器状态无法完整恢复，FusionSF已完成的旧epoch3和两项未完成轮次需重算。原权重保存为各checkpoint目录下`before_input_optimization.pth`，原日志备份为逐组及outputs目录下`*.before_input_optimization.log`；分支来源在原runs.json的interruption。暂停前队列安排为GPU0 TimeVLM两协议→ChronosX两协议、GPU3 FusionSF六组；CPU/num_workers/batch/精度/预算/协议不变。用户随后再次停止两队列，见下文最新决定。

公共NWP标准化数组按坐标缓存，消除逐样本两次pandas分组提取；功率+NWP复用底层已切片的张量。TimeVLM直接返回官方77字符固定前缀并缓存token IDs；图像整批下传，保留原PIL bicubic/crop，上传uint8后通过枚举原NumPy归一化的256值查表得到逐位相同的float32像素；记忆更新仍按官方顺序和回绕行为执行，仅把多次指针读取合并一次。严格加载已有模型state_dict无新增权重键。

两协议power_nwp的train/val/test及in_domain含卫星train/test各抽查40个跨站点/时间样本，四序列和全部covariate逐位相同。TimeVLM batch1/10/32的文本/像素/token IDs一致；真实batch32、FP16、相同权重/RNG的预测、loss、全部可训练梯度和记忆更新逐位相同，最大差0。64批数据独立计时：in_domain无卫星40.79→3.55ms/批（11.48倍），zeroshot39.02→3.61ms（10.81倍），含卫星43.36→8.52ms（5.09倍）；TimeVLM固定真实批次前后向中位数83.03→44.92ms（1.85倍、耗时减少45.90%）。这些不是完整轮次或正式测试成绩，尚无新MAE/RMSE/R²；旧耗时估计已标记需更新。唯一核验证据：`reports/TimeVLM_ChronosX_queue_20261011/input_optimization_check.json`。修改文件为data_provider/mmsp_storage.py、data_loader.py、layers/timevlm/model.py、vlm_manager.py，以及必要的run.py/FusionSF恢复入口和两队列断点启动逻辑。

### FusionSF原版对照与99%停止条件

用户指出原baseline利用率正常，要求认真核对；随后明确“检查解决完之后继续训练，训练的时候要监控GPU利用率，如果无法达到99%，就停止不做了”。因此保留两正式队列暂停，在释放的同一GPU0上按原CPU0,7两线程、batch16/FP32/num_workers0、seed42、24→24分别加载原版三模态最佳epoch10与当前功率+未来NWP最佳epoch2，进行3个真实步骤的CPU/CUDA profiler及各25秒含数据读取/优化器更新的连续训练步骤计时；仅诊断模型副本，未更新正式权重、未开展完整训练或正式测试。

每秒一次的nvidia-smi采样排除前两次启动采样后，原版三模态均值57.65%、范围21–93%；当前无卫星均值13.22%、范围11–14%，差-44.43个百分点、相对-77.07%。这是本次同条件复测，历史监控记录未存GPU利用率，不能冒称此前正式训练的历史利用率。原版约5211 CUDA kernel/步、profile GPU活跃84.21ms/步；无卫星约2288 kernel/步、GPU活跃10.92ms/步，GPU活跃时间减少73.29ms（87.03%）。无卫星独立连续步骤中位数70.60ms，数据读取4.63ms（约6.56%）；大量小算子的CPU调度及kernel launch为剩余热点。VQ存在标量/同步，但本次无卫星profile中cudaStreamSynchronize只有约0.14ms/步，并非最大的剩余CPU热点。移除卫星VQ/encoder/cross-attention后计算量大降，不能用原三模态的利用率直接要求该输入变体达到同水平。

两个版本均无采样达到99%，按用户停止条件于06:52结束剩余暂停的FusionSF训练及协调器；TimeVLM及其协调器此前已停止，ChronosX/其余FusionSF组保持未启动。不擅自改变batch、CPU限制、精度、输入或架构，不用无效计算制造满载；不自动恢复。当前最佳已保存权重仍TimeVLM epoch10、FusionSF epoch2；已有完整原版结果保持不变，新测试MAE/RMSE/R²缺失。模型/日志保存在两runs.json记录的checkpoints/outputs/reports路径；未生成图表。剖析与停止证据：`reports/FusionSF_inputs_two_protocols_seed42_20261011/original_vs_variant_gpu_profile.json`，原始采样为同目录`gpu_utilization_original_three_modal.csv`与`gpu_utilization_power_future_nwp.csv`；两队列runs/status已标记stopped_by_user_gpu_utilization_requirement。当前任务结束，保持停止，等待用户重新明确决定。

## A01/A02 Weather Adapter（2026-10-12）

用户批准命名A01/A02并编写代码、开展实验。A01映射当前已验收的b131_linear_v1/E11，复用本项目同协议最佳权重和正式结果，不重新训练；E12原编号和记录保留。A02冻结Chronos-2、B131及Linear读出，仅训练新Weather Adapter，未指定为Ours。

A02将2×768适配后表征按16/8小时对齐，与逐小时未来15维NWP及缺失标记各投影32维，拼接→ReLU→MLP(64,32,1)，功率残差加到反归一化后的A01预测；末层零初始化，共29249个新参数。借鉴ChronosX OIB的融合方式，输出为功率残差而非token logits。骨干原有历史/未来NWP输入和归一化不改。

沿用Adapter读出预算：in_domain→zeroshot_v1串行，seed42、24→24、FP32、batch512/eval1024、AdamW lr0.000016/wd0.05/betas0.9,0.999、L1、max100/patience10、无调度，按验证MAE选权重（包括epoch0零残差）。单空闲GPU、CPU9,10两线程、num_workers0；失败停止，不自动重跑、调参或扩展消融。

既有冻结特征保留，未来NWP从Dataset_MMSP读取，全部标签/站点/时间索引及训练scaler对齐后缓存。草稿CPU原生编码存在近零方差数值差异，保留诊断；GPU原生/缓存预测一致性、冻结权重、零初始化及梯度检查通过才运行，未改Chronos-2数学结构。

当前状态：`completed_artifact_acceptance_passed`；各协议：in_domain=completed_artifact_acceptance_passed；zeroshot_v1=completed_artifact_acceptance_passed。

唯一入口`scripts/run_chronos2_weather_adapter.py`；唯一运行与验收记录`reports/Chronos2WeatherAdapter_two_protocols_seed42_20261012/runs.json`，GPU检查`preflight_gpu.json`，原CPU诊断保留。正式验收后更新既有Adapter独立表，不覆盖官方Chronos-2主表。A02对A01的比较不能单独证明天气帮助利用表征；仅天气/仅表征/联合的贡献对照未启动，需另行授权。

## A03 History-Aware Weather Adapter（2026-10-12）

用户批准保留A02协议并记录限制，继续三个独立A03版本：a为历史均值/标准差/有效比例，b为成对有效的有符号Pearson/有效比例/低方差有效性，c为逐小时小MLP对历史功率、天气、乘积、有效标记与相对位置编码后池化。三者均以32维关系向量的1+tanh门条件化A02未来天气编码，不串联，不复制Cross-Unet。

仅zeroshot_v1、seed42、24→24；复用冻结A01，Weather Adapter从seed42重新初始化并与新分支共同训练，max100/patience10、batch512/eval1024、FP32、AdamW lr0.000016/wd0.05、L1、验证MAE选权重含epoch0。初始输出等于同样初始化的A02/A01，不从已拟合A02继续增加训练预算。a/c新增2624、共31873参数，b新增2528、共31777参数，三者总可训练参数相差0.31%，比A02多约9%；未开展参数匹配控制，不能将改善归因于历史关系机制。

保留既有分窗前NWP双向插值与未来NWP发布时间缺失限制。原CSV共29项缺失，站点0–19涉及23项；全网格与仅前缀插值在9项原缺失值上不同。新增分支只使用起点前历史，检查不等同于证明数据在线可用。审查/协议核对见[review.json](../reports/Chronos2HistoryWeatherAdapter_zeroshot_seed42_20261012/review.json)，GPU形状、有限梯度、冻结哈希、初始一致性、原生/缓存、时间边界及窗口/站点/scaler核对见[preflight_gpu.json](../reports/Chronos2HistoryWeatherAdapter_zeroshot_seed42_20261012/preflight_gpu.json)。

当前状态：`completed_artifact_acceptance_passed`；A03-a=completed_artifact_acceptance_passed；A03-b=completed_artifact_acceptance_passed；A03-c=completed_artifact_acceptance_passed。

唯一入口scripts/run_chronos2_history_weather_adapter.py，唯一记录reports/Chronos2HistoryWeatherAdapter_zeroshot_seed42_20261012/runs.json；每组正式测试和验收后才登记既有Adapter独立表，不改A01/A02、官方主表或其他队列。逐horizon、各自最差10%及A02共同最差10%窗口、按训练NWP变动强度三分位分层与相对A02改善保存于各输出analysis.json；仅为单seed探索证据。42/2026/2027稳定性与参数匹配实验本次不自动启动，三组完成后停止。

三组已完成正式测试与完整产物验收并登记独立表：a完成54轮/最佳44，b完成70轮/最佳60，c完成54轮/最佳44。相对A02，a的MAE改善0.008971%、RMSE变差0.386197%；b的MAE/RMSE变差0.016672%/0.035092%；c的MAE改善0.071229%、RMSE变差0.181670%。R²均低于A02；不能证明显式历史关系改善zero-shot泛化，相关性比统计分支的RMSE较低和学习型分支的微小MAE优势仍不足以证明机制或必要性。

三者低变化组改善，高变化组MAE/RMSE均变差；各自最差10%窗口的RMSE均变差，固定A02最差10%窗口中b略好而a/c变差。逐小时和各分层相对变化已写入各analysis.json，汇总在runs.json的evaluation。未确认值得推进多种子或容量控制，本阶段停止；不能将负向/极小的单seed差异解释为机制已经无效。

保留首轮检查断言修正及GPU交接失败证据；仅修正协调器稳定空闲等待，a/b未重训，c在加载模型前的失败不计训练。A01/A02源码、checkpoint及原始预测/Metric未改，必要配置、日志、权重、预测和独立输入缓存保留；临时审查脚本及临时scaler核对目录已清理。

## A04 Weather-Conditioned Chronos Readout（2026-10-12）

用户确认首轮方案并要求in_domain也做：两个协议、seed42、24→24，单GPU串行四组。A04-a天气Query读取B131后的两个预测Token；A04-b按原encode提取两个功率历史上下文Token，与两个适配后预测Token组成Memory，排除REG，不增加预测patch或上下文长度，不将B131用于历史Token。历史位置状态经双向时间/分组注意力，已包含已知未来NWP，不能称纯历史观测编码。

未来NWP值及缺失标记[ B,24,30 ]经A02 cov_out得到32维天气表示，加固定正弦小时位置构成Query。共享A02 LayerNorm/Linear投影Memory为K=V，单头32维，softmax(QKᵀ/√32)读取[ B,24,32 ]，与原天气表示拼接后走原残差MLP；最终Y_A01+Delta。两版均29249可训练参数，与A02完全相同；A01全部冻结。两个预测Token的注意力仅为两向量之间的动态混合，不能解释为24个独立Chronos小时状态。

Weather Readout复用seed42的A02初始权重与随机数流，从头训练，不热启动拟合A02。FP32、batch512/eval1024、AdamW lr0.000016/wd0.05、L1、max100/patience10，验证MAE选权重含epoch0。沿用原始站点、窗口、scaler及A02 NWP双向插值/发布时间缺失限制，不更改公共接口、A01/A02实现或checkpoint，不依据测试成绩调参。

当前状态：`completed_artifact_acceptance_passed`；GPU5，CPU9,10两线程、num_workers0；in_domain/A04-a=completed_artifact_acceptance_passed；in_domain/A04-b=completed_artifact_acceptance_passed；zeroshot_v1/A04-a=completed_artifact_acceptance_passed；zeroshot_v1/A04-b=completed_artifact_acceptance_passed。

唯一入口scripts/run_chronos2_weather_readout.py；唯一记录reports/Chronos2WeatherReadout_two_protocols_seed42_20261012/runs.json。检查证据为同目录preflight_gpu.json；b历史Token缓存单独保存并逐窗口核对原forecast、A01预测、全部输入/标签/站点/时间及历史边界。各输出保留attention.npy、analysis.json：全量MAE/RMSE/R²、24个horizon、各自及固定A02最差10%窗口、注意力熵/均值/样本差异/Memory相似度；零天气Query、同批天气Query置换、均匀注意力均保持Memory及直接天气支路不变，只用于诊断。注意力不是因果解释。全部正式测试及产物验收后登记既有Adapter独立表，完成即停，不扩展种子或调参。

### 2026-10-13 A04验收与结果分析

四组全部通过保存产物复核：全量指标、原CSV标签、与A02一致的站点/时间、配置/协议/scaler、有限权重及每站非零预测、验证MAE选优（含epoch0）、patience10、正式日志及唯一结果索引一致。A04-a/b可训练参数均29,249；attention.npy的形状、概率归一化及熵重算通过。复用原GPU选定权重重放证据（最大差4.18×10⁻⁷以内），本次未重新推理。机器复核补入原runs.json的acceptance_review_20261013。

同协议A02基线MAE/RMSE/R²：in_domain=0.043908633/0.095870190/0.849283622，zeroshot_v1=0.043621510/0.092941636/0.858350874。Δ=当前−A02；负误差变化为改善。

| 协议 / 版本 | 最佳轮 / 完成轮 | ΔMAE（相对变化） | ΔRMSE（相对变化） | ΔR² |
| --- | --- | --- | --- | --- |
| in_domain / A04-a | 70 / 80 | +0.000001323（+0.0030%） | -0.000056692（-0.0591%） | +0.000178195 |
| in_domain / A04-b | 46 / 56 | +0.000053266（+0.1213%） | +0.000292176（+0.3048%） | -0.000920053 |
| zeroshot_v1 / A04-a | 52 / 62 | +0.000172274（+0.3949%） | +0.000419494（+0.4514%） | -0.001281559 |
| zeroshot_v1 / A04-b | 42 / 52 | +0.000088126（+0.2020%） | +0.000498165（+0.5360%） | -0.001522543 |

A04内部域内三项最佳均为a（MAE/RMSE/R²=0.043910/0.095813/0.849462）；跨站b的MAE较低（0.043710），a的RMSE较低/R²较高（0.093361/0.857069），没有三项统一最优的跨站A04。相对A02，a域内仅RMSE微降0.0591%，MAE略差；b域内及a/b跨站三项指标全部变差。跨站相对A01仍有改善，但A02已具有该收益，因此不能视为新增天气条件化读出优于A02的证据。

既有analysis.json显示：跨站a/b仅2/7个horizon的MAE优于A02，RMSE为8/6个；两版在低NWP变化组改善，中/高组MAE和RMSE均变差。各自最差10%窗口RMSE变差0.6117%/0.8823%；固定A02最差10%窗口中a改善0.2596%，b变差0.1967%，尾部收益不一致。NWP变化分层衡量预报变化强度，不是天气不确定性，重叠窗口也不构成独立统计样本。

注意力归一化熵0.8602–0.9116，权重存在小时及样本差异，未退化为严格均匀分配；这不证明注意力有效。跨站a置均匀后RMSE反而从0.093361降到0.093150，而MAE从0.043794升到0.044199；b零天气Query时RMSE也略降、MAE变差，域内同批Query置换的两版MAE/RMSE反而略改善。这里保留Memory及直接天气支路，仅干预Query，零Query仍有位置编码；置换局限同批样本，不能作为完整天气消融或因果证明。a只能混合两个预测patch Token；b的历史位置Token已包含已知未来NWP，不能称纯历史编码。

四组无缺失正式指标，A02与A04参数量及读出预算匹配；但单seed尚未验证稳定性，未来NWP发布时间/双向插值限制仍在。完整日志、权重、预测和attention.npy/analysis.json路径见原runs.json各run，未新增图表。结论：本轮没有支持A04替代A02的稳定结果；下一步最小行动是用户验收当前证据，保持本阶段停止，不自动增种子或调参。

## 2026-10-13 统一验收与运行隔离

用户明确批准修改公共框架。`run.py` 默认使用带时间的独立运行名，所有训练入口及共同测试入口拒绝复用已有目录；续训、独立评估须另开目录。`utils/acceptance.py` 统一核对原CSV标签、完整站点/时间窗口、有限且非退化预测、重算MAE/RMSE/R²、配置/协议/scaler与验证最优checkpoint。普通模型共同验收通过后登记；FusionSF输入变体、TimeVLM、ChronosX及A02–A04还须通过既有专项检查。登记带锁并原子更新，失败或重新等待专项验收时移除该运行的索引行，验收不继承旧通过标记。

共同训练循环新增逐轮CSV，旧产物只读原日志；A03/A04重放使用保存配置，允许更换空闲GPU，保留原数值模型、数据处理及权重身份核对。新配置/协议/日志不再被旧重复文件的通配忽略规则误排除。说明与检查命令见README“记录与验收”，回归用例为`tests/test_experiment_artifacts.py`。

验证：15项合成产物/入口/并发回归检查全部通过；11组现有完整产物共同规则只读复核通过：Cross-Unet双协议，FusionSF域内power_future_nwp修复版与跨站power_history_future_nwp修复版，域内A01/A02/TimeVLM正式版/ChronosX，跨站A03-b/A04-b与域内A04-a。A03/A04历史数值代码、数据处理与checkpoint保护检查通过。另核对的TimeVLM 200轮续训目录无完整测试配置，与原runs.json中的用户停止状态一致，不算正式结果或验收通过。此次未改历史指标/权重/预测/结果索引，未启动训练或GPU重放；代码及保存产物检查不等同于新模型完整实验。

用户随后要求构思Frozen Chronos-2 + Adapter方向，当前仅讨论。建议优先保留所有PV/NWP变量的历史与未来位置表示，再连接轻量Cross-Unet风格的时序/通道Adapter；FusionSF式输入对齐与增加历史长度作为独立候选。当前原生patch尺寸16，24→24对应两个历史及两个未来patch，`[B,2,768]`来自仅保留PV未来位置；完整非REG表示可组织为`[B,16,4,768]`。A04-b已经使用PV历史+未来四个位置，本建议新增的是NWP变量表示与结构化混合，不能宣称“首次增加历史token”。未来patch数增加代表更长预测范围，不会自动提高24小时内的时间分辨率。Cross-Unet使用MSE、FusionSF使用L1+VQ，指标差异不能直接归因于encoder；新结构收益尚未验证，未修改或启动这些模型。

## 2026-10-13 全变量 Token Adapter 实验

用户审查方案后明确“基于上述实验方案，推进实验”。先执行第一阶段T1/T2/T3×in_domain/zeroshot_v1六组，seed42、24→24。冻结同协议A01；T1只读PV未来两个原生Token，T2读16变量×4位置原生Token，均使用逐Token残差MLP；T3以两尺度时间注意力、同一24小时PV/NWP相关矩阵的可学习通道混合替换MLP。共享投影64、天气/小时Query和零初始化小时残差读出；不使用Cross-Unet额外PV历史块，不改变patch尺寸/数据/scaler。T1/T2为178315参数，T3为178369参数，MLP宽906由参数量匹配确定，无性能搜索。T3是Cross-Unet启发的组合模块，不是官方encoder直接复现。

沿用A02的FP32、AdamW lr0.000016/wd0.05、batch512/eval1024、L1、max100/patience10，验证MAE选权重含epoch0。单空闲GPU串行、CPU9,10两线程、workers0。新运行独立目录；原生全变量缓存逐窗口核对已有A01预测表示、标签、站点、时间、scaler；训练前核对零残差、冻结权重、梯度、padding/mask；正式测试后从共享原始Dataset重放全部测试窗口，通过统一及专项验收后登记既有Adapter表。入口`scripts/run_chronos2_token_adapter.py`，唯一进度/路径/指标记录`reports/Chronos2TokenAdapter_stage1_seed42_20261013/runs.json`。

后续F1/F2输入位置实验与L1/L2长历史实验遵循审查方案的验证门槛：相应对照在两个协议的验证RMSE均改善至少1%，每个协议验证MAE退化不超过0.5%。不以新测试成绩决定后续结构。第一阶段完成后核对门槛再推进；不满足则保留负结果并停止。单种子探索不代表显著性，NWP发布时间缺失与双向插值限制仍在。

启动核对：15项公共验收回归通过。首个启动入口漏传use_gpu=1，实际CPU原生表示与历史GPU缓存不一致，在任何训练epoch/权重写出前被拦截；原输入及冻结权重一致，失败日志与CPU检查保留于同目录`failed_20261013T215400_*`，登记在runs.json.failed_attempts，不作为GPU检查证据。修正后显式使用cuda:0，两个协议×三变体共六项真实GPU检查通过，历史A01 Token最大差异1.90735e-6，零残差、冻结权重、梯度及padding核对通过，证据为preflight.json。有效批次20261013T215919在GPU0串行完成，最后一组于2026-10-13 22:43通过验收；状态为`stage1_accepted`，后续阶段门槛未通过，已停止。

### 2026-10-14 结果核对与阶段结论

六组保存的metrics.json、acceptance.json、training_summary.json与runs.json一致，必要日志、权重、预测均存在，独立指标CSV与分协议索引已登记。沿用昨夜完成的统一验收和每组25,450个测试窗口原始Dataset全量权重重放证据，最大预测差异2.384186e-7；本次未重复训练、推理或完整验收，只核对保存记录、指标登记和门槛算术。MMSP两协议、seed42、24→24及配置保持不变。

同协议A02测试基线MAE/RMSE/R²：域内0.043908633/0.095870190/0.849283622，跨站0.043621510/0.092941636/0.858350874。下表Δ为当前−A02，负误差变化表示改善；完整测试三指标见上方既有Adapter表。

| 协议 / 版本 | 最佳轮 / 完成轮 | Δ测试MAE（相对变化） | Δ测试RMSE（相对变化） |
| --- | --- | --- | --- |
| in_domain / T1 | 38 / 48 | -0.000167283（-0.3810%） | -0.000150078（-0.1565%） |
| in_domain / T2 | 9 / 19 | +0.000004098（+0.0093%） | +0.000133649（+0.1394%） |
| in_domain / T3 | 12 / 22 | -0.000127910（-0.2913%） | +0.000065702（+0.0685%） |
| zeroshot_v1 / T1 | 13 / 23 | +0.000031704（+0.0727%） | -0.000104912（-0.1129%） |
| zeroshot_v1 / T2 | 6 / 16 | +0.000225179（+0.5162%） | +0.001476621（+1.5888%） |
| zeroshot_v1 / T3 | 17 / 27 | +0.000372764（+0.8545%） | +0.001329914（+1.4309%） |

T1是本轮T1/T2/T3中两个协议三项测试指标均最佳的版本：域内MAE/RMSE/R²=0.043741/0.095720/0.849755，跨站=0.043653/0.092837/0.858670。该描述不用于后续结构选择；相对A02，T1跨站MAE仍略差，且参数量178,315高于A02的29,249，不能归因为结构收益。

后续门槛仅使用各自验证MAE最优checkpoint的验证指标：两协议RMSE均改善至少1%，且各自MAE退化不超过0.5%。

| 验证对照 | 域内RMSE改善 | 跨站RMSE改善 | 域内MAE退化 | 跨站MAE退化 | 门槛 |
| --- | --- | --- | --- | --- | --- |
| T2相对T1 | -0.4123% | -2.1851% | +0.3549% | +0.9972% | 未通过 |
| T3相对T2 | -0.0628% | +0.7987% | -0.0451% | -0.6269% | 未通过 |

因此不进入F1/F2或L1/L2。当前全变量Token与组合Adapter未达到预定推进标准；单seed不能证明机制普遍无效，NWP发布时间缺失与双向插值限制保留。六组正式结果无缺失，启动失败记录保留。

证据及日志、模型、输出目录均由[原runs.json](../reports/Chronos2TokenAdapter_stage1_seed42_20261013/runs.json)索引；模型在`checkpoints/{protocol}/Chronos2TokenAdapter_T{1,2,3}_seed42_sl24_pl24_20261013T215919/`，预测、Metric及逐站/逐小时分析在对应`outputs/`目录。结果表为[chronos2_adapter_metrics.csv](../reports/chronos2_adapter_metrics.csv)，未新增图表或独立报告。下一步最小行动是用户验收本阶段负结果，实验保持停止。

## 2026-10-14 Baseline收敛检查与100/10统一

原正式日志显示：Time-VLM两个协议均最佳epoch9、终止epoch10（轮数上限），最佳验证RMSE分别0.116523/0.110671；末轮分别0.116537/0.110719，只有1轮未改善，不能认定已经收敛。此前200/20续训中途被用户停止，未替代主表。TimeXer两个协议均最佳epoch4、终止epoch7（patience3），最佳验证RMSE分别0.111718/0.108074；末轮0.112507/0.108745，训练损失继续下降、验证RMSE不再改善，表现为当前配置下的验证平台或退化，而非撞到10轮上限。上述均为验证指标，不能代替测试结果。

两者原type1学习率逐轮减半，单纯增加最大轮数不能排除学习率过早衰减的影响。本轮保留初始lr（Time-VLM 0.001、TimeXer 0.0001），改为ReduceLROnPlateau factor0.5、scheduler patience2（连续3轮未达到调度器改善条件时减半；沿用PyTorch默认相对threshold=1e-4），early-stopping patience10、最大100轮。新实验从seed42重新初始化，保留旧结果；预算和调度同时改变，因此新旧差异不视为单纯轮数因果对照。

Cross-Unet原本100/10且按验证MSE选模，与验证RMSE选模数学等价，已有完整验收结果复用，不重复训练或改写旧配置。ChronosX保留冻结Chronos-T5-small、IIB+OIB、CE训练、batch1024/eval64、BF16及lr0.01线性调度；100轮使总调度步数变为7500。每轮在全量验证集计算与测试相同的20采样中位数预测的MAE/MSE/RMSE，按RMSE选权重，CE仅作诊断；验证固定seed42并隔离、恢复CPU及当前GPU随机数，避免验证采样改变训练随机序列。不会在验证时使用未来真实功率生成预测。

| 模型 | 本轮输入信息 | 处理 |
| --- | --- | --- |
| Cross-Unet | 24小时功率、历史/未来15维NWP、额外相关历史功率块（通常为再往前24小时，边界回退） | 复用已验收100/10结果 |
| Time-VLM | 历史功率+衍生图像/固定文本 | 0.068934 | 0.119808 | 0.764624 | 0.068747 | 0.119382 | 0.766292 |
| ChronosX | 历史功率+历史/未来NWP | 0.047528 | 0.108813 | 0.805843 | 0.048782 | 0.107841 | 0.809296 |
| TimeXer | 历史功率+历史NWP | 0.072885 | 0.120689 | 0.761150 | 0.072329 | 0.120635 | 0.761361 |

这些是各模型原输入下的baseline比较，不能将结果差异归因为架构本身；本轮不增加同输入模型改造。Time-VLM保留原验证/测试期间按forward顺序更新记忆库的行为，未借此次预算调整改变结构或状态处理。两协议各一条串行任务，顺序Time-VLM→TimeXer→ChronosX；两个协议合计最多两GPU、CPU9/10与0/7各两线程、workers0。每个模型通过共同验收和既有专项验收后才更新原主表并进入下一模型，失败停止。仅当前三模型六组，结束即停。

原四条收敛曲线的逐轮数值与日志来源保存在`reports/Baseline_alignment_100_10_seed42_20261014/convergence_before.json`；每个模型的`runs.json`保存独立运行名、参数、旧结果路径及新日志/权重/预测路径。准备阶段18项回归检查通过；真实ChronosX GPU预测验证检查另存`preflight_chronosx.json`。当前新实验结果以运行记录为准，准备检查不代表已经完成训练；新结果验收前主表仍展示旧结果，未新增图表。

运行核对更正：沙箱进程命名空间中的`ps`未显示宿主PID，一度误判Time-VLM退出；宿主`ps`及GPU进程表确认原协调器和两条训练始终存活，日志持续写入。未重启、未续训、未重复实验。由该误判生成的`reports/Baseline_alignment_100_10_seed42_20261014_retry1/`仅含未启动配置，状态统一标记`cancelled_duplicate_not_started`，没有checkpoint/output或正式结果。

Site #0协议`site0_v1`沿用相同目标时间边界、训练期NWP scaler与评价口径，窗口7657/2545/2545。A01消融准备阶段20项回归检查通过；三条真实GPU训练窗口验证Full与删除首变量均产生[3,2,768]目标Token但编码不同，Linear梯度有限且B131冻结检查通过，证据`reports/A01_site0_covariate_ablation_seed42_20261014/preflight_gpu.json`。这不是正式训练或性能结果，未访问验证/测试指标。


Time-VLM两协议100/10、验证RMSE选模的新实验已通过正式测试与专项验收，主表已更新；旧实验产物保留。

in_domain完成51轮，最佳第41轮；测试MAE/RMSE/R²=0.068934/0.119808/0.764624。

zeroshot_v1完成51轮，最佳第41轮；测试MAE/RMSE/R²=0.068747/0.119382/0.766292。

证据：`reports/Baseline_alignment_100_10_seed42_20261014/TimeVLM/runs.json`；保留原输入、官方验证及测试期间顺序更新记忆库的行为，不声称纯架构对照。


### A01 Site #0 静态筛选可靠性检查（2026-10-14）

此前Full、15个LOO、联合删除及条件分析均于17:42完成；变量选择在正式测试前冻结。当前固定删除6/7/8/9/10/13/14，不依据本次测试检查更改组合。MMSP site0_v1、seed42、24→24，A01 Chronos-2+B131+Linear，训练100/10；本次只读取已有预测，未重新训练。

使用2545个连续小时预测起点的配对循环时间块bootstrap，保留每个起点的全部24个horizon，两模型共用采样索引。重采样10000次，随机种子20261014（区别于训练seed42），预先指定7天块为主，1天/3天作敏感性检查。报告percentile 95%区间；判据为主分析和敏感性分析的RMSE差值区间上界均小于0。

Full测试RMSE 0.0870160704，联合删除0.0863224567，差值−0.0006936138（−0.7971%）。7天块95%区间[−0.0079168906, +0.0085849704]；1天[−0.0074868379, +0.0078570482]、3天[−0.0075330512, +0.0079832783]也包含0，故主指标收益可靠性尚未确认，不能解读为证明无收益。Full/联合删除MAE为0.0384787080/0.0371585524，差值−0.0013201556（−3.4309%），7天区间[−0.0026116844, −0.0001532616]；这是次要指标证据，不能事后替代RMSE判据。

该检验仅描述固定模型在当前测试时段的采样不确定性，依赖近似平稳及时间块假设，不覆盖训练随机性或跨站泛化；原NWP发布时间缺失及双向插值限制仍保留。动态选择机制的设计前提尚未满足，本次未启动设计或训练。下一步最小建议为讨论固定组合的多种子复核，需单独确定范围，不继续按测试成绩筛选变量。

复现脚本：`scripts/check_a01_static_reliability.py`；完整结果及原预测路径：`reports/A01_site0_covariate_ablation_seed42_20261014/static_reliability.json`。训练日志、checkpoint和原结果见同目录`runs.json`与`summary.json`；本次无新增模型和图表。


### A01固定静态组合多种子复核（2026-10-14 21:04启动）

用户明确授权启动多种子复核。复用seed42，新增2026/2027；各自仅训练Full和固定删除6/7/8/9/10/13/14，共四组新增训练及正式测试。MMSP site0_v1、A01 Chronos-2+B131+Linear、24→24、原100/10预算及MAE选checkpoint不变；比较主指标为测试RMSE，不重新筛选变量。每个种子及变量组合有独立B131/Linear训练路径。单空闲GPU串行，CPU9,10两线程，num_workers0；每组前重查空闲卡，无空闲卡等待，失败停止。入口scripts/run_a01_static_multiseed.py；tmux会话pvtsfm-a01-multiseed；状态与日志/权重路径索引reports/A01_site0_static_multiseed_20261014/runs.json（当前任务详情在seed2026或seed2027/runs.json）。完成后汇总同测试集的配对种子差异，再结合时间块不确定性判断；动态机制设计仍以可靠性确认为前提。


### A01多种子复核验收与动态设计判断（2026-10-14）

新增四组于21:08:26完成，均正常退出并通过产物验收；本次复核六组预测/标签/时间/站点对齐、有限值及RMSE重算一致。固定组合在seed42/2026/2027的测试RMSE相对变化分别为−0.7971%、−6.1633%、−1.6112%。三种子的RMSE算术均值从0.0864446243降至0.0839436895，绝对变化−0.0025009348（−2.8931%）；这不是集成预测成绩。最低单次RMSE为联合删除seed2027的0.0823763132，不据此选种子。

对六组预测使用相同时间块索引（7天主分析，1/3天敏感性，10000次，bootstrap seed20261014），计算每次重采样的三个种子RMSE差值均值。7天95%区间[−0.0076619786,+0.0030192677]；1天[−0.0073415435,+0.0027648661]，3天[−0.0074320808,+0.0028012589]均跨0。seed2026单独区间不跨0，但不能事后只选此种子替代整体证据。多种子支持收益方向的一致性，未解决测试时段不确定性；区间条件于这三个已训练模型，并不覆盖全部训练随机性，也不能把三个种子当作三个独立测试期。

结论：依据此前约定的可靠性前提，暂不进入动态选择机制设计；不认定无效，但不宣称稳定显著获益。下一步仅建议讨论固定方案的独立时段验证，尚未启动扩展实验。证据reports/A01_site0_static_multiseed_20261014/summary.json及reliability_review.json；模型、日志、预测路径见runs.json及各seed的test_result.json；无失败及新增图表。


### A01动态软门控开发（2026-10-14最新授权）

用户在明确知晓静态筛选可靠性尚未确认后，要求进入动态设计并启动。首版仅MMSP site0_v1/seed42/24→24，训练与验证开发，不访问旧测试集、不新划分所谓独立测试期；独立最终评估时段尚未确定。新增models/Chronos2DynamicGate.py与scripts/run_a01_dynamic_gate.py，不修改原A01结构或已验收结果。

选择器为64→32→15的MLP（2575参数），输入历史功率均值/标准差/末值、历史15维NWP均值/标准差/末值、逐预测小时未来NWP和归一化horizon，sigmoid输出[B,24,15]，初值0.95。历史NWP使用未来24小时门控均值。门控乘在原生instance_norm之后、patch embedding之前的数值段；时间编码、有效性mask及功率通道保持原值，避免归一化抵消。门控为0不等同于删除变量token，是软数值加权，不能按精确子集选择解释。

Chronos-2冻结但保留输入反向传播。阶段1从头训练selector+B131，lr分别0.000016/0.0016；阶段2冻结selector+B131，训练Linear lr0.000016。两阶段均100/10、L1、AdamW wd0.05、验证MAE选checkpoint，比较报告RMSE，沿用A01阶段定义。有效训练batch512，第一阶段microbatch8累积（末组按实际样本数归一）；FP32、单空闲GPU/CPU9,10两线程/workers0。仅选择器冻结后允许缓存编码特征，不复用旧固定编码训练门控。

CPU真实训练样本32/56检查通过：全开门控与原生编码最大误差0，零门控表征差3.279443，选择器梯度L1为0.0063389，冻结主干无梯度。属于必要实现检查，不是完整训练或独立性能验收。启动后重做GPU真实批检查，空闲卡出现才开始，失败停止、不自动扩展种子或搜索。报告reports/A01_site0_dynamic_gate_seed42_20261014，tmux会话pvtsfm-a01-dynamic-gate；state=completed_validation_only仅表示开发完成。相对原A01增加了可训练参数，单一比较尚不能隔离动态性贡献。


2026-10-14收工：用户要求今天结束并推送GitHub，已停止pvtsfm-a01-dynamic-gate等待会话，确认PID2773005已退出，动态门控尚未训练，runs.json标为paused_before_training。手动恢复命令见README的“手动运行A01动态软门控”；仅恢复未开始训练的队列，不支持将已有训练当作新运行覆盖。CPU必要检查已通过，完整GPU训练与独立测试未完成。数据、预训练权重、checkpoint二进制与预测数组继续只保存在本机。

### A01动态软门控验证审阅（2026-10-15）

最新授权与对照定义：用户要求继续固定门控对照以区分动态性贡献。固定版保留相同2575参数MLP、0.95初始化和门控lr0.00016，但输入固定为7657个训练窗口全部horizon的64维输入均值（仅训练集计算），每变量输出一个全局可学习权重，不随样本或horizon改变。该均值保存在config.json的fixed_gate_input中；其余Site #0/seed42/24→24、B131/Linear学习率、FP32、batch512/microbatch8、两阶段100/10、验证MAE选模均保持一致。独立运行reports/A01_site0_fixed_gate_lr10_seed42_20261015，入口新增--gate-mode fixed；单空闲GPU、CPU9/10两线程。必要检查在非零末层权重下核对门控对输入及horizon不变。仅此一组训练/验证对照，完成后审阅，不访问测试集或扩展种子。相同参数量不等于相同函数容量或优化几何，结论限定于此配对设计。

后续授权：用户要求按分析继续推进至阶段性成果并发邮件。本轮先仅提高门控学习率10倍至0.00016，保持Site #0、seed42、24→24、初始化、B131/Linear学习率、有效batch512/microbatch8、FP32及100/10不变，从头训练一个对照。原实验不覆盖，测试集不访问；对照完成后审阅指标、门控变化并发送完整阶段总结。入口scripts/run_a01_dynamic_gate.py新增独立运行目录及selector学习率参数；记录reports/A01_site0_dynamic_gate_lr10_seed42_20261015/runs.json，后台会话pvtsfm-a01-gate-lr10，CPU9/10两线程、单空闲GPU。

执行结果：10倍门控学习率对照于2026-10-15约02:04完成，GPU0已释放。第一阶段36轮/最佳26轮，第二阶段53轮/最佳43轮。验证MAE/RMSE/R²为0.033988587/0.081479230/0.849074836；相对原门控MAE减少0.000556287（1.6103%）、RMSE减少0.001453804（1.7530%）；相对原A01 Full分别减少0.000261751（0.7642%）和0.000799104（0.9712%）。门控范围扩大到0.366399–0.990394，相对初值平均绝对变化0.040987，跨样本标准差均值0.019211；支持原学习率限制更新幅度，但未隔离动态性贡献。固定联合删除仍有更低验证RMSE0.076951。本次2545个窗口对齐与指标重算通过，两阶段冻结权重一致，最终权重两验证窗口CPU重放最大预测误差1.49e-7。完整阶段总结及邮件正文为reports/A01_site0_dynamic_gate_lr10_seed42_20261015/stage_summary.md，结果表validation_comparison.csv、审阅validation_review.json和重放checkpoint_replay.json在同目录。仅单种子验证证据，无测试集访问或显著性结论；该轮阶段目标完成；固定门控对照随后完成，验收见下文。

用户授权恢复后，在空闲GPU0、CPU9/10两线程运行；GPU真实批前后向检查通过。selector+B131训练36轮，验证MAE最佳第26轮；Linear训练15轮，按既定验证MAE选第5轮。约01:23完成，状态completed_validation_only，未访问测试集。Site #0、seed42、24→24及两阶段100/10保持不变。

已读取保存预测重算指标：2545个验证窗口，预测/标签/门控均有限；与原A01 Full和固定联合删除的标签、站点、时间逐元素一致，指标重算误差为0。动态门控验证MAE/RMSE/R²为0.034544874/0.082933034/0.843640984；原Full为0.034250338/0.082278335/0.846099934。动态门控MAE增加0.000294536（0.8600%），RMSE增加0.000654699（0.7957%）。固定删除6/7/8/9/10/13/14的验证RMSE为0.076950676，动态门控比其增加0.005982358（7.7743%）；固定删除组合本身由该验证集筛选，不能将此比较解释为独立泛化证据。

门控范围0.947723–0.952584，均值0.950143，距初值0.95的平均绝对变化仅0.000705；跨样本标准差均值0.000258，跨horizon标准差均值0.000118。门控有变化但幅度很小，当前没有形成明显的动态抑制。第一阶段最佳轮后训练MAE继续下降而验证变差；第二阶段按MAE选第5轮，最低RMSE实际在第1轮（0.082776728），仍差于原Full，不更改预定选模准则。尚不能归因于学习率、初始化或动态机制本身，未进行新消融或重训。

结论：首版完成训练/验证，但未优于原A01；下一步最小行动是定位门控更新幅度小的原因，再决定是否修改配置。证据reports/A01_site0_dynamic_gate_seed42_20261014/validation_review.json；日志run.log、GPU检查preflight_gpu.json与产物索引runs.json在同目录。权重位于checkpoints/site0_v1/Chronos2DynamicGate_A01_site0_seed42_20261014，预测、门控及训练历史位于outputs/site0_v1/Chronos2DynamicGate_A01_site0_seed42_20261014；无新增图表，独立最终评估尚未完成。


### A01固定门控对照验收（2026-10-15）

固定门控已完成训练/验证：selector+B131完成36轮、验证MAE最佳第26轮；Linear完成16轮、最佳第6轮，均按patience10结束。2545个验证窗口的预测/标签/门控均有限，与动态lr10的标签、站点及时间逐元素对齐，MAE/RMSE/R²重算一致。两阶段selector与B131权重完全一致，固定门控在全部样本及horizon上不变，范围0.873536–0.976254；最终checkpoint在验证窗口32/56的CPU重放最大预测误差2.682209e-7，门控误差1.192093e-7。沿用已有真实GPU前后向检查，不重复训练或全量推理。验收为验证开发产物验收，不是独立测试验收；通用Metric字段test_windows在本运行实际表示验证窗口数。

| 配对版本（Site #0、seed42、门控lr0.00016） | 验证MAE | 验证RMSE | 验证R² |
| --- | --- | --- | --- |

动态版相对固定版MAE减少0.000399621（1.1621%），RMSE减少0.001210109（1.4634%）。该配对支持输入条件化门控在当前验证集上优于全局固定权重；同参数量不等于同函数容量或优化几何，单种子验证不足以证明动态机制稳定泛化或显著收益。独立最终评估时段仍未确定，未访问测试集、未更新正式主表、未启动新实验。

唯一运行索引reports/A01_site0_fixed_gate_lr10_seed42_20261015/runs.json已登记acceptance，核验细节在同目录validation_review.json；原run.log、preflight_gpu.json、两阶段checkpoint、配置、协议、scaler、预测及训练历史完整保留。当前执行环境无法访问GPU驱动，进程退出不依赖沙箱ps推断；完成状态由完整两阶段记录及最终产物交叉确认。


### A01门控迁移至两协议：准备验收（2026-10-15）

用户批准先做in_domain/zeroshot_v1迁移核对与必要检查；本阶段未授权启动完整训练。仅将scripts/run_a01_dynamic_gate.py的协议及产物目录参数化，保留Site #0默认行为，非Site0要求显式独立run-id/report-dir；未改模型数学结构、公共Dataset或协议。

两协议均为76,570训练窗口、25,450验证窗口。与已验收A01逐项核对保存的协议JSON（坐标tuple按JSON列表规范化）及scaler数组完全一致；100/10、FP32、有效batch512、AdamW/wd0.05、B131 lr0.0016、Linear lr0.000016、L1及验证MAE选模一致，已有A01可复用为原版对照。A01采用冻结编码缓存，门控需microbatch8输入反传与累积，训练随机顺序和浮点轨迹不宣称完全相同；固定/动态门控之间沿用相同训练流程。

两协议×固定/动态共四项CPU及四项GPU3真实训练样本前后向检查通过：采样首末站点0/9或10/19，全开门控与原生编码最大误差0，零门控改变表征，selector梯度有限且非零，骨干冻结；固定门控在非零末层权重下对样本/horizon不变，动态版对输入变化有响应。固定输入均值仅由各自全部训练窗口计算，不复用Site #0均值。检查不是完整训练或正式性能复现。

拟运行四组：每协议固定与动态门控，seed42、24→24、门控lr0.00016、两阶段100/10、有效batch512/microbatch8、FP32，单空闲GPU串行、CPU9/10两线程、workers0；每组前重新检查空闲GPU。仅训练/验证开发，验证MAE选权重，主要比较验证RMSE，同时报告MAE/R²，不增加结构或种子。zeroshot_v1验证使用源站点#10–#19，不能据此宣称目标站点#0–#9泛化获益；既有目标测试已在此前实验中被观察，不称全新独立留出集。本阶段未生成测试预测或访问测试数组，正式测试边界在后续运行批准时明确。

唯一准备索引reports/A01_gate_two_protocols_seed42_20261015/runs.json，协议复用核对migration_review.json及各协议/门控目录下preflight_cpu.json、preflight_gpu.json。准备阶段状态为preparation_checks_passed_awaiting_training_approval；后续启动授权见下文。临时核查脚本已清理，证据保留，无新增Markdown报告。


2026-10-15后续授权：用户明确批准启动上述四组训练/验证。唯一串行入口scripts/run_a01_gate_two_protocols.py，顺序in_domain固定→动态→zeroshot_v1固定→动态，已有A01原版不重训。各组由原门控入口等待空闲GPU并在启动前重查，CPU9/10两线程、workers0；训练退出后核对全量验证标签/站点/时间、有限值、Metric重算、MAE选模、第二阶段冻结权重及两个窗口CPU权重重放，验收通过才进入下一组，失败停止、不自动重跑。每组验收后在此登记验证指标；主表不改，不执行测试。后台tmux会话pvtsfm-a01-gate-two-protocols，队列索引仍为reports/A01_gate_two_protocols_seed42_20261015/runs.json。


2026-10-15双GPU授权覆盖前述单GPU串行：用户明确批准两个协议并行，各协议内部固定→动态串行并逐组验收。in_domain原PID3595457在GPU4/CPU9,10继续，未停止或重启；zeroshot_v1固定版PID3636705在GPU7/CPU0,7启动，两进程各两线程、workers0，总计两GPU/四CPU核。新tmux会话pvtsfm-a01-gate-parallel；唯一入口仍scripts/run_a01_gate_two_protocols.py，父索引runs.json汇总，各协议queue.json负责实时状态。后续组仅在指定GPU空闲时开始，配置、预算及测试禁用边界不变。

旧协调器PID3595448仅暂停调度，保留为原训练父进程；in_domain接管worker等待原训练退出后回收旧协调器，不再由旧队列启动任何任务。串行接管前记录serial_handoff.json保留。首次接管因SIGSTOP状态尚未同步而未写就绪记录，新协调器在启动worker之前退出；修正后启动成功，无训练重跑，原失败日志parallel_coordinator_pre_handoff_failure.log保留。任一协议失败设置STOP，阻止另一协议启动下一组，已运行训练可完成并保存验收证据，不擅自中断。


### 三模型有限调参（2026-10-15授权）

用户要求仅TimeXer、Cross-Unet、ChronosX，资源有限、有限次数，可先in_domain或Site #1。本轮选择in_domain，复用已有同协议seed42/24→24基线验证成绩；预先固定每模型2组、共6组，不根据新测试成绩扩展搜索。TimeXer从512/512分别改d_model/d_ff=128/128，或单独dropout从0.1改0.2；Cross-Unet保持官方type3，仅学习率从0.0001分别改0.00005/0.0002；ChronosX保持冻结主干、IIB+OIB、batch1024/BF16/20采样及线性调度，仅lr从0.01分别改0.003/0.001。各候选均相对原基线独立修改，不累积改动。TimeXer较小宽度是超参变体，不声称原默认配置；各模型原输入不变。

顺序TimeXer两组→Cross-Unet两组→ChronosX两组，seed42、24→24、max100/patience10不变。统一按全量验证RMSE选checkpoint及候选（Cross-Unet原MSE与RMSE选模等价），原已验收基线也参与候选，若均不优则保留原版。ChronosX验证仍固定seed42、20次采样中位数，不用CE选最终候选，调度上限仍75步/轮×100轮。保留全部候选配置/权重/逐轮日志/验证指标，不写入正式测试索引或主表，不自动测试、跨协议推广或多种子。

入口scripts/run_baseline_limited_tuning.py只调用既有train/vali，不调用run.main或test，不修改公共框架。六组参数解析及与已有基线的差异检查通过；ChronosX max_steps/eval_steps是训练阶段派生字段，仍由原训练方法设置。各组开始前进行真实训练样本GPU有限前后向/梯度检查，再重新初始化并从头训练；失败停止不自动重跑。训练结束核对最佳epoch、全量验证权重重放与选模指标一致后才进入下一组。属于验证调参验收，不称正式测试复现。

当前门控双GPU任务保持不动。本队列等待门控总状态completed_validation_acceptance后才申请一张空闲GPU，CPU9/10两线程、workers0、单组串行；门控失败则停止等待并报错，不擅自接续。唯一索引reports/Baseline_limited_tuning_in_domain_seed42_20261015/runs.json，日志及preflight/result在同目录，tmux会话pvtsfm-baseline-limited-tuning。模型权重与配置沿用checkpoints/in_domain，验证产物沿用outputs/in_domain独立limited_tuning目录。准备检查不代表GPU检查或完整训练已完成。

两协议门控队列：zeroshot_v1/fixed完成验证产物验收，验证MAE/RMSE/R²=0.039073/0.092607/0.780281；证据reports/A01_gate_two_protocols_seed42_20261015/zeroshot_v1/fixed/validation_review.json。未访问测试集。


### 门控in_domain CPU重放异常定位（2026-10-15夜间）

用户仅批准定位重放差异，未授权恢复队列或修复模型。固定门控两协议训练均已完成；in_domain在两个验证窗口[32,25449]的CPU重放失败，zeroshot已通过原验收。两条动态门控及六组baseline调参均未启动，失败日志和STOP保留。

根因为Chronos-2原生chronos.chronos_bolt.InstanceNorm.forward中的FP32归约及scale==0分支：完全恒定的历史NWP在CPU求均值后有微小舍入残差，标准差为2.980232e-8或5.960464e-8，因非零未替换为eps；GPU同样输入的标准差为0，替换为1e-5。除以极小标准差并arcsinh后，恒定通道的归一化值相差0.8813736。失败窗口涉及NWP index12总降水、index7低云量，非权重加载失败或普通输出末位误差。

原两个窗口CPU预测最大差异0.014773458；另一张同型号空闲GPU6上重放仅1.043081e-7。共26个不同验证窗口的小批检查中，按原batch8及末尾batch2组织的GPU重放与保存预测逐位一致；CPU按原批次仍不一致，排除仅由验收重组批次引起。进一步只在诊断进程内交换归一化设备：CPU网络+GPU归一化误差1.192093e-7，GPU网络+CPU归一化误差0.014773458，差异跟随归一化路径。原阈值1e-5未改，模型、权重、协议均未改，未完整重跑验证或访问测试集。

证据为reports/A01_gate_two_protocols_seed42_20261015/in_domain/fixed/normalization_diagnosis.json及replay_diagnosis_cpu.json、replay_diagnosis_cuda_6.json，原runs.json仅增加诊断引用，不标记验收通过。建议后续按原GPU推理精度/批次核验权重身份，保留现有阈值；若要修改原生归一化则属于不同数学实现，不能当作此次验收脚本修正。当前仅完成定位，所有队列继续停止。临时诊断脚本及中间张量已清理，最终诊断证据保留。


### 固定门控GPU路径验收完成（2026-10-15夜间）

用户批准按原GPU计算路径完成验收，仅本阶段，不恢复队列。scripts/run_a01_gate_two_protocols.py的accept增加显式device参数，并保持原验证批次边界；此次使用与原GPU4同型号的空闲GPU6、FP32、matmul highest/TF32 false、原默认cudnn.allow_tf32 true，不改模型数学结构、权重、协议或阈值。全量25,450验证窗口标签/站点/时间、有限值和Metric重算通过；两阶段最优MAE权重选择及selector/B131冻结一致性通过。失败窗口所在原批次[32..39]和[25448,25449]共10个窗口GPU重放，预测与门控最大误差均0，保持原预测1e-5、门控1e-6阈值。不是全量GPU推理重跑或正式测试验收。

in_domain固定门控验证MAE/RMSE/R²=0.039912/0.094079/0.797180；第一阶段19轮/最佳9轮，Linear50轮/最佳40轮。证据reports/A01_gate_two_protocols_seed42_20261015/in_domain/fixed/validation_review.json，运行索引及两个协议queue.json已同步为固定版验收通过、停止等待恢复。zeroshot已有验收不重复。原CPU失败、归一化诊断、STOP及baseline等待失败日志保留；两组动态门控、baseline六组仍未启动，本次无训练、测试或队列恢复。


### 2026-10-16正式测试与剩余队列恢复授权

用户明确同意：固定门控两协议先用各自最佳验证MAE权重做一次全量正式测试、验收并替换Adapter CSV中的验证行；随后恢复动态门控两协议，同样先验证选模、再正式测试验收。六组baseline调参仍只用验证集选参，待门控全部正式验收后单GPU串行执行。不得按本次测试成绩更换配置或权重。

新增scripts/evaluate_a01_gate.py，沿用原GPU FP32/batch8路径，不改模型、归一化、协议或训练预算。正式测试输出使用原run_id加_formal_test_20261016的独立目录，保存预测/标签/站点/时间、config、protocol、scaler、Metric、验收及权重哈希；复用公共原始CSV样本与scaler检查，另核对原两阶段JSON历史选模/冻结权重，并重载checkpoint按原GPU批次重放。该门控训练未使用公共CSV历史接口，使用其原始JSON历史作专项选模验收，不伪造历史训练记录。

首次in_domain入口在推理前因内存tuple与保存JSON列表比较退出，保存协议核对完全相同，已将比较统一为JSON语义。未生成预测或训练；失败日志及仅含协议/scaler的目录保留。正式推理仍为每协议首次。原CPU验收失败及归一化诊断保留。恢复仅跳过已正式验收的固定版，不重训；后续GPU重放保持原阈值。

固定门控两协议正式测试及产物验收完成，已同步Adapter表；原验证成绩仍保留在各validation_review.json。证据reports/A01_gate_two_protocols_seed42_20261015/{in_domain,zeroshot_v1}/fixed/test_acceptance.json。

固定门控两协议正式测试已验收并填表，现按用户授权恢复剩余动态门控；in_domain使用GPU6/CPU9,10，zeroshot等待除GPU6外任一空闲GPU、CPU0,7，最多两GPU/四CPU核。动态训练→原GPU路径验证验收→正式测试验收逐阶段串行，失败停止。两个协议全部正式验收后，baseline六组单GPU验证调参才启动。STOP旧标记改名留存，旧错误/日志保留，固定版不重训不重测。

动态门控两协议正式测试及产物验收完成，已同步Adapter表；原验证成绩仍保留在各validation_review.json。证据reports/A01_gate_two_protocols_seed42_20261015/{in_domain,zeroshot_v1}/dynamic/test_acceptance.json。

有限baseline调参六组完成，验证选参及验收见reports/Baseline_limited_tuning_in_domain_seed42_20261015/runs.json；未做测试，正式主表不改。

### 2026-10-16有限调参最优配置正式测试

用户明确批准对验证集选出的TimeXer width128（d_model/d_ff=128/128，最佳epoch14）和Cross-Unet lr0.0002（最佳epoch28）执行in_domain正式测试及验收。固定seed42、24→24、原输入、精度和评估batch；从现有权重测试，不重训、不按测试成绩重新选参。ChronosX保留原版。记录复用reports/Baseline_limited_tuning_in_domain_seed42_20261015/runs.json的formal_tests；独立输出目录后缀formal_test_20261016。

正式测试结果：两组均覆盖25,450×24个预测值，原始标签/站点/时间、协议/scaler、有限非退化预测、Metric重算和验证最优权重检查通过；按原评估batch重载权重重放首尾批，最大误差均0。GPU6串行、CPU9/10两线程、workers0；未重训，无失败。

| 模型/配置 | 原测试MAE | 当前MAE | 原测试RMSE | 当前RMSE | RMSE绝对变化 | RMSE相对变化 | 当前R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeXer/width128 | 0.072885 | 0.070145 | 0.120689 | 0.117773 | -0.002916 | -2.4158% | 0.772551 |
| CrossUnet/lr2e4 | 0.043377 | 0.042289 | 0.084374 | 0.083879 | -0.000495 | -0.5872% | 0.884629 |

总表in_domain已替换为验证集预先选定的新配置；zeroshot保持原配置，未将本轮调参跨协议推广。TimeXer in_domain宽度128/128、Cross-Unet in_domain lr0.0002；两协议配置差异须保留说明。ChronosX两候选验证均未优于原版，保留已有结果。

证据：reports/Baseline_limited_tuning_in_domain_seed42_20261015/runs.json的formal_tests包含完整Metric、基线变化、权重及输出路径；同目录TimeXer_formal_test_20261016.log、CrossUnet_formal_test_20261016.log为本次日志。原训练权重位于checkpoints/in_domain/*limited_tuning*，测试预测/标签/配置/scaler/Metric/acceptance位于outputs/in_domain/*limited_tuning*_formal_test_20261016。无新增图表。

结论：两组验证选优配置均改善in_domain测试，Cross-Unet本轮RMSE最低0.083879；单种子且测试时段曾在旧实验中观察，未来NWP发布时间可用性仍为协议假设。当前任务完成后停止，下一步最小行动为审阅并接受本轮正式结果，不自动扩展实验。

### Cross-Unet追加两组学习率（2026-10-16）

用户明确要求继续调参、正式测试并验收。本轮预先固定lr0.0003/0.0004两组，仅in_domain、seed42、24→24；其余沿用已验收lr0.0002的batch128/eval128、FP32、type3、max100/patience10、验证RMSE选模及原输入。复用已有三组验证结果，所有候选按验证RMSE选优，选定写盘后仅对新胜出配置做一次正式测试及验收；若原lr0.0002仍最佳，复用其完整测试结果。不得按测试表现改选或扩展搜索。

入口scripts/run_crossunet_lr_extension.py；索引reports/CrossUnet_lr_extension_in_domain_seed42_20261016/runs.json。启动时八卡均忙，单GPU等待空闲后逐组串行，每组前重查；CPU9/10两线程、workers0。训练复用既有必要GPU梯度检查及全量验证权重重放；测试复用公共产物验收并追加原GPU批次首尾重放。成功后更新现有总表与本文档，失败保留证据并停止。本轮测试已在此前被观察，仅作持续开发基准，不称全新独立留出集。

### STAEformer baseline接入（2026-10-16）

用户要求将STAEformer加入baseline。本阶段完成官方结构、来源、默认配置和统一入口接入；随后已授权把完整实验加入GPU等待队列。官方模型以多传感器为节点；当前Dataset_MMSP按单站点窗口取样，因此固定num_nodes=1，输入历史功率、小时和星期，保留时间/空间注意力、自适应嵌入和mixed projection，但空间注意力在单节点下退化，不称交通多节点等价复现。MMSP配置为seed42、24→24、batch16、Adam lr0.001、FP32、max100/patience10、验证RMSE，结构dim152/FF256、3层/4头、dropout0.1；待Cross-Unet追加调参进入终态且GPU空闲后，串行运行两个协议并正式测试验收。

真实in_domain首末训练窗口CPU前后向通过：76,570窗口中抽取2个，预测/目标shape均为2×24×1，参数1,121,520，loss0.357405424，梯度有限非零，小时/星期索引范围有效。该结果仅为接入检查，不是训练或Metric。代码位于models/STAEformer.py与layers/staeformer，来源commit及配置见models/baseline_sources.json、models/official_presets.py；总表新增空行，待正式测试验收后填数。

### A05实现核对与AutoDL脚本准备

用户要求核对A05并补足运行脚本。当前有Chronos2MCDecoder的a/b/c三个残差变体：a为四层解码器，b增加P-Corr，c增加4→2→1多尺度记忆；冻结同协议已验收A01，仅训练新增head。固定seed42、24→24、width128/heads4/FF256、dropout0.05、FP32、batch128/eval256、AdamW lr0.0001/wd0.05、MSE、100轮/patience10；验证RMSE包含epoch0选模，正式测试后全量原始输入重放并比较A01。

新增scripts/run_a05.py，默认只准备A05-a两协议命令及资源检查；--run才执行，可显式选择变体/协议。单GPU、两CPU核心、workers0，每组验收通过才继续。A05专用允许A01的原机器绝对路径按项目checkpoints/pretrained相对路径迁移，A02路径规则保持原样。必要资源和命令见README；原生token缓存两协议约48GiB，不依赖已删除的TokenAdapter缓存。

六组参数解析与合成张量的零残差、shape及三步梯度检查通过；未启动真实GPU编码、完整训练或正式测试，尚无A05正式Metric。脚本不自动扩展实验或选定Ours。

### 当前待办队列（2026-10-16）

1. Cross-Unet追加学习率调参：lr0.0003、0.0004，in_domain验证RMSE选优后正式测试验收；现有队列reports/CrossUnet_lr_extension_in_domain_seed42_20261016/runs.json等待空闲GPU。
2. STAEformer：待Cross-Unet队列进入完成或失败终态后，串行运行in_domain、zeroshot_v1，seed42、24→24、batch16、max100/patience10、验证RMSE选模，逐协议正式测试及验收；入口scripts/run_staeformer.py，记录reports/STAEformer_two_protocols_seed42_20261016/runs.json。

两项均使用单张空闲GPU、两个CPU核、workers0；STAEformer失败即停止且不自动重跑。全部完成后停止，不扩展种子、超参数或模型。

Cross-Unet本轮补充调参及测试验收完成。验证RMSE：lr3e4=0.091199995；lr4e4=0.092468228；已有lr0.0002=0.089116154。验证选定lr2e4，测试MAE/RMSE/R²=0.042288504/0.083878796/0.884628735。相对上一轮测试RMSE变化+0.000000000（+0.0000%）。主表按验证选优结果更新，不按测试反馈改选。若沿用原版则复用已验收测试，没有重复推理。完整日志、配置、权重、选模、预测与验收路径见reports/CrossUnet_lr_extension_in_domain_seed42_20261016/runs.json；无新增图表。单种子、测试已被观察及未来NWP可用性假设的限制保留。当前两组完成后停止，不扩展其他参数或协议。
