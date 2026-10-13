# 项目工作约定

- main 使用 TSLib 核心分层：`run.py`、`exp/`、`data_provider/`、`models/`、`layers/`、`utils/`；当前 BasicTS 版保存在 `basicts-snapshot-20261010` 分支。
- 后续光伏实验在本项目目录运行，使用 `datasets/MMSP/data` 和 `pretrained/` 中的本地资源。main 的训练和评估不依赖 BasicTS；所需计算层和许可证保留在layers，不保存整套上游仓库。
- 所有模型共用 `Dataset_MMSP` 和 `data_provider(args, flag)`；站点、时间窗口和 scaler 由 `site1_v1`、`in_domain`、`zeroshot_v1` 协议统一决定，不建立模型专用 Dataset。历史 `paper_main_v1` 保留原含义，不作为无全量 scaler 泄漏的正式协议。
- 数据协议与历史迁移先读 `docs/DATA_PROTOCOLS.md`；历史探索保留在独立FusionSF项目，PV-TSFM负责新的可验收论文证据。fixed_v1是历史窗口/预处理名称，不能跨项目自动认定站点协议；必须核对实际站点、时间索引、scaler、输入和评价范围。输入或协议不一致时旧成绩只保留为历史经验，正式结果重新运行；迁入代码、权重及新实验仍逐阶段审批。当前FusionSF双模态入口是功率+卫星，不能代替历史功率+NWP对照。
- 按协议将模型、预测及报告保存到本项目的 `checkpoints/`、`outputs/`、`reports/`，不再向相邻 BasicTS 或 FusionSF 项目写入新实验。
- 只执行用户当前明确授权的任务；模型清单不构成启动全部实验的授权。
- 用户要求主动监督实验流程：新指令若与已确认的协议、输入、配置或验收顺序冲突，或可能造成重复运行、混用指标、未验收即推进，先指出具体冲突与影响，提出一个必要的澄清问题；暂停相关修改或启动，待用户明确决定后执行。用户明确批准的变更更新已有记录，不反复索要同一授权。
- 按“明确任务与协议 → 核对官方结构/输入/配置 → 必要正确性检查 → 获批运行 → 正式评估并保存证据 → 用户验收”的顺序推进；已完成且仍适用的检查不重复。不得为了赶进度跳过必要核对，也不得自行增加阶段、实验或审批流程。
- 2026-10-10 最新决定：历史实验留作经验与证据；停止 BasicTS 旧训练及协调进程，保留 PV-TSFM 两条已从头训练的 FusionSF，作为本轮新实验继续记录并验收，不重跑。验收以最终测试 MAE/RMSE/R² 为准，训练中验证成绩须明确单列，历史成绩不替代新实验成绩。
- 2026-10-10 执行记录：用户批准“验收，然后启动后续模型”；FusionSF两个协议已完整训练、正式评估并通过产物验收，继而启动Cross-Unet两个协议seed42、24→24。记录为reports/CrossUnet_two_protocols_seed42_20261010/runs.json。
- 2026-10-11 当前执行：用户已明确要求估计Cross-Unet耗时后运行TimeXer；Cross-Unet两个协议已完成并通过完整产物验收，原先未完成/无空闲卡的冲突已解除，依此授权启动TimeXer两个协议seed42、24→24。记录为reports/TimeXer_two_protocols_seed42_20261011/runs.json；TimeXer完成后先验收，不自动启动其他模型。
- 用户明确授权删除内部BasicTS归档及无用验收/调试产物，接受相关材料丢失，后续需要时重新实验；不得再次创建完整内部归档。独立FusionSF目录保留。已有正式结果和必要科研证据保留，历史成绩不进入新正式结果索引。
- 每个阶段修改或运行前须经用户审批；本阶段完成后停止，验收并等待下一阶段审批。已验收内容跳过；发现冲突或含糊之处先讨论。当前正式训练只做 `in_domain` 和 `zeroshot_v1`，只运行 seed 42，不做多种子。按 FusionSF → Cross-Unet → TimeXer 逐模型推进，每个模型完成两个协议并验收后，再申请下一模型审批；Ours 单独审批。
- 2026-10-10 Site1决定：用户恢复Site1作为重要探索协议，数据协议核对已完成。暂不修复Site1的FusionSF全零预测，尤其不得影响正在进行的实验；正式十站点协议保持不变。未批准Site1新的完整训练、官方架构变更或调参。已核对Site #1的24→24窗口7657/2545/2545；旧FusionSF两次在warmup内被中止，不能将早期全零预测当作永久退化的证明。
- 当前 baseline 共8个：DLinear、PatchTST、FusionSF、Time-VLM、Cross-Unet、ChronosX、TimeXer、Chronos-2。用户已移除 Time-LLM 并要求删除 Llama 本地权重；一次性验收产物按本次明确授权清理，保留现有正式实验记录。
- 2026-10-11用户批准Chronos-2主版本固定history_future并复用经一致性验收的历史完整推理结果，不重新训练/推理；唯一证据为reports/Chronos2_two_protocols_seed42_20261011/runs.json。另维护reports/chronos2_input_metrics.csv及docs/EXPERIMENT_STATUS.md中的四输入表，固定8列：models、input、两个协议各MAE/RMSE/R²，四行none/history/future/history_future，六位小数；表内只放输入和Metric。主表与分协议正式索引仅登记history_future，不按测试成绩选择，不计新增实验或seed，不自动扩展后续实验。此项明确授权为历史成绩默认不入正式索引规则的例外。
- 2026-10-11用户接受DLinear/PatchTST旧版本与原作者官方实现/配置的差异，批准复用经完整证据验收的in_domain结果并填入主表；必须标明FusionSF仓库旧实现、历史功率+经纬度，分协议索引baseline_config=legacy_fusionsf，不称官方原版复现。唯一验收记录为reports/DLinear_PatchTST_migration_seed42_20261011/runs.json。随后用户明确批准将zeroshot历史登记R0004/R0005成绩填入主表，必须标注原产物缺失、未完成当前完整验收；仅展示，不写入分协议正式测试索引，不自动补跑或改模型。此项为历史结果默认不进入正式索引规则的明确例外。
- 统一实验总表为reports/experiment_summary.csv，阅读入口为docs/EXPERIMENT_STATUS.md的“实验总表”；固定8列：models、input、MMSP in_domain的MAE/RMSE/R²、MMSP zeroshot的MAE/RMSE/R²。第二列input写实际消费信息，未确认版本写待确认。当前seed42、24→24；每次正式验收后从实际metrics.json与验收记录同步更新CSV和该既有文档，总表CSV和文档统一保留六位小数，原始精度保留在metrics.json及分协议索引，缺失留空/显示“—”，不填训练/验证、历史探索或未验收结果；用户明确批准展示的DLinear/PatchTST zeroshot历史登记为例外，必须标注来源与缺失证据，不能称完整验收通过。不重复新建总表报告。
- 长期工作先读 `docs/EXPERIMENT_STATUS.md`，其中记录协议、最新授权、进度、异常、待办和产物路径；恢复任务时先核对已有进程与 checkpoint，避免重复启动。Cross-Unet、TimeXer及本次明确批准的Chronos-2 Adapter两协议seed42均已完成必要产物验收和记录；没有批准其他后续模型、多种子或调参。
- 本次最多使用两张空闲 GPU；两个并行进程合计最多 4 个 CPU 核，每个进程绑定两个核心并限制计算线程为 2，DataLoader 使用 0 个子进程。实验产物按既有规范保存；必要说明更新已有文档，会话按下述简洁交付规范汇报，不自动发邮件。
- TimeXer两个协议已在2026-10-11 04:05写出正式测试指标且退出，完整产物验收已通过并填表，不再视为活跃训练；FusionSF和Cross-Unet已完成，旧检查进程确认完成后自行退出；BasicTS不作为活跃监控对象。正常不频繁记录或汇报，不调用模型API。检查退出、NaN/Inf及疑似全零/连续不变验证指标，异常单独记录供核对；不自动停止、改配置或重跑。不得恢复每分钟监控。

## PV-TSFM 科研执行与交付规范（2026-10-10）

历史实验负责提供研究线索，新框架负责建立可信的论文证据。

以下为用户最新要求，覆盖此前有冲突的文档生成、总结格式与交付要求。

1. 框架冻结：未经明确授权，不修改实验框架、数据协议及公共接口。
2. 最小修改：只修改当前任务必要的代码，不主动重构、扩展功能。
3. 限制文档：默认不新建 Markdown/TXT 报告，不重复生成总结。必要时更新已有文档。
4. 证据保留：实验配置、Metric、Checkpoint、运行日志按项目既有规范自动保存，不得为简洁而删除。
5. 结果可信：区分代码检查、单批次测试、完整训练和正式评估，不得将前者称为实验复现成功。
6. 简洁交付：每次任务完成后，仅输出以下内容，总计不超过 300 字：
   - 完成内容
   - 修改文件
   - 验证结果及证据路径
   - 关键 Metric（如有）
   - 遗留问题（最多 3 项）
7. 不重复介绍项目背景，不输出冗长过程记录。仅在出现重要异常或需要人工决策时详细解释。

## 文件清理规范

1. 自动清理：任务结束后，删除本次任务产生的无用临时文件、调试产物、重复报告和已失效的过程文档。
2. 精简文档：同一任务只保留必要的最终说明，不保留多个内容重复的中间版本。
3. 保留科研证据：不得擅自删除原始数据、Checkpoint、实验配置、正式 Metric、预测结果、必要日志及历史实验记录。
4. 安全检查：删除前检查 Git 状态、文件引用和使用情况；无法确认是否仍有价值的文件，列出清单交由用户决定。
5. 简洁汇报：任务结束时仅汇报完成内容、关键结果、修改文件、清理文件及遗留问题，不超过 300 字。

原则：自动清理确定无用的文件，保留可复现的科研证据，不因精简而破坏已有实验。

## Time-VLM复现实验（2026-10-11最新任务）

- 用户明确要求接下来复现Time-VLM，批准沿用MMSP in_domain、zeroshot_v1、seed42、24→24，执行官方CLIP full-shot配置的两组训练、正式测试和必要产物验收。唯一运行记录为reports/TimeVLM_two_protocols_seed42_20261011/runs.json；最多两张空闲GPU、合计四个CPU核、num_workers=0。保留官方记忆库按forward更新、77字符文本截断及图像uint8转换行为；不改公共框架、协议、官方架构或调参。完成并验收后更新既有总表，停止，不启动下一模型。

## FusionSF三种输入串行实验（2026-10-11最新授权，覆盖上条完成即停）

- 用户明确要求TimeVLM结束后执行FusionSF：历史功率+未来NWP、历史功率+历史卫星+历史/未来NWP、历史功率+历史/未来NWP；编写脚本逐组串行。沿用in_domain、zeroshot_v1、seed42、24→24，共六组，每种先in_domain再zeroshot_v1。
- TimeVLM两组退出、正式测试及产物验收通过后启动；三种输入各自完成训练、正式测试和产物验收才进入下一组，失败停止。唯一队列为scripts/run_fusionsf_inputs.py，证据为reports/FusionSF_inputs_two_protocols_seed42_20261011/runs.json。串行使用一张空闲GPU、CPU0,7两线程、num_workers=0，不改协议，不扩展种子或调参。
- 必要输入改造：历史功率与15维历史NWP按历史时刻拼接，未来NWP保留原guide17；无卫星输入跳过并移除卫星分支及context VQ。保留原script训练配置、ReLU输出及TS VQ。共享covariate字典及协议不变，仅在不加载卫星时补齐fusion_time的1×1时间编码。研究变体标为fusion_input_variant，独立表reports/fusionsf_input_metrics.csv，不覆盖原FusionSF主表。全部完成后停止，不启动其他实验。

## TimeVLM续训与ChronosX串行复现（2026-10-11最新授权）

- 2026-10-11 06:52最新决定覆盖下述自动继续授权：用户先停止TimeVLM、随后暂停GPU3，要求聚焦FusionSF，并明确训练GPU利用率达不到99%就停止不做。同卡GPU0/CPU0,7、batch16/FP32/num_workers0的原版三模态与功率+未来NWP各25秒真实步骤剖析均未达到99%（原版均值57.65%、峰值93%；无卫星均值13.22%、峰值14%），因此两项训练、两条队列均已退出，ChronosX及其他待运行组未启动。不自动恢复、不擅自增加batch/CPU/实验规模或用额外计算制造99%；保留已有最佳权重、原日志和等价加速代码。证据reports/FusionSF_inputs_two_protocols_seed42_20261011/original_vs_variant_gpu_profile.json及两个gpu_utilization_*.csv。恢复须由用户重新明确决定。

- 用户随后明确要求暂停当前实验、解决GPU0/GPU3低利用率后继续，授权必要的公共NWP读取及TimeVLM等价预处理优化。NWP按坐标缓存标准化后的原数组；TimeVLM保留官方77字符截断和PIL bicubic/crop，缓存分词、整批图像传输及原NumPy uint8归一化查表，并减少记忆指针同步。两协议真实样本、TimeVLM像素/预测/梯度/记忆状态逐位一致后才能恢复；不改batch、精度、训练预算、数据协议或官方模型数学结构。
- 本次暂停只存在最佳权重，不能完整恢复未保存状态：TimeVLM从当前最佳epoch10继续epoch11，FusionSF从epoch2继续epoch3，丢弃并重算后续未保存状态。暂停前权重与日志保留在对应before_input_optimization文件中，来源记录在两runs.json的interruption。唯一加速核验证据reports/TimeVLM_ChronosX_queue_20261011/input_optimization_check.json；不得把单批吞吐或逐位检查称为正式性能复现完成。

- 用户发现各baseline训练预算不同，最新决定：已经运行的其他实验先不改；本次TimeVLM改为总上限200轮/patience20继续训练。当前FusionSF实际为100/20，现有六组队列预算保留，固定GPU3；接管协调器时不停止或重复已有训练。
- TimeVLM两协议从现有第9轮最佳checkpoint恢复模型、记忆、优化器和原type1调度器，epoch10开始；GPU0/CPU9,10两线程、num_workers0，串行执行。第10轮末状态、原随机数及AMP scaler未保存，不能声称完整训练轨迹恢复；旧10/5配置、指标、权重及日志完整保留。原lr0.001的type1保持每轮减半，恢复lr3.90625e-6。本次标extended_budget，证据reports/TimeVLM_resume200_pat20_seed42_20261011/runs.json，验收后才更新主表。
- 用户随后明确要求TimeVLM后接ChronosX，两者使用同一GPU，覆盖此前ChronosX只写脚本不运行的限制。仅in_domain、zeroshot_v1，seed42、24→24；两个TimeVLM协议完成并验收后，串行运行ChronosX两个协议，仍GPU0/CPU9,10，与FusionSF合计两GPU/四CPU核。
- ChronosX固定本地Chronos-T5-small、IIB+OIB，冻结主干，只训练注入层；历史功率+历史/未来NWP、官方30通道含缺失标记；结构、token CE、AdamW lr0.01、accumulation2、FP32、clip1、20采样中位数保留。为遵循用户统一200/20预算，从官方fixed5000 steps改为每完整epoch验证CE/选权重/早停，线性调度按200轮的239400优化步计算；不是官方默认预算。证据reports/ChronosX_two_protocols_seed42_20261011/runs.json。
- 唯一GPU0队列脚本scripts/run_timevlm_chronosx.py，队列证据reports/TimeVLM_ChronosX_queue_20261011/status.json；必要前后向/恢复/采样检查记录在两模型runs.json的preflight。ChronosX预测前保存CPU/CUDA随机数用于小批权重身份验收，不重做全量推理。两个模型均正式测试并验收后停止，不扩展模型、种子、超参搜索或发邮件。

## Chronos-2 Adapter迁移与复现实验（2026-10-11）

- 用户明确批准保持统一输入，迁入E11/E12结构，在in_domain、zeroshot_v1重新训练seed42、24→24；不复用旧raw-NWP权重/特征/Metric作当前正式结果。新实验仍经Dataset_MMSP，框架其他协议与模型不改。
- 版本固定b131_linear_v1（历史E11）与b131_mlp63_v1（历史E12）；B131_v1为两者共享、按当前协议重新训练的第一阶段依赖，不是追加baseline。命名、源版本、验证选优、配置与验收记录见reports/Chronos2Adapter_two_protocols_seed42_20261011/runs.json。
- 两协议可用两张空闲GPU并行，每协议的两个版本串行；合计最多4个CPU核、num_workers=0。仅执行这四组及其必要两阶段训练/测试/验收，结束后停止，不扩展种子、搜索、模型或队列。

- 用户本次明确要求先验收TimeXer并登记主表，再监控Adapter；TimeXer两协议已通过完整产物验收并填表（7轮/最低验证MSE权重第4轮），证据为reports/TimeXer_two_protocols_seed42_20261011/runs.json的acceptance。不得再次视为活跃训练或重复运行。

- Adapter四组已完整训练、测试并通过产物验收，训练进程均已退出；当前任务完成后停止，等待用户验收，不启动下一模型或队列。研究变体的独立表为reports/chronos2_adapter_metrics.csv及docs/EXPERIMENT_STATUS.md对应表，不覆盖官方Chronos-2主表或指定为Ours。唯一验收证据为reports/Chronos2Adapter_two_protocols_seed42_20261011/runs.json的acceptance，原始配置、权重、缓存、日志、预测和启动失败记录保留。

## ChronosX独立复现等待空闲GPU（2026-10-12最新授权）

- 用户明确要求复现ChronosX，注意GPU利用率并可适当调参，随后明确“运行，但是等有了空的GPU再开始”。允许采用已测batch1024/BF16，即使平均利用率低于99%也运行；此前99%停止条件不再用于本次ChronosX。
- 仅in_domain、zeroshot_v1，seed42、24→24、max200/patience20，冻结Chronos-T5-small，训练IIB+OIB。固定batch1024/eval64、BF16、accumulation1、AdamW lr0.01/wd0、clip1、20次采样中位数，验证CE选权重；75步/轮、线性调度总上限15000步，记录与官方默认差异。
- 单GPU串行，CPU9,10两线程，num_workers0。等待任意完全空闲GPU（无计算进程、显存<=1024MiB、利用率0%），每个GPU阶段开始前再次检查；空闲卡出现后自动完成必要GPU BF16检查、两个协议训练/正式测试/产物验收并更新主表。失败停止，不自动重跑。
- 唯一入口scripts/run_chronosx.py；运行状态reports/ChronosX_two_protocols_seed42_20261011/runs.json，协调日志coordinator_20261012.log。仅本次ChronosX，TimeVLM/FusionSF队列保持停止；两协议完成后停止，不发邮件或扩展实验。

## A01/A02 Weather Adapter（2026-10-12最新授权）

- 用户批准命名A01、A02并编写代码和开展实验。A01为已验收b131_linear_v1/E11的显示编号，保留旧名、权重路径和结果，不重训。A02冻结同协议A01的Chronos-2、B131及Linear读出，仅训练29249参数Weather Adapter；使用适配后表征与逐小时未来NWP作功率残差修正，不改官方骨干、数据协议或公共数据接口。
- 仅in_domain、zeroshot_v1，seed42、24→24；沿用Adapter读出预算max100/patience10、FP32、batch512/eval1024、AdamW lr0.000016/wd0.05、L1、验证MAE选权重（含epoch0零残差）。单空闲GPU串行，CPU9,10两线程、num_workers0；两协议真实GPU原生/缓存一致性、零初始化、冻结权重、梯度及小时对齐检查通过后运行，失败停止，不自动重跑。
- 唯一入口scripts/run_chronos2_weather_adapter.py；唯一运行及验收记录reports/Chronos2WeatherAdapter_two_protocols_seed42_20261012/runs.json。正式评估和完整产物验收后更新既有reports/chronos2_adapter_metrics.csv及docs/EXPERIMENT_STATUS.md，不覆盖官方Chronos-2主表，不指定为Ours，不启动其他模型或消融。仅天气、仅表征及联合分支贡献对照尚未批准。

## A02验收后继续FusionSF输入实验（2026-10-12最新授权）

- 用户明确要求做完A02并验收后，在GPU4继续原FusionSF三种输入、两个协议串行六组，允许调参或改变batch以提高利用率；随后明确取消此前99%停止条件，按真实吞吐优化并记录利用率。此授权覆盖旧GPU3固定及99%停止要求；不恢复TimeVLM或其他队列。
- A02两协议已完成训练、正式测试及必要产物验收，唯一证据reports/Chronos2WeatherAdapter_two_protocols_seed42_20261012/runs.json的acceptance；A01原已验收结果不重训，A02独立表已同步。保留登记顺序/协议整字典比较的失败日志与修正记录，未重复训练或全量推理。
- FusionSF沿用in_domain、zeroshot_v1、seed42、24→24、FP32、max100/patience20及原学习率/优化器/架构/输入协议。先对batch16/64/256做有界真实步骤吞吐比较，逐输入固定最快可用batch及经显存检查的eval batch；不按测试Metric调参，不扩展种子或模型。GPU4单卡串行、CPU0,7两线程、num_workers0。
- 首组power_future_nwp/in_domain从已有第2轮最佳checkpoint继续epoch3；其余五组未运行。保留本次续训前配置、权重和旧日志，记录batch调整导致轨迹不等价及原RNG/后续状态未保存的限制。唯一队列scripts/run_fusionsf_inputs.py及原reports/FusionSF_inputs_two_protocols_seed42_20261011/runs.json；每组训练、正式测试、产物验收通过才继续，失败停止，不自动重跑。全部完成停止，既有独立输入表更新，不覆盖原FusionSF主表。

## FusionSF首组全零修复重跑（2026-10-12最新授权）

- 用户明确批准“修复重跑”。原首组至epoch23正式测试25450×24预测全零、验收失败，原epoch2权重已在ReLU死区；原失败Metric/预测/权重/日志/配置保留，失败尝试登记在同一runs.json的failed_attempts，不进入正式测试索引或独立指标表。
- 功率+未来NWP两个协议统一使用readout_repair_v1，从seed42、epoch1重新训练，不复用退化checkpoint/优化器。仅把读出Linear权重缩小并保留随机方向（行L2≤0.05/sqrt(128)）、bias0.1；读出组学习率为主干1/100（0.000016），主干仍0.0016。仅正初始化的短检查会再次全零，保留该失败诊断；降低读出lr后的两协议真实batch256梯度及非零输出检查通过后才启动。
- 原架构/ReLU/VQ/L1+VQ/协议及100轮/patience20、FP32、warmup5-cosine100不变，batch256/eval64；另两种输入仍按原初始化/batch64或256运行。固定GPU4、CPU0,7两线程、num_workers0。首组完整训练/正式测试/产物验收通过后再接原队列后五组，失败停止，不自动进行下一次重跑。
- 唯一记录与队列不变，修复实验使用独立checkpoint/output目录及日志；必要检查为reports/FusionSF_inputs_two_protocols_seed42_20261011/positive_readout_preflight_20261012.json。正式索引在产物验收后登记，研究表标明readout_repair_v1，不称官方默认初始化或单纯输入因果消融，不影响Site1或原FusionSF主表。

## FusionSF剩余输入扩展修复（2026-10-12最新授权）

- 用户明确选择“停止当前组，扩展修复后重跑”。功率+未来NWP两协议已正式测试并验收，不重跑；含卫星/历史未来NWP的旧in_domain训练已于第10轮后停止，原最佳epoch2在六样本中ReLU前值全负、输出及读出监督梯度全零。旧权重、配置、日志、GPU采样及诊断保留，登记同一runs.json的failed_attempts；该旧尝试没有全量正式测试，不称已完成复现。
- 将同一positive_v1初始化和读出lr=主干1/100扩展到power_satellite_history_future_nwp与power_history_future_nwp的两个协议，共剩余四组；从seed42/epoch1重新训练，独立readout_repair_v1目录和日志/GPU采样，不复用退化checkpoint。原架构、ReLU、VQ开关、L1+VQ、数据协议、100/patience20、FP32及warmup5-cosine100保持。
- 固定GPU4、CPU0,7两线程、num_workers0，含卫星batch64/eval64、无卫星batch256/eval64。四组经真实批次初始正输出、warmup和峰值学习率的有界功率监督梯度/非零预测检查后，按原顺序串行训练、正式测试及验收；失败停止，不自动再重跑。必要检查为reports/FusionSF_inputs_two_protocols_seed42_20261011/positive_readout_remaining_preflight_20261012.json，唯一队列及runs.json不变。
- 每组验收后更新已有独立输入表和文档，保留单种子及batch不同的比较限制；修复不称官方默认配置，不影响原FusionSF主表、Site1、A01/A02或其他队列。全部完成后停止。
