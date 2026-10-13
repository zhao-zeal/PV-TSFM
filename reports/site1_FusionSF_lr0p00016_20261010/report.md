# 实验总结

本次按用户明确批准，只将 FusionSF Site1 学习率从0.0016降为0.00016重跑一次。MMSP Site #1、24→24、seed42、官方三模态guide17、batch16、AdamW、余弦预热5轮、100轮上限、验证MAE早停20轮、GPU0、CPU核心0–3、DataLoader0均保持不变；从头训练，不加载旧权重。

## 结果及比较

原运行完成4轮后中止，第5轮未完成；降学习率运行完成3轮后中止，第4轮未完成，以training.log为准。两次均发现全零预测，但不是正常早停完成，也不属于充分训练后的有效baseline结果。

| 配置 | 最优验证MAE | 最优验证RMSE | 最优轮次 | 完整测试MAE/RMSE/R² |
|---|---:|---:|---:|---|
| lr0.0016 | 0.11134913287192007 | 0.23242938078803516 | 2 | 未测 |
| lr0.00016 | 0.11134913287192007 | 0.23242938078803516 | 2 | 未测 |

重跑对原运行的验证MAE、RMSE绝对变化均为0，相对变化均为0%；完整测试变化和R²均未测。不存在本次可验收的最佳有效结果；这里只记录数值最低的第2轮异常checkpoint。

## 异常证据与官方核对

两次运行第2、3轮完整验证指标均与全零预测诊断参考完全一致。降低学习率后的第2轮权重，在训练集首、中、末三个窗口上，训练态和评估态预测均为0；回归输出头权重和偏置梯度均为0。训练态ReLU前数值范围为 `-3.512603` 到 `-1.983612`。

确认官方来源commit f979e6faa80e75deea4f80302ceefc47f915d147也采用末端ReLU、L1+VQ训练损失以及验证MAE选择checkpoint和早停。这些已核对部分与项目一致，没有为了取得指标而替换官方激活或损失。抽查证据不能证明每个训练批次的梯度都为0，也不能确定首次退化的直接原因。

降低十倍学习率没有解决问题，不能认定学习率是唯一根因。下一步应定位首次退化的训练步骤，区分输出头优化问题和训练实现/依赖差异；进一步配置或模型改动、完整重跑须讨论并获用户批准。

未运行测试集，未用测试指标调参，未启动第三次尝试或其他baseline。NWP预报发布时间缺失的既有在线可用性限制仍保留。

## 保存位置

- 重跑配置 `/home/zhaopp/workspace/PV-TSFM/reports/site1_FusionSF_lr0p00016_20261010/config.json`。
- 重跑日志 `/home/zhaopp/workspace/PV-TSFM/reports/site1_FusionSF_lr0p00016_20261010/training.log`；原运行日志 `/home/zhaopp/workspace/PV-TSFM/reports/site1_FusionSF_20261010/training.log`。
- 重跑权重 `/home/zhaopp/workspace/PV-TSFM/checkpoints/site1_v1/FusionSF_official_script3_guide17_seed42_lr0p00016_site1/checkpoint.pth`；训练协议和scaler位于同一目录。
- 梯度诊断 `/home/zhaopp/workspace/PV-TSFM/reports/site1_FusionSF_lr0p00016_20261010/gradient_diagnosis.json`；原运行诊断 `/home/zhaopp/workspace/PV-TSFM/reports/site1_FusionSF_20261010/diagnosis.json`。
- 汇总 `/home/zhaopp/workspace/PV-TSFM/reports/site1_FusionSF_lr0p00016_20261010/summary.json`；两次逐轮数据各保存为相应报告目录的training_curve.csv。
- 结果对照表 `/home/zhaopp/workspace/PV-TSFM/reports/site1_FusionSF_lr0p00016_20261010/comparison.csv`；图 `/home/zhaopp/workspace/PV-TSFM/reports/site1_FusionSF_lr0p00016_20261010/training_comparison.png`。
- 本次没有完整测试预测或正式测试结果，不写入正式baseline结果索引。

Time-LLM 已移除，当前 baseline 共8个；Llama本地残留及指定缓存已删除，移除明细 `/home/zhaopp/workspace/PV-TSFM/reports/site1_FusionSF_20261010/removal.json`。源码与报告已保存本地，尚未提交或推送GitHub。

结论：获批的降学习率重跑未通过验收，已停止。下一步最小行动是先讨论并批准一次保留官方结构的训练退化诊断，暂不继续盲目调参或推进下一模型。

2026-10-10后续核对：site1_v1站点、窗口、样本输入/时间及训练网格NWP scaler检查通过，证据见summary.json的protocol_recheck。两次旧运行均在5轮warmup内被中止；现有十站点实验在类似第2–6轮全零平台后于第7轮恢复，因此旧抽查只能证明当时输出/梯度状态，不能证明Site1永久失效。本次没有新训练或正式测试，也没有修改官方架构、数据协议或超参数。
