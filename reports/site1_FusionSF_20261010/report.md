# 实验总结

本次为 MMSP Site #1 的 FusionSF 独立训练：24→24、seed42、官方三模态 guide17、batch16、AdamW lr0.0016、余弦预热5轮、最多100轮、验证 MAE 早停20轮。

已完成4轮后发现全零预测并主动停止；第5轮尚未完成，以training.log为准。最优验证权重来自第2轮，验证 MAE0.11134913287192007、RMSE0.23242938078803516。第2–4轮指标与全零预测诊断参考完全相同，绝对变化0、相对变化0%；该参考仅用于诊断，不是新增研究baseline。完整测试 MAE/RMSE/R² 未测，不可将验证指标当成测试结果。

权重在验证集首、中、末3个窗口的预测均为0；输出头 ReLU 前数值为−4.7071～−4.0145，回归输出被截断。训练损失下降不能作为该模型有效收敛的证明。默认学习率过大是待验证假设，不能确定为唯一原因。

日志 training.log；配置 config.json；逐轮表 training_curve.csv；诊断 diagnosis.json；权重 /home/zhaopp/workspace/PV-TSFM/checkpoints/site1_v1/FusionSF_official_script3_guide17_seed42_sl24_pl24_all_site1/checkpoint.pth。未生成图表或完整测试预测。

Time-LLM已从当前baseline清单、实现和运行入口移除；Llama本地残留和指定缓存已删除，明细见removal.json；其他模型权重保留，历史验收记录保留。

结论：默认学习率的本次Site1运行异常，不予验收。用户已批准仅将学习率降为0.00016、其余配置不变，从头重跑一次；新运行目录 reports/site1_FusionSF_lr0p00016_20261010，未启动其他baseline。
