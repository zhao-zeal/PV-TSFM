# FusionSF 验收记录

状态：两个协议完整训练及正式测试完成，2026-10-10按用户请求验收通过。MMSP、seed42、24→24、官方三模态guide17；配置详见两个启动config.json及checkpoint/config.json。

| 协议 | 完成轮数 | 验证最优轮次 | 测试MAE | 测试RMSE | 测试R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| in_domain | 30 | 10 | 0.040875803 | 0.091264941 | 0.863415558 |
| zeroshot_v1 | 29 | 9 | 0.041598372 | 0.089822095 | 0.867700063 |

权重按验证MAE选择，连续20轮无改善后早停；测试不参与选模。验收逐项核对全部25,450×24标签与原始CSV一致、站点/时间索引与协议一致、形状及数值有限、各站预测非全零、权重有限、配置/协议/scaler对应、checkpoint最优轮次与日志一致。独立重算MAE/RMSE/全局R²，与metrics.json、日志和唯一结果索引行一致；Zero-shot checkpoint记录源站点，测试输出记录目标站点，差异符合协议。

本轮为完整训练与正式评估，验收期间没有重新推理、训练或改配置。第2–6轮曾匹配全零验证参考，第7轮后恢复，初期原因仍未确认。只有seed42；无同协议、同输入、同模型配置的已验收旧基线，因此绝对/相对改进不可计算，不跨协议排名。NWP预报发布时间缺失、双向插值的严格在线可用性限制仍保留。

证据：本目录runs.json的acceptance字段、两个training.log与启动配置；`checkpoints/{protocol}/FusionSF_official_script3_guide17_seed42_sl24_pl24_all_full/`保存最优权重与训练配置/协议/scaler，`outputs/{protocol}/同运行名/`保存测试配置/协议/scaler、pred/true/site_ids/forecast_timestamps.npy及metrics.json；`reports/{protocol}/results.csv`保存正式指标索引。没有生成新图表。

用户已批准验收后启动下一个模型Cross-Unet的in_domain与zeroshot_v1、seed42；新实验记录于`reports/CrossUnet_two_protocols_seed42_20261010/runs.json`。Cross-Unet完成两个协议后先验收，TimeXer需后续批准。Site1本轮不重训。
