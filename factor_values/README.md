# 因子值代码

仅计算及更新因子值，不计算IC。固定范围见factor_scope.json，公式参数、输入窗口和频率见factor_catalog.csv。

src/factor_engine包含当前通用计算引擎、Tushare本地数据适配、PIT、历史股票池、调度与存储。此代码不等于所有历史独立修复批次的冻结版本；不能不经版本绑定就覆盖历史。Tushare原始行情、财报版本档案、ST状态及凭证在本地提供，数据缺失不能关闭门禁强跑。

## 更新结果打包

上游批准的Tushare计算流程先生成不可变因子结果manifest。该manifest包含financial_version_policy=latest_available_pit_v1、data_quality_status=APPROVED、partitions列表；每分区包含factor_id、year、path、sha256、formula_version、source_version、data_cutoff。

执行 `python sync_factor_values.py --manifest /本地/新增因子值清单.json --output /本地/版本化交付目录`。脚本核验固定247项、重复因子年度、SHA256、结果字段及版本记录，再保存独立快照和增量ZIP，保留之前版本。它只处理上游已经计算好的因子值，不自行取数或调用计算引擎。

取数及计算由本地自动任务每日北京时间23:30检查推进，沿用对应版本和批准输入；当前尚未完成247项全量增量计算的端到端验证，缺少绑定的项必须报告阻塞。原频率日137/月110保持不变。
