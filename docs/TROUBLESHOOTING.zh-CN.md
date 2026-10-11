# 排错记录与常见问题

| 症状 | 常见根因 | 处理与验证 |
|---|---|---|
| `pynvml` FutureWarning | 旧包被依赖间接安装 | 非阻塞；可安装 `nvidia-ml-py`，重新运行环境检查 |
| vLLM import 成功但运行时报 CUDA symbol 错误 | PyTorch、CUDA wheel、vLLM wheel ABI 不一致 | 在同一环境打印版本和 `nvidia-smi`；按 VERL vLLM 0.8.3 官方镜像或配套 wheel 重建环境 |
| Hugging Face 连接超时 | 计算节点无外网/代理未配置 | 在可联网节点预下载模型和数据，运行时把 `MODEL_PATH` 指向共享盘；不要把 token 写入仓库 |
| `ForkingPickler` import error | vLLM 0.8 升级带来的 tensordict 组合问题 | `pip install tensordict==0.6.2`，再跑 `check_environment.py --require-verl` |
| Ray 看不到 4 张 GPU | 调度器资源与进程环境不一致 | 检查 `CUDA_VISIBLE_DEVICES`、`ray status` 和 `torch.cuda.device_count()`；使 `trainer.n_gpus_per_node` 与实际一致 |
| vLLM OOM | rollout KV cache 或并发过高 | 降低 `gpu_memory_utilization`、prompt batch 或 response length；每次修改后从 1 step smoke test 开始 |
| GRPO reward 全为 0 | 模型未遵循 boxed 格式、ground truth schema 错、解析失败 | 抽查 `solution_str`；单独调用 `compute_score`；确认 prompt 要求 `\\boxed{}` |
| GRPO group advantage 全为 0 | 一个组内 8 条全部同奖（全对或全错） | 这是相对优势的数学结果；检查整体比例，必要时增加探索或改善起点，而不是篡改判分 |
| Hydra 提示 unknown key | 安装的 VERL commit 与脚本面向的配置版本不同 | 对照 `python -m verl.trainer.main_ppo --help` 与官方当前配置；记录 commit，调整该单个 override |
| pass@k 不是单调的 | 聚合公式、样本数或 k 输入错误 | 必须基于每题的 `n,c` 使用无偏公式再跨题平均；运行 `validate_artifacts.py base` |

版本排错时先保存：`pip freeze`、VERL commit、完整命令和 traceback。不要一口气升级多个核心包，否则无法定位是哪项改变解决或引入了问题。

