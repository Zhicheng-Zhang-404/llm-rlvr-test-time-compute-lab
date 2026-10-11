# LLM + RL 实验：Test-time Compute Scaling 与 VERL GRPO

这是一个可直接迁移到 4×NVIDIA A800 80GB 节点的中文实验项目，只覆盖课程的基础任务与进阶任务：

- Qwen3-0.6B 在 MATH-500 前 100 题上每题采样 16 次；
- Math-Verify 规则判分；
- 无偏 pass@k（k=1/2/4/8/16）、总曲线与难度分层曲线；
- VERL 数据构造、规则 reward、4 卡 GRPO 3-step smoke training；
- 每个阶段的 fail-fast 验收、排错文档和报告模板。

> 本仓库不提交模型权重、数据 parquet、生成结果或训练 checkpoint。它们体积大且可能包含机器路径，统一由 `.gitignore` 排除。

## 一分钟开始

```bash
conda activate pedagogy
python -m pip install -e .
make test
python scripts/check_environment.py --require-gpu
make base
make prepare-verl
bash scripts/install_verl.sh
make grpo-smoke
```

不要跳过前一条命令的 `CHECKPOINT PASSED`。完整操作、成功标准和截图要求见 [A800 集群逐关手册](docs/CLUSTER_RUNBOOK.zh-CN.md)。

## 目录

```text
src/ttc_rlvr/                 基础实验：数据、vLLM、判分、pass@k、绘图
verl_reward/math_reward.py    VERL 自定义规则奖励
scripts/                      环境检查、数据转换、运行和产物验收
tests/                        纯 CPU 单元测试
docs/                         集群手册、排错表、报告模板
```

## 方法上的关键选择

pass@k 对每题先按

\[
1-\frac{\binom{n-c}{k}}{\binom nk}
\]

计算，再对题目取平均。不能用“把所有回答混在一起”代替，也不能把 16 个样本的经验命中率直接冒充所有 k 的 pass@k。

GRPO 的 validation 是 MATH-500 第 0–99 题，training 是第 100–499 题。这样 smoke run 不会把基础评估题用于训练。reward 只检查数学等价性；无法抽取的回答记为 0，不让单条脏输出中断分布式训练。

VERL 固定为 `v0.4.1`，因为这个 tag 的依赖范围明确覆盖题目给出的 PyTorch 2.6 与 vLLM 0.8.3；不跟随持续变化的 `main`。

## 验证边界

仓库内单元测试不需要 GPU，覆盖 pass@k、聚合、JSONL 和 VERL reward contract。真正的 1,600 次生成、Math-Verify 实测判分和 GRPO step 必须在题目给出的 Linux/A800 环境执行；成功与否由脚本检查真实产物/日志，而不是仓库预置“示例成功日志”。

## 结果如何写

基础实验跑完后从 `artifacts/base/metrics.json` 读取数字，用 `100 × (pass@16-pass@1)` 报告“提升了多少个百分点”。分析与表格骨架见 [实验报告模板](docs/REPORT_TEMPLATE.zh-CN.md)，环境/依赖错误见 [排错记录](docs/TROUBLESHOOTING.zh-CN.md)。

## 官方参考

- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [Math-Verify](https://github.com/huggingface/Math-Verify)
- [VERL Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- [VERL GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html)
- [VERL + vLLM 0.8](https://verl.readthedocs.io/en/latest/README_vllm0.8.html)

## License

MIT
