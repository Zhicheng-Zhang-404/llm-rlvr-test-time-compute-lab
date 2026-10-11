# A800 集群逐关实践手册

这份手册面向题目给出的环境：Python 3.11、PyTorch 2.6.0+cu118、vLLM 0.8.3、4 张 A800 80GB。所有长任务都由前置检查点保护；看到 `CHECKPOINT PASSED` 后才进入下一关。

## 0. 获取项目并确认环境

```bash
git clone <本仓库 URL>
cd math-llm-rlvr-lab
conda activate pedagogy
python -m pip install -e .
python scripts/check_environment.py --require-gpu
```

期望看到：CUDA 为 `true`、4 张 `NVIDIA A800 80GB PCIe`，以及 torch、vLLM、transformers、datasets、math-verify 均可导入。`pynvml` 的弃用信息只是 warning；若希望消除，可安装 `nvidia-ml-py`，但它不阻塞实验。

## 1. 基础实验（约 100 × 16 次生成）

建议先只生成 2 题，检查模型、模板和显存：

```bash
ttc-eval generate --limit 2 --samples 16 --output artifacts/dryrun.jsonl
wc -l artifacts/dryrun.jsonl
```

输出应为 2。确认后执行正式流水线：

```bash
make base
```

它严格依次执行：

1. 环境/GPU 检查；
2. 对前 100 题各生成 16 条，逐题追加到 JSONL，可断点续跑；
3. 验证恰好有 100 行，每行 16 个输出；
4. 用 Math-Verify 判分；
5. 计算 `k = 1, 2, 4, 8, 16` 的无偏 pass@k；
6. 生成总曲线和难度分层曲线；
7. 验证指标位于 `[0,1]` 且随 k 单调不降。

产物位于 `artifacts/base/`：

- `generations.jsonl`：原始回答；
- `scored.jsonl`：每个回答的布尔判定；
- `metrics.json`：总指标和按难度指标；
- `pass_at_k.png`：曲线图。

如果任务被调度器中断，原命令重跑即可；完整的题目 ID 会被跳过。不要在不同采样参数之间复用同一个输出文件。

## 2. 安装/验证 VERL

官方针对 PyTorch 2.6 + vLLM 0.8.3 的路线是从源码安装 VERL，并将 `tensordict` 固定到 0.6.2。项目把 VERL 固定在 `v0.4.1`：该版本的安装元数据明确支持 vLLM 0.8.2–0.8.5，且本项目使用的配置键已经逐项对照该 tag。运行：

```bash
bash scripts/install_verl.sh
```

若 `flash-attn` 尚未安装，再执行：

```bash
python -m pip install flash-attn --no-build-isolation
```

安装后二次验收：

```bash
python scripts/check_environment.py --require-gpu --require-verl
python -m verl.trainer.main_ppo --help >/dev/null
```

脚本会安装 VERL 的 `vllm,math` extras，但现有 vLLM 0.8.3 已满足约束，不应被升级。生产/课程环境更稳妥的方式仍是新建 Python 3.10 conda 环境，或使用 VERL 官方文档给出的容器；不要在正在承担其他课程作业的环境里盲目升级 torch。

## 3. 准备无泄漏的 GRPO 数据

```bash
make prepare-verl
```

脚本将 MATH-500 的题号 `0..99` 写入 `validation.parquet`，题号 `100..499` 写入 `train.parquet`。这保证基础实验的 100 题没有进入训练集。每行包含 VERL 所需的：`data_source`、chat 格式 `prompt`、`ability`、`reward_model.ground_truth`、`extra_info`。

快速检查：

```bash
python - <<'PY'
from datasets import load_dataset
d = load_dataset("parquet", data_files={
    "train": "artifacts/verl_data/train.parquet",
    "validation": "artifacts/verl_data/validation.parquet",
})
print(d)
print(d["train"][0])
assert len(d["train"]) == 400
assert len(d["validation"]) == 100
assert d["train"][0]["extra_info"]["index"] == 100
print("CHECKPOINT PASSED: no train/eval overlap")
PY
```

## 4. 启动 3 步 GRPO smoke run

建议申请一台 4×A800 节点，确认 `CUDA_VISIBLE_DEVICES` 中正好有 4 张卡，然后：

```bash
make grpo-smoke
```

关键配置为：每题 8 个 rollout、全局 16 个 prompt（每步 128 条轨迹）、GRPO advantage、actor KL loss 0.001、无 critic、vLLM rollout、FSDP actor。脚本只训练 3 步，目的是验收全链路而不是得到有意义的收敛结果。

成功标准：

- `artifacts/grpo_smoke.log` 中出现 `step:0`/`global_step`；
- 无 traceback、OOM 或 RayTaskError；
- 日志出现 reward/score、response length、actor loss 或 KL 等训练指标；
- 最后打印 `CHECKPOINT PASSED`。

截图时同时截入步数与至少三项指标；不要把环境检查或模型下载日志当成“训练已启动”的证据。

## 5. 资源调整

0.6B 模型在 4×80GB 上余量很大。如果共享节点实际只分配到更少 GPU，必须同步修改 `trainer.n_gpus_per_node`，而不是只改 `CUDA_VISIBLE_DEVICES`。若仍然 OOM，按顺序降低：

1. `data.train_batch_size`（同时保证轨迹数能被 mini-batch 合理划分）；
2. `data.max_response_length`；
3. `actor_rollout_ref.rollout.gpu_memory_utilization`；
4. per-GPU micro batch size。

基础采样默认 tensor parallel 为 1，能让单卡独立承载小模型；如果希望只使用一张卡，可设 `CUDA_VISIBLE_DEVICES=0`，不会改变 pass@k 定义。
