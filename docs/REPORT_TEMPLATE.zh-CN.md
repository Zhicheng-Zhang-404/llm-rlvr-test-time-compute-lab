# 基础与进阶实验报告模板

## 1. 环境

记录日期、节点、GPU、CUDA driver、Python、torch、vLLM、transformers、datasets、math-verify 与 VERL 的实际版本。粘贴 `scripts/check_environment.py` 的输出。

## 2. Test-time Compute Scaling

### 2.1 设置

- 模型：Qwen/Qwen3-0.6B
- 数据：MATH-500 前 100 题
- 每题样本数：16
- Temperature / Top-p / Top-k：0.6 / 0.95 / 20
- 最大生成长度：2048 tokens
- 判分：Math-Verify；抽取失败按错误计
- 随机种子：42

### 2.2 结果

从 `artifacts/base/metrics.json` 填入：

| k | 1 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| pass@k |  |  |  |  |  |

插入 `artifacts/base/pass_at_k.png`。

### 2.3 分析

1. 提升百分点：`100 × (pass@16 - pass@1)`。百分点不是相对百分比。
2. 边际收益：比较相邻增量，而不是只看曲线“向上”。通常随着 k 增长，至少命中一次的新增概率会递减，但题目间异质性会影响曲线形状。
3. 对 RL 的启示：若某些题 pass@1 低但 pass@16 明显非零，说明当前策略已经能偶尔探索到正确轨迹，规则奖励可给出学习信号；若 16 条全错，该组 advantage 退化，单纯增加训练步数未必有效，需要改善探索、提示或初始模型。
4. 难度分层：比较每层 `pass@16-pass@1`，同时报告每层题数，避免把小样本波动误判为 scaling 更强。
5. 局限：16 次采样来自伪随机生成；只评 100 题；Math-Verify 可能有抽取/等价判定误差；pass@k 表示“有 verifier 能选中正确候选”的潜力，不等于无 oracle 的用户准确率。

## 3. VERL + GRPO

### 3.1 数据与防泄漏

说明前 100 题作为 validation，后 400 题作为 smoke-run training；记录 parquet 行数和 schema 检查结果。

### 3.2 配置

列出 group size 8、prompt batch 16、response length 1024、学习率 1e-6、KL coefficient 0.001、4 GPU、3 个 smoke steps。解释为什么没有 critic，以及全同 reward 组对 GRPO 有什么影响。

### 3.3 规则奖励

说明 `verl_reward/math_reward.py` 从最终回答中抽取 LaTeX 并与 ground truth 做符号/数值等价验证。正确为 1，错误、无法抽取或判分异常为 0。

### 3.4 启动证据

插入包含训练 step、reward、KL、loss/长度指标的日志截图。注明这只是流程 smoke test，不能声称模型已经收敛或精度提升。

## 4. 遇到的问题与解决

逐项写出症状、根因、验证方法、修复和修复后的证据。不要只粘贴最终命令。

## 5. 结论

用实测数字回答三个基础思考题，并明确进阶部分已经验证到哪个训练 step。

