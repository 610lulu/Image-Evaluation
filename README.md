# Image-Evaluation

一个面向面试展示的 **Photo Agent Evaluation MVP**。项目参考 DeepEval 的思路，但只保留最容易讲清楚、最容易运行的核心闭环：

> Dataset → Photo Agent → Tool Calls → Evaluators → Batch Runner → Report → Regression Test

目标不是做完整商业平台，而是证明你理解 **Agent 测试、自动化评测、Tool Call 验证、Badcase 分析和测试工程化**。

## 1. MVP 能做什么

当前内置一个可控的 `PhotoAgent`，支持两个工具：

- `search_photos`：按地点、时间、场景、对象、排除条件搜索照片
- `rank_photos`：按 `aesthetic_quality` 对候选照片排序

示例请求：

```text
找出去年在东京拍的夜景照片，排除自拍，然后选出最好看的三张。
```

预期 Agent 轨迹：

```text
search_photos(location=Tokyo, time=last_year, scene=night, exclude=[selfie])
        ↓
rank_photos(criterion=aesthetic_quality, limit=3)
        ↓
返回 3 张照片
```

## 2. 评测指标

MVP 先采用确定性评测，避免为了 Demo 引入外部 API Key。

### Task Completion

验证最终任务是否完成，例如用户要求 3 张照片，最终是否确实返回 3 张。

### Tool Correctness

验证 Agent 是否调用了正确的工具，以及调用顺序是否正确。

### Argument Correctness

验证工具参数，例如：

- `location=Tokyo`
- `time=last_year`
- `scene=night`
- `exclude=[selfie]`

这种确定性字段优先使用规则判断，而不是交给 LLM Judge。

## 3. 项目结构

```text
Image-Evaluation/
├── data/
│   └── cases.jsonl
├── src/image_evaluation/
│   ├── __init__.py
│   ├── photo_agent.py
│   ├── evaluators.py
│   └── runner.py
├── tests/
│   └── test_evaluation.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 4. 快速运行

```bash
git clone https://github.com/610lulu/Image-Evaluation.git
cd Image-Evaluation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

运行自动化测试：

```bash
pytest -q
```

运行完整评测：

```bash
PYTHONPATH=src python -m image_evaluation.runner
```

运行后会生成：

```text
reports/report.json
reports/report.md
```

其中 `report.md` 包含：

- 总 Case 数
- Pass Rate
- Average Score
- 每个 Case 的结果
- Bad Case 及失败原因

## 5. 测试集设计

`data/cases.jsonl` 每一行都是一条测试 Case：

```json
{
  "id": "case_001",
  "query": "找出去年在东京拍的夜景照片，排除自拍，然后选出最好看的三张。",
  "tags": ["search", "night", "ranking", "constraint"],
  "expected_tools": ["search_photos", "rank_photos"],
  "expected_arguments": {
    "search_photos": {
      "location": "Tokyo",
      "time": "last_year",
      "scene": "night",
      "exclude": ["selfie"]
    }
  },
  "expected_photo_count": 3
}
```

`tags` 是为了后续做 Slice Analysis，例如：

```text
night
portrait
backlight
motion_blur
selfie
ranking
```

真实影像项目中可以继续扩展为曝光、白平衡、噪声、锐度、动态范围等摄影维度。

## 6. 为什么 MVP 不直接用 LLM-as-a-Judge

对于 Tool Name、Tool Arguments、返回数量这类确定性结果，规则评测具备：

- 可重复
- 无额外成本
- 易定位失败原因
- 适合 CI 回归

开放式问题才适合引入 Judge，例如：

```text
“这三张照片哪张最好看？”
```

后续可新增：

```text
AestheticJudge
├── exposure
├── color
├── sharpness
├── noise
├── dynamic_range
└── composition
```

可以由多模态模型评分，并用人工 Golden Set 做校准。

## 7. 后续扩展方向

为了保持 MVP 简洁，以下功能暂未实现，但非常适合作为面试中的下一步方案：

1. 接入真实 LLM / LangGraph Agent
2. LLM-as-a-Judge / Multimodal Judge
3. Agent trajectory 更灵活的 strict / subset / unordered 匹配
4. 按 `tags` 输出 Slice 指标
5. old vs new 模型 Regression Diff
6. 并发执行 10 万 Case
7. FastAPI / Streamlit Dashboard
8. GitHub Actions CI
9. Classification / Detection / Segmentation 指标
10. FiftyOne Badcase 可视化

## 8. 面试怎么讲这个项目

建议用下面这条主线：

> 我做了一个轻量级相册 Agent 自动化评测框架。测试集使用 JSONL 管理，Runner 批量调用 Agent，并记录 Tool Calling trajectory。Evaluator 分别评估 Task Completion、Tool Correctness 和 Argument Correctness，再输出汇总指标与 Badcase 报告。对于确定性输出我优先用规则评测，以保证可重复和可回归；对于审美这类开放式问题，我会进一步引入 Multimodal LLM-as-a-Judge，并通过人工 Golden Set 校准。

这能对应真实岗位中的：测试方案、测试集构建、自动化脚本、Agent 缺陷定位、测试报告和 AI 测试提效。
