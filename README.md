# Image-Evaluation

目标不是做完整相册产品，而是展示一套可解释、可回归、可横向对比不同 Agent 框架的质量评测闭环：

```text
Test Cases -> Agent Adapter -> Tool Calls / Result -> Evaluators -> Bad Cases / Report
```

当前支持四种运行模式：

- `mock`：确定性 Mock Agent，零 API 成本，用于 CI 和框架回归。
- `responses`：直接使用 OpenAI Responses API，由本项目维护 Tool Calling Loop。
- `openai-sdk`：使用 OpenAI Agents SDK，由 SDK 管理 Agent / Tool Loop。
- `langgraph`：使用 LangChain `create_agent`，底层运行在 LangGraph runtime。

`llm` 保留为 `responses` 的兼容别名。

> 三种真实 Agent 的 LLM 是真实的；`search_photos` / `rank_photos` 后端仍是确定性 Stub。这样能在相同 Tool 环境下比较不同 Agent 框架的意图理解、Tool Selection、参数生成和任务完成质量。

## 1. 当前评测什么

### Task Completion

验证最终任务是否完成，例如用户要求 3 张照片，最终是否返回 3 张。

### Tool Correctness

验证工具调用及顺序，例如：

```text
search_photos -> rank_photos
```

### Argument Correctness

验证 Function Call 参数是否准确，例如：

```text
location=Tokyo
time=last_year
scene=night
exclude=[selfie]
rank_photos.limit=3
```

## 2. 架构

```text
                    data/cases.jsonl
                           |
                           v
                  +----------------+
                  | Eval Runner    |
                  +-------+--------+
                          |
       +------------------+------------------+
       |                  |                  |
       v                  v                  v
 ResponsesPhotoAgent  OpenAIAgentsSDK    LangGraphPhotoAgent
       |                  |                  |
 Responses API        Agents SDK        LangGraph runtime
       +------------------+------------------+
                          |
                 search_photos / rank_photos
                       (Stub Tools)
                          |
                          v
                     AgentResult
                          |
                          v
          Task / Tool / Argument Evaluators
                          |
                          v
                 report / benchmark report
```

统一输出结构：

```json
{
  "answer": "已选出三张照片。",
  "tool_calls": [
    {
      "name": "search_photos",
      "arguments": {
        "location": "Tokyo",
        "time": "last_year",
        "scene": "night"
      }
    }
  ],
  "selected_photo_ids": ["photo_001", "photo_002", "photo_003"]
}
```

Evaluator 只依赖统一 `AgentResult`，不依赖具体 Agent 框架。

## 3. 安装

```bash
git clone https://github.com/610lulu/Image-Evaluation.git
cd Image-Evaluation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 运行 Mock Agent

不需要 API Key：

```bash
PYTHONPATH=src python -m image_evaluation.runner --agent mock
```

报告输出：

```text
reports/mock/report.json
reports/mock/report.md
```

运行自动化测试：

```bash
pytest -q
```

## 5. 运行真实 Agent

配置 API Key：

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-5.4"   # 可替换成你账号可用的模型
```

### Responses API

```bash
PYTHONPATH=src python -m image_evaluation.runner --agent responses
```

### OpenAI Agents SDK

```bash
PYTHONPATH=src python -m image_evaluation.runner --agent openai-sdk
```

### LangGraph

```bash
PYTHONPATH=src python -m image_evaluation.runner --agent langgraph
```

也可以单次覆盖模型：

```bash
PYTHONPATH=src python -m image_evaluation.runner --agent langgraph --model gpt-5.4
```

每次模型实际执行的 Tool Name 和 Arguments 都会被记录，并与 `cases.jsonl` 中的 Ground Truth 比较。

## 6. 一键 Benchmark

默认比较三种真实 Agent：

```bash
PYTHONPATH=src python -m image_evaluation.benchmark
```

等价于：

```text
responses,openai-sdk,langgraph
```

也可以指定模式：

```bash
PYTHONPATH=src python -m image_evaluation.benchmark \
  --agents mock,responses,openai-sdk,langgraph \
  --model gpt-5.4
```

生成：

```text
reports/benchmark/benchmark.json
reports/benchmark/benchmark.md
```

Benchmark 对比字段：

```text
Pass Rate
Average Score
Passed Cases
Elapsed Time
Error
```

> Benchmark 会真实调用模型；多框架 × 多 Case 会产生多次 API 请求和费用。CI 默认只跑 Mock。

## 7. 测试 Case

`data/cases.jsonl` 每一行是一条 Case。当前有 6 条 Case，覆盖时间、地点、夜景、对象、排除自拍和审美排序。

示例：

```json
{
  "id": "case_001",
  "query": "找出去年在东京拍的夜景照片，排除自拍，然后选出最好看的三张。",
  "expected_tools": ["search_photos", "rank_photos"],
  "expected_arguments": {
    "search_photos": {
      "location": "Tokyo",
      "time": "last_year",
      "scene": "night",
      "exclude": ["selfie"]
    },
    "rank_photos": {
      "criterion": "aesthetic_quality",
      "limit": 3
    }
  },
  "expected_photo_count": 3
}
```

## 8. Bad Case 怎么定位

例如某个 Agent 错误输出 `rank_photos(limit=5)`，但用户要求三张，报告会同时暴露：

```text
Task Completion: FAIL
expected_photo_count=3, actual=5

Argument Correctness: FAIL
rank_photos.limit: expected=3, actual=5
```

因此可以区分：

| Metric | 主要定位层 |
|---|---|
| Task Completion | End-to-End 任务结果 |
| Tool Correctness | Planning / Tool Selection |
| Argument Correctness | Intent / Constraint / Tool Args |

## 9. 为什么 CI 只跑 Mock

GitHub Actions 只跑 `mock`：结果稳定、无需 Secret、不产生模型调用成本，适合作为 regression gate。真实 Agent Eval 和多框架 Benchmark 手动运行。
