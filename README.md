# Image-Evaluation

一个面试向的 **Photo Agent Evaluation MVP**。目标不是做完整相册产品，而是展示一套可解释、可回归的 Agent 质量评测闭环：

```text
Test Cases -> Agent -> Tool Calls / Result -> Evaluators -> Bad Cases / Report
```

V2 同时支持两种 Agent：

- `mock`：确定性 Mock Agent，零 API 成本，用于 CI 和框架回归。
- `llm`：真实 LLM Function Calling Agent，用于测试意图理解、Tool Selection、参数生成和多步工具调用。

> `llm` 模式里的 LLM 是真实的；`search_photos` / `rank_photos` 的后端目前仍是确定性 Stub。这样可以把评测重点放在 Agent 行为上，而不是先搭完整手机相册服务。

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
               +-------------------+
               |   Agent Runner    |
               +---------+---------+
                         |
              +----------+----------+
              |                     |
              v                     v
      MockPhotoAgent          LLMPhotoAgent
                                  |
                           Responses API
                                  |
                       Function Calling Loop
                                  |
                      +-----------+-----------+
                      |                       |
               search_photos             rank_photos
                 (Stub)                    (Stub)
                      +-----------+-----------+
                                  |
                                  v
                             AgentResult
                                  |
                                  v
                Task / Tool / Argument Evaluators
                                  |
                                  v
                         report.json / report.md
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

只要未来的 LangGraph / Dify / 公司内部 Agent 能转换成这个结构，就可以继续复用现有 Evaluator。

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

## 5. 运行真实 LLM Agent

配置 API Key：

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-5.6"   # 可替换成你可用的模型
```

然后运行：

```bash
PYTHONPATH=src python -m image_evaluation.runner --agent llm
```

也可以单次覆盖模型：

```bash
PYTHONPATH=src python -m image_evaluation.runner --agent llm --model gpt-5.6
```

LLM 模式会真实执行：

```text
User Query
   -> LLM
   -> search_photos Function Call
   -> Tool Result
   -> LLM
   -> rank_photos Function Call（如需要）
   -> Tool Result
   -> Final Answer
```

每次模型实际生成的 Tool Name 和 Arguments 都会被记录，并与 `cases.jsonl` 中的 Ground Truth 比较。

## 6. 测试 Case

`data/cases.jsonl` 每一行是一条 Case。当前有 6 条 Case，覆盖时间、地点、夜景、对象、排除自拍和审美排序。

## 7. Bad Case 怎么定位

例如真实 Agent 错误输出 `rank_photos(limit=5)`，但用户要求三张，报告会同时暴露：

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

## 8. 为什么 CI 只跑 Mock

GitHub Actions 只跑 `mock`：结果稳定、无需 Secret、不产生模型调用成本，适合作为 regression gate。真实 LLM Eval 更适合手动运行，或者以后在单独的 nightly / release workflow 中运行。

## 9. 面试怎么讲

> 我搭了一个 Photo Agent Evaluation Harness。测试集定义 query、expected tool trajectory 和 tool arguments；Runner 可以切换 deterministic mock Agent 或真实 LLM Function Calling Agent；Evaluator 分别从 Task Completion、Tool Correctness 和 Argument Correctness 做自动评分，失败 Case 会输出具体的错误参数用于 badcase 定位。CI 使用 mock 做稳定回归，真实 LLM Eval 用于评估模型/Prompt 变更。

## 10. 下一步扩展

1. 增加真实图片与 `AestheticJudge`
2. Exposure / Color / Sharpness / Noise / Dynamic Range / Composition 影像维度
3. 按 `tags` 做 Slice Analysis
4. old vs new Agent Regression Diff
5. LLM-as-a-Judge / Multimodal Judge
6. LangGraph Agent Adapter
7. Classification / Detection / Segmentation 指标
8. FiftyOne Badcase 可视化
