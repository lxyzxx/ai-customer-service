# AI 智能客服 MVP

这是一个适合面试展示和后续 GitHub 维护的最小版 AI 智能客服项目。它使用问题分层 + Agentic RAG / Direct Corpus Interaction 风格的检索作为核心实现方式，覆盖智能客服的核心闭环：问题路由、知识库入库、文本切分、直接语料检索、上下文核验、模型回答、来源追踪和会话记录。

当前版本刻意保持零外部运行依赖，方便快速启动和讲清楚核心链路。后续可以逐步替换为 FastAPI、全文检索引擎、向量召回补充通道和生产级大模型网关。

## 功能

- 文档入库：支持粘贴 FAQ、产品说明、售后政策等文本
- 文本切分：按段落优先，长文本使用滑动窗口
- 问题分层：先判断问题应走寒暄、确定性规则、业务工具查询还是知识库证据检索
- 直接语料检索：从用户问题抽取精确线索，在原始知识库片段中搜索命中项
- 弱线索补充：使用 TF-IDF 作为辅助召回，不把系统限制在单次语义 top-k
- 上下文核验：命中后读取同一文档相邻片段，避免只看孤立 chunk
- Agentic RAG 问答：基于检索证据、原文上下文和会话历史构造回答
- 来源追踪：回答下方展示命中文档、证据线索、相关度和上下文
- 会话记录：SQLite 保存多轮对话
- 模型适配：配置 `OPENAI_API_KEY` 后可调用 OpenAI-compatible Chat Completions 接口；未配置时返回检索摘要

## 快速开始

```bash
cd ai-customer-service
python3 -m app.server
```

浏览器打开：

```text
http://127.0.0.1:8000
```

首次启动会自动导入 `data/knowledge/sample_faq.md` 作为示例知识库。

## 配置模型

复制环境变量模板：

```bash
cp .env.example .env
```

根据你使用的模型服务设置：

```bash
export OPENAI_API_KEY="你的 API Key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```

如果使用其他兼容 OpenAI Chat Completions 的服务，只需要修改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。

## API

### 健康检查

```http
GET /api/health
```

### 新增知识库文档

```http
POST /api/documents
Content-Type: application/json

{
  "title": "售后政策",
  "content": "用户签收商品后的 7 天内..."
}
```

### 查看文档

```http
GET /api/documents
```

### 客服问答

```http
POST /api/chat
Content-Type: application/json

{
  "question": "退款多久到账？",
  "session_id": "可选，会话 ID"
}
```

## 项目结构

```text
ai-customer-service/
  app/
    chunker.py      # 文本切分
    problem_layers.py # 问题分层：chatbot / rule_engine / tool_call / dci_retrieval
    retriever.py    # DCI 风格检索：精确线索搜索 + TF-IDF 弱线索 + 上下文读取
    llm.py          # 模型调用和 fallback
    rag.py          # RAG 编排
    storage.py      # SQLite 持久化
    server.py       # HTTP API 和静态资源服务
  data/knowledge/   # 示例知识库
  static/           # 前端页面
  tests/            # 单元测试
```

## 测试

```bash
python3 -m unittest discover -s tests
```

## 面试讲解重点

可以按这条链路讲：

1. 用户上传知识库文档，系统将文本切分成 chunk。
2. 用户提问时，系统先做问题分层：普通寒暄走 chatbot，高风险或人工诉求走规则，订单/物流/退款进度走业务工具，政策/FAQ 证据问题走 DCI 检索。
3. 进入 DCI 检索后，系统从问题中抽取关键词、短语和数字等精确线索。
4. 检索器直接扫描原始知识库 chunk，找出命中线索的文档片段，相当于在应用内提供 `grep/read_context` 能力。
5. 对没有明显精确命中的情况，再用 TF-IDF 做弱线索补充，避免完全依赖关键词。
6. 命中片段后读取同一文档的相邻 chunk，检查原文上下文。
7. 模型只基于检索证据和上下文回答，并返回来源、证据线索和相关度。
8. 会话 ID 绑定历史消息，支持多轮上下文扩展。

可主动说明当前取舍：

- MVP 优先做可控召回和证据链，而不是一开始接 embedding + 向量数据库。
- 向量检索适合语义近似召回，但客服知识库经常需要精确条款、数字、地区、SKU、时间和例外条件。
- 单次 top-k 相似度召回如果漏掉关键条款，后面的模型很难补救；所以这里采用直接语料交互，让系统能先搜原文、再读上下文、最后回答。
- 生产环境可以把 embedding + 向量数据库作为辅助召回通道，但不让它成为唯一检索接口。
- 当前模型调用走 OpenAI-compatible 协议，方便切换不同供应商。

## 问题分层设计

这个项目不把所有客服问题都塞进 RAG，而是先做路由：

```text
普通寒暄
-> chatbot

投诉、赔偿、账号安全、金额异常、明确要求人工
-> rule_engine

查询订单、物流、退款进度、售后进度
-> tool_call

退款政策、发票规则、物流异常处理、售后条款、FAQ
-> dci_retrieval
```

这对应真实客服系统的边界：确定性问题走规则，实时状态走业务 API，证据查找走 DCI，泛化表达再交给模型组织语言。

## 为什么不是纯向量检索

这个项目的核心观点不是“向量检索没用”，而是“复杂客服 Agent 不应该只有一个固定相似度 top-k 接口”。

传统向量 RAG 的链路通常是：

```text
用户问题 -> embedding -> 向量库 top-k -> LLM 回答
```

这个链路很快，也适合语义近似问题。但客服场景里，很多问题更像证据定位：

- 退款到账时间是否受支付渠道影响
- 定制商品、拆封商品、生鲜商品是否适用 7 天无理由
- 某地区是否支持配送或上门取件
- 订单超过 48 小时未揽收时应该如何处理
- 某个活动规则、SKU、型号或日期是否命中特殊条款

这些问题需要精确搜索、组合多个稀疏线索、查看原文上下文和保留证据链。Direct Corpus Interaction 的做法是让系统直接和原始知识库交互：

```text
用户问题
-> 抽取精确线索
-> 搜索原始知识库
-> 读取命中片段的上下文
-> 用弱线索召回补充
-> 基于证据回答并展示来源
```

因此，本项目的定位是可审计的 Agentic 客服检索系统。向量召回可以加入，但它是工具之一，不是架构中心。

## 后续路线

- 后端替换为 FastAPI，增加 OpenAPI 文档和鉴权
- 检索层增加 BM25 / SQLite FTS / Elasticsearch，提高大规模全文检索能力
- 增加可选向量召回通道，与精确线索检索做融合排序
- 增加多轮检索计划：先查精确词，再查别名和同义词，再读上下文
- 增加检索轨迹面板，展示系统如何搜索、命中和核验证据
- 文档解析支持 PDF、Word、网页和 Markdown 批量导入
- 增加人工转接、工单流转、敏感词过滤和回答反馈
- 增加管理端：知识库版本、命中率、无答案问题、客服质检
