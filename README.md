# Internal QA Bot

这是一个面向公司内部落地的知识问答机器人。系统先判断问题应该走普通对话、确定性规则、业务系统查询还是知识库证据检索，再把请求交给对应的处理链路。

知识库检索采用 Agentic RAG / Direct Corpus Interaction 风格：先搜索原始语料和精确线索，再融合 TF-IDF 与向量语义召回，最后读取原文上下文并基于证据回答。

## 功能

- 文档入库：支持粘贴 FAQ、制度说明、流程文档、操作手册等文本
- 文本切分：按段落优先，长文本使用滑动窗口
- 问题分层：先判断问题应走寒暄、确定性规则、业务系统查询还是知识库证据检索
- Chatbot 内置知识：普通帮助类问题使用轻量向量召回匹配机器人能力说明
- 直接语料检索：从用户问题抽取精确线索，在原始知识库片段中搜索命中项
- 弱线索补充：使用 TF-IDF 作为辅助召回，不把系统限制在单次语义 top-k
- 向量辅助召回：在知识库证据层补充语义近似匹配，适合口语化和同义表达
- 上下文核验：命中后读取同一文档相邻片段，避免只看孤立 chunk
- Agentic RAG 问答：基于检索证据、原文上下文和会话历史构造回答
- 来源追踪：回答下方展示命中文档、证据线索、相关度和上下文
- 会话记录：SQLite 保存多轮对话
- 模型适配：通过 OpenAI Python SDK 调用 OpenAI-compatible Chat Completions 接口；未配置 `OPENAI_API_KEY` 时返回检索摘要

## 快速开始

```bash
cd internal-qa-bot
python3 -m pip install -e .
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

```env
OPENAI_API_KEY=你的 API Key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

启动时会自动读取项目根目录下的 `.env`。真实 `.env` 已被 `.gitignore` 忽略，不要提交到 GitHub；提交 `.env.example` 即可。系统通过 OpenAI Python SDK 发起请求。如果使用其他兼容 OpenAI Chat Completions 的服务，只需要修改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。

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
  "title": "差旅报销制度",
  "content": "员工完成出差后，需要在 10 个工作日内提交差旅报销申请..."
}
```

### 查看文档

```http
GET /api/documents
```

### 问答

```http
POST /api/chat
Content-Type: application/json

{
  "question": "差旅报销需要哪些材料？",
  "session_id": "可选，会话 ID"
}
```

## 项目结构

```text
internal-qa-bot/
  app/
    chatbot.py      # Chatbot 层：轻量向量召回内置说明
    chunker.py      # 文本切分
    problem_layers.py # 问题分层：chatbot / rule_engine / tool_call / dci_retrieval
    retriever.py    # 证据检索：DCI 精确线索 + TF-IDF + vector recall + 上下文读取
    vector_retriever.py # 零依赖 hash vector recall，可替换为 embedding/向量库
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

## 工作流程

1. 管理员上传知识库文档，系统将文本切分为 chunk。
2. 用户提问时，系统先做问题分层：普通寒暄和帮助问题走 chatbot，高风险或负责人介入诉求走规则，审批/工单/报销/考勤等状态问题走业务工具，制度/流程/FAQ 证据问题走知识库检索。
3. Chatbot 层使用轻量向量召回匹配能力说明、使用方式和证据优先原则。
4. 进入知识库证据层后，系统从问题中抽取关键词、短语和数字等精确线索。
5. 检索器直接扫描原始知识库 chunk，找出命中线索的文档片段，相当于在应用内提供 `grep/read_context` 能力。
6. 系统再融合 TF-IDF 弱线索和向量语义召回，补足口语化、同义表达和措辞差异。
7. 命中片段后读取同一文档的相邻 chunk，检查原文上下文。
8. 模型只基于检索证据和上下文回答，并返回来源、证据线索和相关度。
9. 会话 ID 绑定历史消息，支持多轮上下文扩展。

## 检索设计

- 系统优先保证可控召回和证据链，而不是只依赖 embedding + 向量数据库。
- 向量检索适合语义近似召回，但内部制度和流程经常需要精确条款、数字、角色、系统名称、有效期和例外条件。
- 单次 top-k 相似度召回如果漏掉关键条款，后面的模型很难补救；所以这里采用直接语料交互，让系统能先搜原文、再读上下文、最后回答。
- embedding + 向量数据库适合作为辅助召回通道，但不作为唯一检索接口。
- 当前模型调用走 OpenAI-compatible 协议，方便切换不同供应商。

## 问题分层设计

这个项目不把所有内部问题都塞进 RAG，而是先做路由：

```text
普通寒暄、能力说明、帮助
-> chatbot + builtin knowledge recall

数据泄露、账号安全、合规风险、金额异常、明确要求负责人介入
-> rule_engine

查询审批、工单、报销、假期余额、考勤、权限申请状态
-> tool_call

制度说明、流程规范、系统操作手册、FAQ
-> dci_retrieval + tfidf + vector recall
```

这对应真实内部问答系统的边界：确定性高风险问题走规则，实时状态走业务 API，泛化帮助走 chatbot 内置知识，证据查找走 DCI + 向量辅助召回，最后再交给模型组织语言。

## 为什么不是纯向量检索

这个项目的核心观点不是“向量检索没用”，而是“复杂问答机器人不应该只有一个固定相似度 top-k 接口”。

传统向量 RAG 的链路通常是：

```text
用户问题 -> embedding -> 向量库 top-k -> LLM 回答
```

这个链路很快，也适合语义近似问题。但内部知识问答里，很多问题更像证据定位：

- 差旅报销需要哪些材料
- 生产系统权限是否需要填写有效期
- 病假是否可以返岗后补交证明
- 某个系统是否必须先开通 VPN
- 会议室预约是否有人数或提前时间要求

这些问题需要精确搜索、组合多个稀疏线索、查看原文上下文和保留证据链。Direct Corpus Interaction 的做法是让系统直接和原始知识库交互：

```text
用户问题
-> 抽取精确线索
-> 搜索原始知识库
-> 用 TF-IDF 弱线索补充
-> 融合向量语义召回
-> 读取命中片段的上下文
-> 基于证据回答并展示来源
```

因此，本项目的定位是可审计的内部知识问答机器人。向量召回已经作为辅助通道接入 chatbot 和知识库证据层，但它是工具之一，不是架构中心。

## 后续路线

- 后端替换为 FastAPI，增加 OpenAPI 文档和鉴权
- 检索层增加 BM25 / SQLite FTS / Elasticsearch，提高大规模全文检索能力
- 将当前零依赖 hash vector recall 替换为 embedding 模型 + FAISS、Milvus 或 pgvector
- 增加多轮检索计划：先查精确词，再查别名和同义词，再读上下文
- 增加检索轨迹面板，展示系统如何搜索、命中和核验证据
- 文档解析支持 PDF、Word、网页和 Markdown 批量导入
- 接入 OA、ITSM、HR、财务等内部系统工具
- 增加管理端：知识库版本、命中率、无答案问题、回答反馈和质检
