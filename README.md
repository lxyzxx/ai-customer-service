# AI 智能客服 MVP

这是一个适合面试展示和后续 GitHub 维护的最小版 AI 智能客服项目。它使用 RAG 作为核心实现方式，覆盖智能客服的核心闭环：知识库入库、文本切分、相关片段检索、模型回答、来源追踪和会话记录。

当前版本刻意保持零外部运行依赖，方便快速启动和讲清楚核心链路。后续可以逐步替换为 FastAPI、向量数据库和生产级大模型网关。

## 功能

- 文档入库：支持粘贴 FAQ、产品说明、售后政策等文本
- 文本切分：按段落优先，长文本使用滑动窗口
- 检索召回：基于 TF-IDF 和余弦相似度，支持中英文 token
- RAG 问答：召回知识片段后构造上下文回答
- 来源追踪：回答下方展示命中文档、片段和相关度
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
    retriever.py    # TF-IDF 检索
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
2. 用户提问时，对问题和 chunk 做 token 化，并用 TF-IDF 计算相似度。
3. 取 Top-K 相关片段作为上下文，拼接进模型 prompt。
4. 模型只基于上下文回答，并返回来源信息。
5. 会话 ID 绑定历史消息，支持多轮上下文扩展。

可主动说明当前取舍：

- MVP 使用 TF-IDF 是为了降低部署门槛，适合小规模 FAQ。
- 生产环境可以替换为 embedding + 向量数据库，提高语义召回能力。
- 当前模型调用走 OpenAI-compatible 协议，方便切换不同供应商。

## 后续路线

- 后端替换为 FastAPI，增加 OpenAPI 文档和鉴权
- 检索层接入 FAISS、Milvus、pgvector 或 Elasticsearch
- 文档解析支持 PDF、Word、网页和 Markdown 批量导入
- 增加人工转接、工单流转、敏感词过滤和回答反馈
- 增加管理端：知识库版本、命中率、无答案问题、客服质检
