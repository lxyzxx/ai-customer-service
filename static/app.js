const chat = document.querySelector("#chat");
const chatForm = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const statusEl = document.querySelector("#status");
const documentsEl = document.querySelector("#documents");
const docForm = document.querySelector("#doc-form");
const docTitle = document.querySelector("#doc-title");
const docContent = document.querySelector("#doc-content");

let sessionId = localStorage.getItem("ai-qa-bot-session-id") || "";

function addMessage(role, content, sources = [], route = null, chatbotKnowledge = []) {
  const item = document.createElement("article");
  item.className = `message ${role}`;
  item.textContent = content;

  if (route) {
    const routeItem = document.createElement("div");
    routeItem.className = "route";
    routeItem.textContent = `分层：${route.layer} / ${route.handler}。${route.reason}`;
    item.appendChild(routeItem);
  }

  if (sources.length > 0) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    sources.forEach((source, index) => {
      const sourceItem = document.createElement("div");
      sourceItem.className = "source";
      const sourceText = source.context || source.content;
      const text = sourceText.length > 140 ? `${sourceText.slice(0, 140)}...` : sourceText;
      const evidence =
        source.evidence && source.evidence.length > 0
          ? source.evidence.join("；")
          : "相关片段召回";
      sourceItem.textContent =
        `来源 ${index + 1}：${source.title}，证据：${evidence}，` +
        `相关度 ${source.score}。${text}`;
      sourceList.appendChild(sourceItem);
    });
    item.appendChild(sourceList);
  }

  if (chatbotKnowledge.length > 0) {
    const knowledgeList = document.createElement("div");
    knowledgeList.className = "sources";
    chatbotKnowledge.forEach((knowledge) => {
      const knowledgeItem = document.createElement("div");
      knowledgeItem.className = "source";
      knowledgeItem.textContent =
        `聊天知识：${knowledge.title}，证据：${knowledge.evidence}，` +
        `相关度 ${knowledge.score}。${knowledge.content}`;
      knowledgeList.appendChild(knowledgeItem);
    });
    item.appendChild(knowledgeList);
  }

  chat.appendChild(item);
  chat.scrollTop = chat.scrollHeight;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

async function loadDocuments() {
  const data = await requestJson("/api/documents");
  documentsEl.innerHTML = "";

  if (data.documents.length === 0) {
    documentsEl.innerHTML = '<div class="empty">暂无文档</div>';
    return;
  }

  data.documents.forEach((doc) => {
    const item = document.createElement("div");
    item.className = "doc-item";
    item.innerHTML = `
      <div class="doc-title"></div>
      <div class="doc-meta">${doc.chunk_count} 个片段 · ${doc.created_at}</div>
    `;
    item.querySelector(".doc-title").textContent = doc.title;
    documentsEl.appendChild(item);
  });
}

async function checkHealth() {
  await requestJson("/api/health");
  statusEl.textContent = "已连接";
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  questionInput.value = "";
  addMessage("user", question);
  addMessage("assistant", "正在检索知识库...");

  try {
    const data = await requestJson("/api/chat", {
      method: "POST",
      body: JSON.stringify({ question, session_id: sessionId || undefined }),
    });
    sessionId = data.session_id;
    localStorage.setItem("ai-qa-bot-session-id", sessionId);
    chat.lastElementChild.remove();
    const chatbotKnowledge = data.chatbot_knowledge || [];
    addMessage("assistant", data.answer, data.sources, data.route, chatbotKnowledge);
  } catch (error) {
    chat.lastElementChild.remove();
    addMessage("assistant", `请求失败：${error.message}`);
  }
});

docForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const title = docTitle.value.trim();
  const content = docContent.value.trim();
  if (!title || !content) return;

  try {
    await requestJson("/api/documents", {
      method: "POST",
      body: JSON.stringify({ title, content }),
    });
    docTitle.value = "";
    docContent.value = "";
    await loadDocuments();
  } catch (error) {
    alert(`入库失败：${error.message}`);
  }
});

checkHealth().catch(() => {
  statusEl.textContent = "未连接";
});
loadDocuments().catch(() => {
  documentsEl.innerHTML = '<div class="empty">加载失败</div>';
});
addMessage("assistant", "你好，我会先检索知识库原文并核验上下文，再回答内部问题。");
