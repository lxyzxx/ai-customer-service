const chat = document.querySelector("#chat");
const chatForm = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const statusEl = document.querySelector("#status");
const documentsEl = document.querySelector("#documents");
const docForm = document.querySelector("#doc-form");
const docTitle = document.querySelector("#doc-title");
const docContent = document.querySelector("#doc-content");

let sessionId = localStorage.getItem("ai-customer-service-session-id") || "";

function addMessage(role, content, sources = []) {
  const item = document.createElement("article");
  item.className = `message ${role}`;
  item.textContent = content;

  if (sources.length > 0) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    sources.forEach((source, index) => {
      const sourceItem = document.createElement("div");
      sourceItem.className = "source";
      const text = source.content.length > 120 ? `${source.content.slice(0, 120)}...` : source.content;
      sourceItem.textContent = `来源 ${index + 1}：${source.title}，相关度 ${source.score}。${text}`;
      sourceList.appendChild(sourceItem);
    });
    item.appendChild(sourceList);
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
    localStorage.setItem("ai-customer-service-session-id", sessionId);
    chat.lastElementChild.remove();
    addMessage("assistant", data.answer, data.sources);
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
addMessage("assistant", "你好，我可以根据知识库回答客服问题。");
