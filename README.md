# 課程顧問 AI 助手 (Course Consultant)

一個基於檢索增強生成 (RAG) 的 AI 助手系統，幫助用戶快速找到課程相關信息並提供專業的顧問服務。

## 📋 功能特性

- **智能知識檢索**：使用向量數據庫 (Milvus) 存儲和檢索課程文檔
- **AI 對話系統**：基於 Google Generative AI 的自然語言對話
- **對話歷史管理**：保持會話上下文，支持多輪對話
- **文本分割處理**：智能分割大型文檔以優化檢索效果
- **分佈式架構**：支持容器化部署，易於擴展

## 🏗️ 系統架構

該項目採用 **RAG (检索增强生成)** 架構：

```
用戶輸入問題
    ↓
[檢索階段] → Milvus 向量數據庫
    ↓
檢索相關文檔
    ↓
[生成階段] → Google Generative AI
    ↓
返回答案給用戶
```

### 核心組件

| 組件 | 功能 | 技術棧 |
|------|------|--------|
| **Agent Graph** | 定義 AI 代理的執行流程 | LangGraph |
| **Vector Store** | 存儲課程文檔的向量表示 | Milvus + Sentence Transformers |
| **LLM** | 文本生成和理解 | Google Generative AI |
| **Message History** | 保持對話上下文 | LangChain Messages |

## 🛠️ 技術棧

- **AI Framework**: LangChain, LangGraph
- **Vector Database**: Milvus (v2.6.8)
- **Object Storage**: MinIO
- **Distributed Storage**: etcd
- **LLM**: Google Generative AI
- **Embeddings**: Sentence Transformers
- **Python**: ≥ 3.11

## 📦 依賴項

詳見 [pyproject.toml](pyproject.toml)

主要依賴：
- langchain >= 1.2.10
- langchain-google-genai >= 4.2.1
- langchain-milvus >= 0.3.3
- langgraph >= 1.0.10
- sentence-transformers >= 5.2.3

## 🚀 快速開始

### 前置要求

- Docker & Docker Compose
- Python 3.11+
- Google API Key (用於 Generative AI)

### 1. 克隆專案

```bash
git clone <repository-url>
cd course-consultant
```

### 2. 設定環境變量

```bash
# 創建 .env 文件中配置你的 Google API Key
export GOOGLE_API_KEY="your-api-key-here"
```

### 3. 啟動依賴服務

```bash
docker-compose up -d
```

此命令將啟動：
- **Milvus** (向量數據庫) - 端口 19530
- **MinIO** (對象存儲) - 端口 9000, 9001
- **etcd** (分佈式存儲) - 端口 2379

### 4. 安裝 Python 依賴

```bash
# 創建虛擬環境
python -m venv .venv
source .venv/bin/activate  # 在 Windows 上使用: .venv\Scripts\activate

# 安裝依賴
pip install -e .
```

### 5. 準備向量數據庫

索引課程文檔到 Milvus：

```bash
python src/db/vector/index_documents.py
```

### 6. 運行 CLI 應用

```bash
python src/cli.py
```

然後開始與 AI 助手對話：

```
請輸入您的問題 (或輸入 'exit' 退出): 哪些課程適合初學者？
AI助手: [AI 回答]
```

## 📁 項目結構

```
course-consultant/
├── src/
│   ├── agent/              # AI Agent 核心
│   │   ├── graph.py        # Agent 執行流程定義
│   │   ├── nodes.py        # 檢索和生成節點
│   │   ├── state.py        # 狀態管理
│   │   ├── prompts.py      # Prompt 模板
│   │   └── configuration.py # 配置管理
│   ├── db/
│   │   ├── vector/         # 向量數據庫相關
│   │   │   ├── store.py    # Milvus 存儲操作
│   │   │   ├── embedding.py # 嵌入向量生成
│   │   │   └── index_documents.py # 文檔索引
│   ├── services/           # 業務服務層
│   └── cli.py              # 命令行界面
├── main.py                 # 主入口
├── docker-compose.yml      # Docker 服務配置
├── pyproject.toml          # Python 項目配置
└── README.md               # 本文件
```

## 🔧 配置

### Agent 配置 (src/agent/configuration.py)

```python
class Configuration:
    generate_model = "gemini-2.0-flash"  # LLM 模型選擇
```

### Milvus 連接配置 (src/db/vector/store.py)

默認連接到本地 Milvus 服務 (localhost:19530)

## 💾 數據流程

### 1. 文檔索引 (初始化)

```
原始文檔 → 文本分割 → 生成嵌入向量 → 存儲到 Milvus
```

### 2. 問答流程 (運行時)

```
用戶問題 → 嵌入向量 → 檢索相似文檔 → 構建 Prompt → LLM 生成答案 → 返回用戶
```

## 🎯 使用示例

```python
from src.cli import main

# 運行 CLI
main()

# 交互式對話示例
# 請輸入您的問題 (或輸入 'exit' 退出): 什麼是機器學習？
# AI助手: 機器學習是...
```

## 📊 性能優化

- **文本分割優化**：使用 `chunk_size=1024` 和 `chunk_overlap=256` 平衡檢索精度
- **向量數據庫**：Milvus 支持高效的相似度搜索和擴展
- **對話管理**：使用 LangGraph 的檢查點機制保存會話狀態

## 🐳 Docker 相關命令

```bash
# 查看服務狀態
docker-compose ps

# 檢查 Milvus 健康狀態
docker-compose exec standalone curl http://localhost:9091/healthz

# 停止服務
docker-compose down

# 清理所有數據並重新啟動
docker-compose down -v
docker-compose up -d
```

## 🔐 安全建議

- 使用 `.env` 文件管理敏感信息 (API Keys)
- 定期輪換 Google API Key
- 在生產環境中使用環境變量而不是硬編碼敏感信息
- 限制 MinIO 的默認憑據 (minioadmin/minioadmin)

## 🐛 故障排除

### Milvus 連接失敗
```bash
# 檢查容器是否運行
docker-compose ps

# 查看 Milvus 日誌
docker-compose logs standalone
```

### 無法檢索文檔
- 確認文檔已通過 `index_documents.py` 索引
- 檢查 Milvus 健康狀態
- 驗證向量相似度檢索參數

### API 配額限制
- 檢查 Google API 配額使用情況
- 考慮選擇不同的 LLM 提供商

## 📝 開發指南

### 添加新的 Agent 節點

```python
# 在 src/agent/nodes.py 中定義新節點
def new_node(state: AgentState):
    # 處理邏輯
    return {"new_field": value}

# 在 src/agent/graph.py 中添加到流程
builder.add_node("new_node", new_node)
builder.add_edge("previous_node", "new_node")
```

### 修改 Prompt

編輯 [src/agent/prompts.py](src/agent/prompts.py) 中的 `GENERATE_PROMPT` 變量

## 🤝 貢獻指南

欢迎提交 Issue 和 Pull Request！

## 📄 許可

MIT License - 查看 LICENSE 文件获取詳細信息

## 📞 聯絡方式

有問題？請提交 GitHub Issue 或聯絡開發者。

---

**最後更新**: 2026年3月
