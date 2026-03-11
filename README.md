# 課程顧問 AI 助手 (Course Consultant)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.1%2B-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

一個基於檢索增強生成 (RAG) 的 AI 助手系統，幫助用戶快速找到課程相關信息、獎學金、助學措施等，並提供專業的顧問服務。

## 📋 功能特性

- **🔍 智能知識檢索**：使用向量數據庫 (Milvus) 存儲和檢索課程文檔、獎學金信息和行政公開資料
- **💬 AI 對話系統**：基於 Google Generative AI 的自然語言會話體驗
- **📝 對話歷史管理**：保持會話上下文，支持多輪對話交互
- **✂️ 智能文本處理**：自動分割大型文檔以優化檢索效果
- **🐳 容器化部署**：支持 Docker Compose 部署，易於擴展和維護
- **⚡ 高性能查詢**：基於向量相似度的快速檢索
- **🌐 Web 和 CLI 介面**：提供兩種使用方式

## 🏗️ 系統架構

該項目採用 **RAG (檢索增強生成)** 架構，通過將用戶問題與檢索到的相關文檔結合，生成更準確、有依據的回答：

```
用戶輸入問題
    ↓
[檢索階段] → Milvus 向量數據庫
    ↓
檢索相關課程/獎學金文檔
    ↓
[增強階段] → 組裝上下文信息
    ↓
[生成階段] → Google Generative AI
    ↓
返回專業的顧問級別回答
```

### 核心組件

| 組件 | 用途 | 技術選型 |
|------|------|--------|
| **Agent Graph** | 定義 AI 代理的執行流程和決策邏輯 | LangGraph |
| **Vector Store** | 存儲和檢索課程/獎學金文檔的向量表示 | Milvus + Sentence Transformers |
| **LLM** | 理解問題和生成回答 | Google Generative AI (Gemini) |
| **Message History** | 維持多輪對話的上下文 | LangChain Messages |
| **Web Server** | 提供 HTTP API 服務 | FastAPI + Uvicorn |
| **CLI Interface** | 命令行交互界面 | Click |

## 🛠️ 技術棧

| 層級 | 技術 | 版本 |
|------|------|------|
| **LLM 框架** | LangChain, LangGraph | 1.2.10+, 1.0.10+ |
| **向量數據庫** | Milvus | 2.6.8 |
| **對象存儲** | MinIO | 2024-12-18+ |
| **分佈式存儲** | etcd | 3.5.25 |
| **Web 框架** | FastAPI | 0.135.1+ |
| **嵌入模型** | Sentence Transformers | 5.2.3+ |
| **LLM 服務** | Google Generative AI | 1.66.0+ |
| **Python 版本** | Python | ≥ 3.11 |

## 📂 項目結構

```
course-consultant/
├── src/
│   ├── agent/              # AI 代理邏輯
│   │   └── ...
│   ├── db/                 # 數據庫操作
│   │   └── ...
│   ├── cli.py              # 命令行界面
│   └── server.py           # Web API 服務器
├── data/                   # 數據文件
│   ├── 114的課程JSON/      # 課程信息數據
│   └── 北科行政公開資料/    # 獎學金、助學金等行政資料
├── volumes/                # Docker 數據卷
│   ├── etcd/               # etcd 持久化存儲
│   ├── milvus/             # Milvus 持久化存儲
│   └── minio/              # MinIO 持久化存儲
├── docker-compose.yml      # Docker 服務配置
├── pyproject.toml          # 項目依賴配置
├── requirements.txt        # Python 依賴列表
├── Makefile                # 任務自動化
└── README.md               # 項目文檔
```

## 🚀 快速開始

### 前置要求

- Python 3.11 或更高版本
- Docker 和 Docker Compose
- Google Generative AI API 密鑰（免費獲取：https://aistudio.google.com/app/apikey）
- uv 包管理工具（推薦）或 pip

### 安裝步驟

1. **克隆或進入項目目錄**
   ```bash
   cd course-consultant
   ```

2. **設置 Python 虛擬環境**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # 或在 Windows 上：.venv\Scripts\activate
   ```

3. **安裝依賴**
   ```bash
   # 使用 uv（推薦）
   uv sync
   
   # 或使用 pip
   pip install -r requirements.txt
   ```

4. **設置環境變量**
   創建 `.env` 文件（或在終端設置）：
   ```bash
   export GOOGLE_API_KEY="your-google-generative-ai-key"
   ```

5. **啟動服務**
   
   - 啟動 Milvus、MinIO 和 etcd：
     ```bash
     docker-compose up -d
     ```
   
   - 初始化向量數據庫（首次運行）：
     ```bash
     make index
     ```
   
   - 運行 CLI 版本：
     ```bash
     make run
     ```
   
   - 或運行 Web 服務：
     ```bash
     PYTHONPATH=src uv run uvicorn server:app --reload
     ```

## 📖 使用方法

### CLI 模式

運行 CLI 進行交互式對話：

```bash
make run
```

然後輸入你的問題，例如：
```
❯ 請問北科有哪些獎學金可以申請？
```

AI 將基於檢索到的行政資料提供詳細的回答。

### Web API 模式

啟動服務後，訪問 API 端點：

```bash
# 健康檢查
curl http://localhost:8000/health

# 發送查詢（示例）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "請問如何申請助學金？", "session_id": "user_123"}'
```

詳見 API 文檔：http://localhost:8000/docs

## ⚙️ 配置說明

### 環境變量

| 變數 | 說明 | 必需 | 默認值 |
|------|------|------|--------|
| `GOOGLE_API_KEY` | Google Generative AI API 密鑰 | ✓ | 無 |
| `MILVUS_HOST` | Milvus 服務器地址 | ✗ | `localhost` |
| `MILVUS_PORT` | Milvus 服務器端口 | ✗ | `19530` |
| `MINIO_HOST` | MinIO 服務器地址 | ✗ | `localhost` |
| `MINIO_PORT` | MinIO 服務器端口 | ✗ | `9000` |
| `LOG_LEVEL` | 日誌級別 | ✗ | `INFO` |

### Docker Compose 服務

- **etcd**：分佈式 KV 存儲，為 Milvus 提供支持
- **minio**：對象存儲，用於存儲大型文件
- **milvus**：向量數據庫，用於語義搜索

## 📚 數據源

項目使用以下數據源：

1. **課程信息** (`data/114的課程JSON/`)
   - 北科大學課程目錄和詳細信息

2. **獎學金和助學金** (`data/北科行政公開資料/`)
   - 校內獎學金計劃
   - 教育部獎學金
   - 社會人士捐贈獎學金
   - 助學措施和生活補助

## 🔄 工作流程

### 文檔索引

1. 將文檔放入 `data/` 目錄
2. 運行 `make index` 進行向量化和索引
3. 文檔被分割成小塊（chunks）
4. 每個塊被轉換為向量並存儲在 Milvus 中

### 查詢流程

1. 用戶輸入問題
2. 問題被轉換為向量
3. 檢索與問題最相似的文檔塊（Top-K）
4. 將相似文檔和問題發送給 LLM
5. LLM 生成基於檢索信息的回答

## 🐛 故障排查

### Milvus 連接失敗
```bash
# 檢查容器是否運行
docker-compose ps

# 查看 Milvus 日誌
docker-compose logs milvus

# 重啟服務
docker-compose restart milvus
```

### 向量索引為空
```bash
# 確認數據文件存在
ls -la data/

# 重新初始化索引
make index
```

### Google API 認證失敗
- 檢查 API 密鑰是否正確設置
- 確認 API 金鑰未過期
- 在 Google Cloud 控制台檢查 API 配額

## 📦 依賴項

詳見 [pyproject.toml](pyproject.toml) 和 [requirements.txt](requirements.txt)

核心依賴：
- `langchain` - LLM 應用框架
- `langgraph` - 代理工作流編排
- `langchain-milvus` - Milvus 集成
- `langchain-google-genai` - Google AI 集成
- `fastapi` - Web 框架
- `sentence-transformers` - 嵌入模型

## ❓ 常見問題 (FAQ)

**Q: 如何更新向量數據庫中的信息？**
A: 
1. 更新 `data/` 目錄中的文件
2. 運行 `make index` 重新索引
3. Milvus 將自動使用新的向量進行搜索

**Q: 可以使用其他 LLM 模型嗎？**
A: 可以。項目使用 LangChain，支持多種 LLM。修改 `src/agent/` 中的配置即可切換到其他模型（如 OpenAI GPT、Anthropic 等）。

**Q: 向量檢索的精度如何提高？**
A: 
- 優化文檔分割的塊大小（chunk_size）
- 使用更好的嵌入模型
- 調整檢索的 Top-K 參數
- 改進文檔的預處理和清理

**Q: 支持多語言嗎？**
A: Sentence Transformers 支持 100+ 語言。當前系統用繁體中文配置，可通過修改嵌入模型輕鬆支持其他語言。

**Q: 如何部署到生產環境？**
A: 
1. 使用 Docker 容器化所有服務
2. 配置持久化卷用於數據持久化
3. 設置環境變數和安全認證
4. 使用 Kubernetes 或其他編排工具進行高可用部署

## 🔐 安全性考慮

- **API 密鑰**：將敏感密鑰存儲在環境變數或密鑰管理服務中，不要提交到版本控制
- **數據隱私**：確保用戶數據符合隱私法規（GDPR、地方法律等）
- **訪問控制**：在生產環境中向 API 端點添加認證機制
- **速率限制**：實施速率限制以防止濫用

## 📊 性能優化建議

1. **向量檢索優化**
   - 使用 Milvus 的索引優化功能
   - 調整 ef 參數以平衡精度和速度
   - 定期重新索引數據

2. **LLM 調用優化**
   - 實施結果緩存策略
   - 批量處理相似查詢
   - 監控 API 使用成本

3. **基礎設施優化**
   - 配置適當的 GPU 資源
   - 使用 CDN 分發靜態內容
   - 實施負載均衡

## 📚 參考資源

- [LangChain 文檔](https://python.langchain.com/)
- [LangGraph 文檔](https://langchain-ai.github.io/langgraph/)
- [Milvus 文檔](https://milvus.io/)
- [Google Generative AI API](https://ai.google.dev/)
- [FastAPI 文檔](https://fastapi.tiangolo.com/)

## 🤝 貢獻指南

歡迎貢獻！以下是貢獻流程：

1. **Fork 項目**
   ```bash
   git clone <repo-url>
   cd course-consultant
   ```

2. **創建特性分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **提交更改**
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

4. **推送到分支**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **開啟 Pull Request**
   - 清楚描述你的更改
   - 包括相關的 issue 編號
   - 附加測試用例和文檔更新

### 代碼風格

- 遵循 [PEP 8](https://pep8.org/) Python 風格指南
- 使用類型提示進行函數簽名
- 為複雜邏輯添加文檔字符串

## 📝 許可證

該項目採用 MIT 許可證。詳見 [LICENSE](LICENSE) 文件。

## 👨‍💻 作者和維護者

- 項目維護者：[你的名字]
- 聯繫方式：[你的郵箱]

## 🔗 相關項目

- [LangChain](https://github.com/langchain-ai/langchain)
- [Milvus](https://github.com/milvus-io/milvus)
- [MinIO](https://github.com/minio/minio)

## 📌 更新日誌

### v0.1.0 (2026-03-11)
- 初始版本發佈
- 實現基於 RAG 的課程顧問助手
- 支持 CLI 和 Web API 兩種交互方式
- 集成 Milvus、MinIO 和 etcd 服務

---

**最後更新於：** 2026 年 3 月 11 日

有問題或想要提出功能建議？歡迎開啟 [GitHub Issue](https://github.com/your-username/course-consultant/issues)！

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
