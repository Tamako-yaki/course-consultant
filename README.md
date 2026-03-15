# 智慧創新大賞 - Advanced RAG Demo

## 快速啟動

1. 建立並啟用虛擬環境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. 安裝依賴

```bash
pip install -r requirements.txt
```

3. 設定環境變數

```bash
cp .env.example .env
```

在 `.env` 中設定：

```env
API_KEY=your_google_ai_api_key
OLLAMA_BASE_URL=http://localhost:11434
RAG_ENGINE=google
```

4. 建立向量索引（首次或資料更新後）

```bash
python src/rag_ingestion.py
```

5. 啟動 Web UI

```bash
python rag_demo_app.py
```

6. （選用）啟動 Webhook 服務

```bash
python server/webhook_server.py
```

## Ollama 本地 Embedding 建置（替代方案）

如果你想改用本地 embedding（不使用 Google embedding）建立 FAISS 索引：

1. 啟動 Ollama 服務並拉取 embedding 模型

```bash
ollama pull nomic-embed-text-v2-moe
```

2. 在 `.env` 中設定 Ollama 服務位置（`OLLAMA_BASE_URL`）

3. 執行替代 ingestion

```bash
python src/rag_ingestion_ollama.py
```

預設會輸出到 `data/faiss_index/pure_rag_ollama`。
此腳本的 embedding 模型固定為 `nomic-embed-text-v2-moe`。

## 引擎切換（Google Embedding / Ollama Embedding）

Web UI 會依照 `.env` 的 `RAG_ENGINE` 載入不同引擎：

- `RAG_ENGINE=google`: 使用 `src/rag_engine.py`（Google embedding + Gemini）
- `RAG_ENGINE=ollama`: 使用 `src/rag_engine_ollama.py`（Ollama embedding + Gemini）

切到 Ollama embedding 時，請確認：

1. `OLLAMA_BASE_URL` 已正確設定
2. 已先執行 `python src/rag_ingestion_ollama.py` 建立索引

## 目前分支的服務模式

- Web UI: Flask + Jinja 模板（`rag_demo_app.py` + `templates/index.html`）
- Webhook: 獨立 Flask 路由（`server/webhook_server.py`）
- RAG 引擎: `src/rag_engine.py`
- 索引建置: `src/rag_ingestion.py`
