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

## 目前分支的服務模式

- Web UI: Flask + Jinja 模板（`rag_demo_app.py` + `templates/index.html`）
- Webhook: 獨立 Flask 路由（`server/webhook_server.py`）
- RAG 引擎: `src/rag_engine.py`
- 索引建置: `src/rag_ingestion.py`
