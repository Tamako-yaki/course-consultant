index:
	@PYTHONPATH=src uv run python -m db.vector.index_documents
run:
	@PYTHONPATH=src HF_HUB_DISABLE_PROGRESS_BARS=1 uv run python -m cli