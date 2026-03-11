index:
	@PYTHONPATH=src uv run python -m db.vector.index_documents
run:
	@PYTHONPATH=src uv run python -m cli