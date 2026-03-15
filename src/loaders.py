import json
import os
from typing import List

from langchain_core.documents import Document


def load_standard_json(file_path: str) -> List[Document]:
    print(f"Loading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: List[Document] = []
    for category, depts in data.items():
        for dept_name, content in depts.items():
            courses = content.get("courses", [])
            for course in courses:
                text = (
                    f"類別: {category} | 系所: {dept_name} | "
                    f"課程名稱: {course.get('name')} | 學分: {course.get('credit')} | "
                )
                documents.append(Document(
                    page_content=text,
                    metadata={"source": "standard.json", "type": "course"},
                ))
    return documents


def load_admin_data(directory: str) -> List[Document]:
    print(f"Loading admin data from {directory}...")
    documents: List[Document] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": os.path.basename(file_path), "type": "administrative"},
                    ))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return documents
