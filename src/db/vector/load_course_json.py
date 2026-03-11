import json
from pathlib import Path
from langchain_core.documents import Document

def format_time(time_dict):
    """將時間字典轉為可讀文字"""
    days = {
        "mon": "週一", "tue": "週二", "wed": "週三",
        "thu": "週四", "fri": "週五", "sat": "週六", "sun": "週日"
    }
    result = []
    for eng, chi in days.items():
        if time_dict.get(eng):
            slots = ",".join(time_dict[eng])
            result.append(f"{chi} 第{slots}節")
    return " / ".join(result) if result else "不定期"

def load_main_courses(file_path: Path) -> list[Document]:
    """讀取 main.json 並轉換為 Document"""
    if not file_path.exists():
        print(f"Warning: {file_path} 不存在")
        return []
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = []
    for course in data:
        zh_name = course.get("name", {}).get("zh", "未知課程")
        en_name = course.get("name", {}).get("en", "")
        course_id = course.get("id", "")
        course_code = course.get("code", "")
        
        # 組合自然語言內容
        content = [
            f"【114學年度課程資訊】",
            f"課程名稱：{zh_name} ({en_name})",
            f"課程ID：{course_id}",
            f"課號：{course_code}",
            f"學分：{course.get('credit', '0')} 學分 / {course.get('hours', '0')} 小時",
            f"課程類型：{course.get('courseType', '')}",
            f"開課班級：{', '.join([c.get('name', '') for c in course.get('class', [])])}",
            f"授課教師：{', '.join([t.get('name', '') for t in course.get('teacher', [])])}",
            f"上課時間：{format_time(course.get('time', {}))}",
            f"教室：{', '.join([r.get('name', '') for r in course.get('classroom', [])])}",
            f"修課人數限制：{course.get('people', '無')}",
            f"課程說明：{course.get('description', {}).get('zh', '無說明')}",
            f"備註：{course.get('notes', '無')}"
        ]
        
        metadata = {
            "source": str(file_path),
            "type": "course_info",
            "course_id": course_id,
            "course_code": course_code,
            "course_name": zh_name
        }
        
        docs.append(Document(page_content="\n".join(content), metadata=metadata))
        
    print(f"從 main.json 載入 {len(docs)} 門課程")
    return docs

def load_standard_courses(file_path: Path) -> list[Document]:
    """讀取 standard.json 並按系所轉為 Document"""
    if not file_path.exists():
        return []
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = []
    # 結構是：學程名稱 -> 系所名稱 -> {credits, courses, rules}
    for program_name, departments in data.items():
        for dept_name, details in departments.items():
            content = [f"【114學年度課程標準表 — {program_name} / {dept_name}】"]
            
            # 學分要求
            credits = details.get("credits", {})
            if credits:
                content.append("畢業學分要求：")
                for key, val in credits.items():
                    content.append(f"  {key}：{val}")
            
            # 課程清單
            courses = details.get("courses", [])
            if courses:
                content.append("\n開設課程：")
                for c in courses:
                    sem_str = f"{c.get('year')}年級{c.get('sem')}學期"
                    content.append(f"- {c.get('name')} ({sem_str}, {c.get('type')}, {c.get('credit')}學分)")
            
            # 修課規定
            rules = details.get("rules", [])
            if rules:
                content.append("\n修課規定：")
                for rule in rules:
                    content.append(f"- {rule.strip()}")
            
            metadata = {
                "source": str(file_path),
                "type": "curriculum_standard",
                "program": program_name,
                "department": dept_name
            }
            docs.append(Document(page_content="\n".join(content), metadata=metadata))
            
    print(f"從 standard.json 載入 {len(docs)} 個系所標準表")
    return docs

def load_syllabus_dir(dir_path: Path) -> list[Document]:
    """讀取 syllabus 目錄下的所有 JSON"""
    if not dir_path.exists():
        return []
        
    docs = []
    for file_path in dir_path.glob("*.json"):
        if file_path.stat().st_size < 10: # 跳過內容太少的檔
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            course_id = file_path.stem
            for teacher_syllabus in data:
                name = teacher_syllabus.get("name", "未知教師")
                content = [
                    f"【114學年度教學大綱 — 課程ID: {course_id}】",
                    f"授課教師：{name} ({teacher_syllabus.get('email', '')})",
                    f"課程目標：{teacher_syllabus.get('objective', '無')}",
                    f"教學進度：{teacher_syllabus.get('schedule', '無')}",
                    f"評分方式：{teacher_syllabus.get('scorePolicy', '無')}",
                    f"教材：{teacher_syllabus.get('materials', '無')}",
                    f"諮商時間/方式：{teacher_syllabus.get('consultation', '無')}",
                    f"課程對應SDGs指標：{teacher_syllabus.get('課程對應SDGs指標', '無')}",
                    f"是否導入AI：{teacher_syllabus.get('課程是否導入AI', '無')}",
                    f"備註：{teacher_syllabus.get('remarks', '無')}"
                ]
                
                metadata = {
                    "source": str(file_path),
                    "type": "syllabus",
                    "course_id": course_id,
                    "teacher": name
                }
                docs.append(Document(page_content="\n".join(content), metadata=metadata))
        except Exception as e:
            print(f"讀取大綱 {file_path} 失敗: {e}")
            
    print(f"從 syllabus 目錄載入 {len(docs)} 份教學大綱")
    return docs
