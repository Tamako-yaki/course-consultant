# Course Data Preprocessing Summary

## Overview
Successfully preprocessed all course data from `data/114/` directory to prepare for embedding. The new preprocessed data is cleaner, more suitable for embedding, and contains **4,709 documents** ready for the RAG pipeline.

## Processing Results

### Input Data
- **main.json** (4.9MB): 2,531 course listings
- **研究所(日間部、進修部、週末碩士班).json** (1.4MB): 812 graduate program courses
- **course/*.json**: 1,982 individual course syllabus files

### Output Data (in `data/processed/`)
- **main_courses.json** (3.2MB): 2,531 cleaned main course listings
- **research_courses.json** (846KB): 812 cleaned research course listings
- **course_syllabi.json** (3.9MB): 1,366 processed course syllabi
- **id_mapping.json** (402KB): Reference mapping of all course IDs

### Total Documents Generated
**4,709 documents** ready for embedding:
- 3,343 course listings (from main + research courses)
- 1,366 course syllabi (detailed instructor information)

## Data Cleaning

### Removed (Useless Navigation Noise)
✗ Internal JSP links: `Subj.jsp?format=-4&year=114&sem=1&code=2652`
✗ Teacher reference links: `Teach.jsp?format=-3&year=114&sem=1&code=12202`
✗ Classroom links: `Croom.jsp?format=-3&year=114&sem=1&code=438`
✗ Empty `id`, `code`, and `snum` reference fields

### Preserved (Meaningful Content)
✓ Course names (both Chinese and English)
✓ Course descriptions
✓ Credit and hour information
✓ Teacher names
✓ Department/class information
✓ Course objectives and syllabi
✓ Schedule and grading policies
✓ Learning materials
✓ Consultation information
✓ SDG indicators
✓ AI integration notes

## Document Structure

### Course Listing Document
```
課程代碼: 2B05002 | 課程名稱: 實務專題 (二) (Special Projects (II)) |
課程描述: 指導學生進行研究專題... | 學分: 1.0 | 時數: 2 |
授課教師: 吳修明, 莊政達, 顏毅廣, 粘朝益 | 開課系所: 智動五

Metadata: {
  "source": "main_courses.json",
  "type": "course_listing",
  "code": "2B05002"
}
```

### Course Syllabus Document
```
課程代碼: 349069 | 授課教師: 劉靜怡 | 課程目標: 本校科技大學的國文課程... |
課程進度: 第一週：課程介紹與說明... | 成績評定: 1. 期中考：筆試，佔30％...

Metadata: {
  "source": "course_syllabi.json",
  "type": "course_syllabus",
  "course_id": "349069"
}
```

## New Functions in `src/loaders.py`

### `load_preprocessed_courses(processed_dir: str) -> List[Document]`
Main function to load all preprocessed course data. Returns a list of LangChain `Document` objects ready for embedding.

**Usage:**
```python
from src.loaders import load_preprocessed_courses

documents = load_preprocessed_courses()
# Returns 4,709 documents ready for embedding
```

### Helper Functions
- `_format_course_as_text()`: Formats course listing dictionary into readable text
- `_format_syllabus_as_text()`: Formats syllabus dictionary into readable text

## ID Mapping Reference

The `id_mapping.json` file provides a lightweight reference of all course IDs without noise:

```json
{
  "2B05002": {
    "type": "course_listing",
    "name": {"zh": "實務專題 (二)", "en": "Special Projects (II)"},
    "credit": "1.0"
  },
  "349069": {
    "type": "course_syllabus",
    "instructor": "劉靜怡",
    "latest_update": "2025-05-23 12:40:50"
  }
}
```

Can be used to:
- Link related documents
- Deduplicate references
- Maintain cross-references without embedding raw IDs

## Files Modified
- Created: `src/preprocess_courses.py` - Preprocessing script
- Updated: `src/loaders.py` - Added new `load_preprocessed_courses()` function
- Created: `data/processed/` - Output directory with 4 JSON files

## Next Steps
The preprocessed data is now ready for:
1. ✅ Loading into vector database
2. ✅ Embedding with any embedding model
3. ✅ Using in RAG retrieval pipeline
4. ✅ Building search indexes

No more JSP link noise or useless IDs in the embedding pipeline!
