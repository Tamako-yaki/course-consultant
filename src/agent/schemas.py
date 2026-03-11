from pydantic import BaseModel, Field

class SubQuestions(BaseModel):
    """ Schema for sub-questions decomposed from the original question. """
    sub_questions: list[str] = Field(description="List of sub-questions decomposed from the original question")    