from dotenv import load_dotenv
load_dotenv()
import os
from pydantic import BaseModel, Field 

class Configuration(BaseModel):
    """ Configuration for the agent. """
    question_expand_model: str = Field(
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), 
        description="The model to use for question expansion"
    )

    generate_model: str = Field(
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        description="The model to use for generation"
    )

    