from pydantic import BaseModel, Field 

class Configuration(BaseModel):
    """ Configuration for the agent. """
    question_expand_model: str = Field("gemini-2.5-flash", description="The model to use for question decomposition")
    
    generate_model: str = Field("gemini-2.5-flash", description="The model to use for generation")

    