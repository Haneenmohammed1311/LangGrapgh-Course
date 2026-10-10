# THE BLUEPRINT
#--------------
# this file make the llm output structured and can be used to validate the output of the llm,
# it uses pydantic to define the schema of the output, and the output of the llm will be validated against this schema

from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="What is missing from the answer?")
    superfluous: str = Field(description="What is superfluous in the answer?")

class AnswerQuestion(BaseModel):
    """Answer to a question."""
    answer: str = Field(description="~250 word answer to the question.")
    reflection: Reflection = Field(description="Reflection on the initial answer.")
    search_queries: List[str] = Field(description="1-3 Recommended search queries to improve the answer.")

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )