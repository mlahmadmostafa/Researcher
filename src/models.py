from pydantic import BaseModel, Field
from typing import Optional

class PaperRelevance(BaseModel):
    """
    Evaluation of a paper's relevance to the research topic.
    """
    id: str = Field(description="The unique ID of the paper (e.g., Arxiv ID)")
    relevance_score: int = Field(description="A score from 0-10 indicating how relevant specific paper is to the user's question. 10 being highly relevant.")
    is_relevant: bool = Field(description="True if the paper is relevant enough to be included in the final report.")
    summary: str = Field(description="A concise summary of the paper, focusing specifically on information relevant to the user's topic.")
    
class Paper(BaseModel):
    """
    Represents a research paper.
    """
    title: str
    url: str
    content: str
    evaluation: Optional[PaperRelevance] = None
