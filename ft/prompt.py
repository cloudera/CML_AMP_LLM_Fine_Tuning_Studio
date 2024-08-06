from pydantic import BaseModel
from typing import List, Optional


class PromptMetadata(BaseModel):

    id: str
    """
    Unique ID of the prompt in question.
    """

    name: str
    """
    Human-friendly name of this prompt template
    for use-cases elsewhere
    """

    dataset_id: str
    """
    ID of the dataset that uses this prompt.
    This dataset should contain column names
    that correspond to the items that are
    in the list of slots.
    """

    slots: Optional[List[str]] = None
    """
    List of slots in this prompt template.
    """

    prompt_template: str
    """
    Python formatted prompt string template.
    """
