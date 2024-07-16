from pydantic import BaseModel
from enum import Enum
from typing import Optional 


class PEFTAdapterType(Enum):
    """
    Type of PEFT adapter.
    """
    
    LOCAL = "local"
    """
    Local PEFT adapter imported from project
    files, probably created after a fine-tuning
    job was ran in our app.
    """

    HUGGINGFACE = "huggingface"
    """
    Huggingface-stored adapter that can be pulled
    down from HF hub.
    """



class PEFTAdapterMetadata(BaseModel):

    id: str
    """
    Unique ID of the PEFT adapter.
    """

    type: PEFTAdapterType
    """
    Type of PEFT adapter.
    """

    model_id: str 
    """
    Corresponding model ID that this adapter is designed for. This is the
    model ID in the FT app. 
    """
    
    location: Optional[str]
    """
    Project-relative directory where the PEFT adapter data is stored.
    
    When training with HF/TRL libraries, a typical output directory 
    for PEFT adapters will contain files like:
    * adapter_config.json
    * adapter_model.bin

    This dataclass currently just stores the location of the PEFT adapter
    in the local directory which can then be used to load an adapter.
    """

    huggingface_name: Optional[str]
    """
    Huggingface PEFT adapter name (identifier used to find
    the adapter on HF hub).
    """