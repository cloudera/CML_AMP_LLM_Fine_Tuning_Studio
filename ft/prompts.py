

from ft.api import *

from cmlapi import CMLServiceApi

from typing import List

from sqlalchemy import delete

from ft.db.dao import FineTuningStudioDao
from ft.db.model import Prompt


def list_prompts(request: ListPromptsRequest, cml: CMLServiceApi = None,
                 dao: FineTuningStudioDao = None) -> ListPromptsResponse:
    """
    Right now we don't do any filtering in this op, but we might in the future.
    """
    with dao.get_session() as session:
        prompts: List[Prompt] = session.query(Prompt).all()
        return ListPromptsResponse(
            prompts=list(map(
                lambda x: x.to_protobuf(PromptMetadata),
                prompts
            ))
        )


def get_prompt(request: GetPromptRequest, cml: CMLServiceApi = None,
               dao: FineTuningStudioDao = None) -> GetPromptResponse:
    with dao.get_session() as session:
        return GetPromptResponse(
            prompt=session
            .query(Prompt)
            .where(Prompt.id == request.id)
            .one()
            .to_protobuf(PromptMetadata)
        )


def add_prompt(request: AddPromptRequest, cml: CMLServiceApi = None,
               dao: FineTuningStudioDao = None) -> AddPromptResponse:
    with dao.get_session() as session:
        prompt: Prompt = Prompt.from_message(request.prompt)
        session.add(prompt)
        return AddPromptResponse(
            prompt=prompt.to_protobuf(PromptMetadata)
        )


def remove_prompt(request: RemovePromptRequest, cml: CMLServiceApi = None,
                  dao: FineTuningStudioDao = None) -> RemovePromptResponse:
    with dao.get_session() as session:
        session.execute(delete(Prompt).where(Prompt.id == request.id))
    return RemovePromptResponse()
