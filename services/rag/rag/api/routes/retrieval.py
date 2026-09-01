from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_rag_module
from ..schemas import RetrievalRequest, RetrievalResponse
from ..serialization import filters_from_request, retrieval_to_response


router = APIRouter(prefix="/v1", tags=["retrieval"])


@router.post("/retrieval", response_model=RetrievalResponse, response_model_exclude_none=True)
def retrieve(request: RetrievalRequest, module=Depends(get_rag_module)) -> RetrievalResponse:
    result = module.retrieve(
        request.query,
        mode=request.mode,
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        filters=filters_from_request(request.filters),
        rerank=request.rerank,
        include_trace=request.debug,
    )
    return retrieval_to_response(
        result,
        debug=request.debug,
        observability=getattr(module.config, "observability", None),
    )
