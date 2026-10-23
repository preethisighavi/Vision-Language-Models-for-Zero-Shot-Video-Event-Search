from fastapi import APIRouter, Depends, Request
from sqlmodel import Session

from app.db.session import get_session
from app.schemas.api import ErrorResponse, SearchRequest, SearchResponse
from app.security import client_ip, search_limiter
from app.services.search_service import run_search

router = APIRouter(tags=["search"])

ERRORS = {404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 429: {"model": ErrorResponse}}


@router.post("/search", response_model=SearchResponse, responses=ERRORS)
def search(req: SearchRequest, request: Request, session: Session = Depends(get_session)) -> SearchResponse:
    search_limiter.check(client_ip(request))
    return run_search(session, req)
