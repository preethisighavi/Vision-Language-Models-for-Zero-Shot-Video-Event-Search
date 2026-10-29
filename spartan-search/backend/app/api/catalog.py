from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.db.session import get_session
from app.models.registry import get_registry
from app.schemas.api import CollectionInfo, Health, ModelInfo
from app.services import catalog

router = APIRouter(tags=["catalog"])


@router.get("/health", response_model=Health)
def health(session: Session = Depends(get_session)) -> Health:
    return Health(
        status="ok",
        models_loaded=get_registry().loaded_ids(),
        index_sizes=catalog.index_sizes(session),
    )


@router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return catalog.model_infos()


@router.get("/collections", response_model=list[CollectionInfo])
def list_collections(session: Session = Depends(get_session)) -> list[CollectionInfo]:
    return catalog.collection_infos(session)
