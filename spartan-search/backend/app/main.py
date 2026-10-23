import base64
import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import admin as admin_api
from app.api import catalog as catalog_api
from app.api import explore as explore_api
from app.api import search as search_api
from app.api import uploads as uploads_api
from app.api import videos as videos_api
from app.config import get_settings
from app.db.session import init_db, session_scope
from app.errors import install_error_handlers
from app.logging_setup import configure_logging
from app.models.registry import get_registry
from app.services.catalog import ensure_default_collections

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    settings = get_settings()
    for path in (settings.data_dir, settings.media_dir, settings.index_dir, settings.features_dir):
        path.mkdir(parents=True, exist_ok=True)
    init_db()
    with session_scope() as s:
        ensure_default_collections(s)
    if settings.preload_models:
        get_registry().preload()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Spartan Search API",
        version="1.0.0",
        lifespan=lifespan,
        openapi_url="/openapi.json",
        docs_url="/docs",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )

    if settings.site_password:
        expected = settings.site_password.encode()

        @app.middleware("http")
        async def basic_auth(request: Request, call_next):
            if request.method == "OPTIONS" or request.url.path.endswith("/health"):
                return await call_next(request)
            header = request.headers.get("authorization", "")
            if header.lower().startswith("basic "):
                try:
                    _, _, password = base64.b64decode(header[6:]).decode().partition(":")
                    if hmac.compare_digest(password.encode(), expected):
                        return await call_next(request)
                except Exception:
                    pass
            return JSONResponse(
                {"error": {"code": "unauthorized", "message": "Password required."}},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Spartan Search"'},
            )

    install_error_handlers(app)
    routers = (catalog_api.router, explore_api.router, search_api.router, uploads_api.router, videos_api.router,
               admin_api.router)
    for router in routers:
        app.include_router(router, prefix=API_PREFIX)
    return app


app = create_app()
