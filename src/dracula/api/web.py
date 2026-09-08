"""Serve the built browser application beside the stateless API.

The browser uses /Dracula/ for files and /Dracula/api for JSON requests.
Only the frontend distribution is exposed as files; model weights and Python
source live elsewhere. Hash navigation stays in the browser, so unknown file
paths return 404 rather than being replaced with the application's HTML.
"""

from pathlib import Path
import re

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles


def create_web_app(api: FastAPI, frontend: Path) -> FastAPI:
    """Attach an already-configured API and a required frontend distribution."""

    if not (frontend / "index.html").is_file():
        raise ValueError(f"frontend build is missing: {frontend}; run make web-build")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/Dracula", include_in_schema=False)
    def trailing_slash() -> RedirectResponse:
        # Relative redirects keep the browser on its public hostname behind a proxy.
        return RedirectResponse("/Dracula/", status_code=308)

    @app.middleware("http")
    async def cache_headers(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/Dracula/api") or response.status_code >= 400:
            response.headers["Cache-Control"] = "no-store"
        elif path.startswith("/Dracula/assets/") and re.search(
            r"-[A-Za-z0-9_-]{8,}\.[^/]+$", path
        ):
            # Vite changes these filenames when their content changes.
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            # HTML and named artwork may change without changing their URLs.
            response.headers["Cache-Control"] = "no-cache"
        return response

    # First match wins: JSON requests must never reach the static-file mount.
    app.mount("/Dracula/api", api)
    app.mount("/Dracula", StaticFiles(directory=frontend, html=True))
    return app
