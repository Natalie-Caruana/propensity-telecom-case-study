"""Frontend router — serves the scoring UI rendered as a Jinja2 template."""

import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(include_in_schema=False)

_TEMPLATES = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the scoring UI with server-injected configuration."""
    predict_url = str(request.url_for("predict"))

    cfg = getattr(request.app.state, "cfg", None)
    model_name = cfg.mlflow.model_name if cfg is not None else "unknown"

    env = os.getenv("APP_ENV", "development")

    return _TEMPLATES.TemplateResponse(  # type: ignore[return-value]
        request,
        "index.html.j2",
        {
            "predict_url": predict_url,
            "model_name": model_name,
            "env": env,
        },
    )
