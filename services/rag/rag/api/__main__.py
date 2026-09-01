from __future__ import annotations

import uvicorn

from .settings import ApiSettings


def main() -> None:
    settings = ApiSettings.from_env()
    uvicorn.run(
        "rag.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
