import uvicorn

from RiskGrabber.backend.api import app


def main() -> None:
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
