import signal
import sys
import uvicorn

from RiskGrabber.backend.api import app


def _exit_on_signal(signum, frame):  # noqa: ARG001
    print("\nЗавершение работы сервиса...", flush=True)
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGINT, _exit_on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _exit_on_signal)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
