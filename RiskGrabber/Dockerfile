FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY RiskGrabber/pyproject.toml /app/
COPY RiskGrabber/requirements.txt /app/requirements.txt

RUN uv pip install --system -r /app/requirements.txt

COPY RiskGrabber /app/RiskGrabber

ENV RISK_GRABBER_DB_URL=postgresql+psycopg2://risk_grabber:risk_grabber@localhost:5432/risk_grabber

EXPOSE 8000

CMD ["python", "-m", "RiskGrabber.main"]