#!/bin/bash
# Single worker on purpose: the per-session round counter is in-memory per
# process (the vectors live in Postgres, not in memory). Uses the Oryx venv.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
