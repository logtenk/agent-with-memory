#!/usr/bin/env bash
export PYTHONPATH=.
uvicorn agent_host.app.main:app --host 0.0.0.0 --port 8080 --reload
