# MedAlloc RL

Hospital Resource Allocation...
# Hospital Resource Allocation Environment

A simulation environment for hospital resource allocation using OpenEnv.

## Features
- 3 difficulty levels
- Realistic patient severity scoring
- Bed allocation logic
- FastAPI server

## Installation

```bash
pip install fastapi uvicorn requests
```

## Run Locally

```bash
uvicorn server.app:app --reload
```

## API Endpoints

- GET /health - Check server status
- POST /reset - Reset environment
- POST /step - Take action step
- GET /state - Get current state

## Tasks

- Easy: 3 patients, 3 beds
- Medium: 5 patients, 3 beds
- Hard: 10 patients, 2 beds

## Reward System

- Severity 4+: 1.0 reward
- Severity 2-3: 0.7 reward
- Severity 0-1: 0.3 reward
