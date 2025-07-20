# CFB + NFL ATS Model API

This is the backend API for a custom betting model serving JSON data to a Retool dashboard.

## Features
- Serves `/api/model-data` endpoint
- FastAPI + Render
- Easily consumed by Retool frontend
- CSV-based for now; can be swapped to real model script later

## Getting Started
1. Clone this repo
2. Push to your own GitHub
3. Deploy to Render as a "Web Service"
4. Connect `/api/model-data` to Retool

All calculations are done server-side before output.
