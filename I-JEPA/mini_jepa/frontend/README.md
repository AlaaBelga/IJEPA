# MiniJEPA Frontend

Minimal React + Vite frontend that uploads a masked image to the FastAPI backend at `/api/predict` and displays the completed image and a prediction panel returned as data URLs.

Run locally:

```bash
cd mini_jepa/frontend
npm install
npm run dev
```

The dev server proxies `/api` to `http://127.0.0.1:8000` by default (see `vite.config.js`).
