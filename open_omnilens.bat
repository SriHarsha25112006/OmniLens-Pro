@echo off
echo Opening OmniLens Pro...
start http://localhost:3000
echo.
echo Please ensure that both servers are currently running in your terminal:
echo 1. Next.js Frontend (omnilens area: npm run dev)
echo 2. Python ML Backend (omnilens-ml area: python -m ml_engine.main)
echo.
pause