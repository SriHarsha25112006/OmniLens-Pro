import asyncio
import sys
import uvicorn

if __name__ == "__main__":
    # CRITICAL: Set ProactorEventLoopPolicy for Windows to support subprocesses (Playwright)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Reload=False on Windows is safer for Playwright subprocesses
    uvicorn.run("ml_engine.main:app", host="127.0.0.1", port=8000, reload=False)
