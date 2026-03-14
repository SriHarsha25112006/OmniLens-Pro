import subprocess
import sys
import os
import time
import webbrowser
import atexit
import signal

def force_kill_ports(ports):
    """Forcefully kill processes on specific ports."""
    if sys.platform != "win32": return
    for port in ports:
        try:
            output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
            pids = set()
            for line in output.splitlines():
                parts = line.split()
                if len(parts) > 4:
                    pids.add(parts[-1])
            for pid in pids:
                if pid != "0": # Ignore idle process
                    print(f"[*] Force killing process {pid} on port {port}...")
                    subprocess.run(["taskkill", "/F", "/PID", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root_dir, "omnilens")
    backend_dir = os.path.join(root_dir, "omnilens-ml")

    print("[*] Starting OmniLens Pro...")
    force_kill_ports([3000, 8000])

    # Start Frontend
    print("[*] Starting Next.js frontend...")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_process = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=frontend_dir,
        shell=False
    )

    # Start Backend
    print("[*] Starting Python ML backend...")
    python_cmd = os.path.join(backend_dir, "venv", "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(backend_dir, "venv", "bin", "python")
    backend_process = subprocess.Popen(
        [python_cmd, "-m", "ml_engine.main"],
        cwd=backend_dir,
        shell=False
    )

    # Cleanup function
    def cleanup(*args):
        print("\n[*] Shutting down servers to save CPU cycles...")
        def kill_proc(proc):
            if proc.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        
        kill_proc(frontend_process)
        kill_proc(backend_process)
        
        print("[*] OmniLens Pro safely closed.")
        sys.exit(0)

    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("[*] Waiting for UI to become available...")
    time.sleep(4)
    print("[*] Opening browser at http://localhost:3000")
    webbrowser.open("http://localhost:3000")

    print("[*] Servers are running! Press Ctrl+C in this terminal to gracefully shut everything down.")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
            # If any process dies unexpectedly, we can also exit
            if frontend_process.poll() is not None or backend_process.poll() is not None:
                print("[!] A server crashed unexpectedly. Shutting down...")
                break
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()
