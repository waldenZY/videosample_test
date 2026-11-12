"""python app.py --host 0.0.0.0 --port 8503"""
import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="抽帧实验台")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8503)
    args = parser.parse_args()
    uvicorn.run("videosample.server:app", host=args.host, port=args.port)
