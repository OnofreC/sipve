from main import app
import argparse
parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
parser.add_argument("--port", default=5000, type=int, help="port number")
args = parser.parse_args()


if __name__ == "__main__":
        app.run(host="0.0.0.0", port=args.port)