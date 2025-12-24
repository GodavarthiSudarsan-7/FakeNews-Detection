
import subprocess
import os
import sys

def download():
    os.makedirs("data/raw", exist_ok=True)
    print("Downloading Fake & Real News Dataset using your Python interpreter...")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
           "-d", "clmentbisaillon/fake-and-real-news-dataset",
           "-p", "data/raw", "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Download complete. Files are in data/raw")

if __name__ == "__main__":
    download()
