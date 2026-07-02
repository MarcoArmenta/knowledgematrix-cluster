"""
    Download the MNIST-1D dataset (Greydanus, github.com/greydanus/mnist1d)
    into extra/experiments/data/mnist1d_data.pkl.

    Usage:
        python extra/experiments/download_mnist1d.py
"""
import os
import urllib.request

URL = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
OUT = os.path.join(os.path.dirname(__file__), "data", "mnist1d_data.pkl")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        print(f"Already present: {OUT}")
        return
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, OUT)
    print(f"Saved to {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
