import gdown
import zipfile
import os

path = os.path.dirname(os.path.abspath(__file__))

def download_and_extract_dataset(file_id, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    output = os.path.join(extract_to, "dataset.zip")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(output)

if __name__ == "__main__":
    file_id = "1PK1iZONTyiQZBgLErUO88p1YWdL4B9Xn"
    extract_to = path
    download_and_extract_dataset(file_id, extract_to)
