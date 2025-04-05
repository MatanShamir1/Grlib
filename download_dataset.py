import requests
import zipfile
import os

def download_and_extract_dataset(google_drive_url, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    download_url = google_drive_url + "&export=download"
    response = requests.get(download_url)
    response.raise_for_status()
    with open('dataset.zip', 'wb') as f:
        f.write(response.content)
    with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove('dataset.zip')

if __name__ == "__main__":
    google_drive_url = "https://drive.google.com/file/d/1PK1iZONTyiQZBgLErUO88p1YWdL4B9Xn/view?usp=sharing"
    extract_to = "dataset"
    download_and_extract_dataset(google_drive_url, extract_to)