## How to build a new docker image including new trained agents:
1. Install docker
2. Make sure you have a dataset.zip at your repo root
3. docker build -t ghcr.io/<your-username>/gr_test_base:latest -f CI/Dockerfile .
(the -f Dockerfile tells docker which Dockerfile to use and the '.' tells docker what's the build context, or where the dataset.zip should live)
4. docker push ghcr.io/<your-username>/gr_test_base:latest
