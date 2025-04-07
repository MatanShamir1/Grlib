## How to build a new docker image including new trained agents:
1. Install docker
2. Make sure you have a dataset.zip at your repo root
3. Make sure you have a classic token in github: https://github.com/settings/tokens . If you don't, create one with package write, read and delete permissions and copy it somewhere safe.
4. Authenticate to ghcr with docker by running:
```sh
echo ghp_REST_OF_TOKEN | docker login ghcr.io -u MatanShamir1 --password-stdin
```
3. docker build -t ghcr.io/<your-username>/gr_test_base:latest -f CI/Dockerfile .
(the -f Dockerfile tells docker which Dockerfile to use and the '.' tells docker what's the build context, or where the dataset.zip should live)
4. docker push ghcr.io/<your-username>/gr_test_base:latest
docker push ghcr.io/MatanShamir1/gr_test_base:latest
