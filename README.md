# script_to_movie
Auto generate tool a movie from script.
This repository's application has only been tested and confirmed to work on WSL2. 
Therefore, we cannot guarantee its functionality in other environments.

# Getting Started

## Start Voicevox API

```
docker pull voicevox/voicevox_engine:cpu-ubuntu20.04-latest
docker run -itd -p 50021:50021 voicevox/voicevox_engine:cpu-ubuntu20.04-latest
```

## Setting ImageMagick

- Install

```
sudo apt install -y imagemagick
```

## Running App

```
poetry install
poetry run python app.py
```
