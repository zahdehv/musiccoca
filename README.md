# MusicCoCa

MusicCoCa is a joint embedding model of text and audio styles for music. This repository contains the implementation of MusicCoCa, allowing you to embed both text and audio into a shared semantic space.

## Installation

Install MusicCoCa using pip:

```sh
uv pip install git+https://github.com/zahdehv/musiccoca.git
```

## Usage

Here's a basic example of how to use MusicCoCa to embed audio:

```py
from musiccoca import musiccoca, audio
import numpy as np
from time import perf_counter

embeddings_model = musiccoca.MusicCoCa()

my_audio = audio.Waveform.from_file('myjam.mp3')
audio = [
  my_audio,
]
text = [
  'heavy metal',
]

start = perf_counter()
embeddings = embeddings_model.embed(audio)
tm = perf_counter() - start

print("audio embedded in:", tm)
```

## Features

- Joint embedding of text and audio styles
- High-quality audio representations

## License

This project is licensed under the Apache 2.0 License.
