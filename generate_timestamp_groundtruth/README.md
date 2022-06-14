# Install Pyannote

```
conda create -n pyannote python=3.8
conda activate pyannote
conda install pytorch torchaudio -c pytorch
pip install https://github.com/pyannote/pyannote-audio/archive/develop.zip
```

# Run script
```
python generate_timestamp_groundtruth.py input [--output_dir OUTPUT_DIR] [--merge_labels]
```
Input can be an audio file, directory of audio files, or a .txt list of audio files.
Pyannote uses torchaudio with the soundfile backend to load audio, so .mp3 is not supported.
See [https://pytorch.org/audio/stable/backend.html#backend](https://pytorch.org/audio/stable/backend.html#backend) for a full list of what audio formats are supported.

If output directory is not specified, groundtruth files will be output to the current directory. One groundtruth file is created per audio file. The groundtruth file for audio file audio_file.wav is audio_file.json.

By default timestamps are by utterance and are not merged.
