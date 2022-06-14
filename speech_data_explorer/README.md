# Getting started

The NVIDIA Speech Data explorer tool takes a JSON manifest file as input. The file should be formatted as one JSON object per line, and each JSON should contain the following fields:
* `audio_filepath`
* `duration` (duration of the audio file in seconds)
* `text` (reference transcript)

To compute error analysis each JSON should also include a field for `pred_text`

To run the speech data explorer run

`python data_explorer.py <path_to_JSON_manifest>`

