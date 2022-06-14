from pyannote.audio import Pipeline
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--merge_labels', action='store_true', default=False)

# takes in a list of audio files
def process_audio_files(audio_files, output_dir, merge):
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

    try:
        for audio in audio_files:
            output = {"groundtruth": []}
            
            vad_out = pipeline(audio)
            all_utterances = vad_out.get_timeline().support()

            for speech in all_utterances:
                # convert to milliseconds
                start = int(speech.start * 1000)
                stop = int(speech.end * 1000)

                timestamp = {"pyannote_start": start, "pyannote_end": stop}
                output["groundtruth"].append(timestamp)

            if merge:
                merged_output = ""

                for utterance in output["groundtruth"]:
                    merged_output += str(utterance["pyannote_start"]) + "-" + str(utterance["pyannote_end"]) + ", "

                    output["groundtruth"] = merged_output.strip(", ")

            basename = os.path.basename(audio)

            filename = os.path.splitext(basename)[0]
            output_path = filename + ".json"

            if output_dir is not None:
                output_path = os.path.join(output_dir, output_path)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=4)

    except Exception as e:
        print(e)


def main():
    args = parser.parse_args()
    audio = args.file
    merge_labels = args.merge_labels
    output_dir = args.output_dir

    if os.path.isfile(audio):
        basename = os.path.basename(audio)
        filename, extension = os.path.splitext(basename)[0], os.path.splitext(basename)[1]

        # text file list of audio files
        if extension == ".txt":
            with open(audio) as f:
                audio_files = f.read().splitlines()
            process_audio_files(audio_files, output_dir, merge_labels)

        # audio file
        else:
            process_audio_files([audio], output_dir, merge_labels)

    # directory of audio files
    else:
        audio_files = [os.path.join(audio, file) for file in os.listdir(audio) if os.path.isfile(os.path.join(audio, file))]
        process_audio_files(audio_files, output_dir, merge_labels)
        

if __name__ == "__main__":
    main()