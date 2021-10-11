#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2021 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2021 The MITRE Corporation                                      #
#                                                                           #
# Licensed under the Apache License, Version 2.0 (the "License");           #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at                                   #
#                                                                           #
#    http://www.apache.org/licenses/LICENSE-2.0                             #
#                                                                           #
# Unless required by applicable law or agreed to in writing, software       #
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

import argparse
import json
import subprocess
import shlex
from datetime import datetime
from typing import Dict, Any, List


def main():
    args = parse_cmd_line_args()

    detection_type = "FACE"

    container_id = set_up_container(args.docker_image)

    start_time = datetime.now()
    output_obj = run_cli_runner_stdin_media(container_id, args.media_path, '-t', 'image', '-')
    end_time = datetime.now()

    tracks = get_image_tracks(output_obj, detection_type)

    shut_down_container(container_id)

    print("\nFound detections: ", len(tracks))
    print("Run time: ", str(end_time - start_time))


def set_up_container(image_name: str):
    command = ('docker', 'run', '--rm', '-d', image_name, '-d')
    print('Starting test container with command: ', shlex.join(command))
    proc = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
    return proc.stdout.strip()


def shut_down_container(container_id: str):
    command = ('docker', 'stop', container_id)
    print('Stopping test container with command: ', shlex.join(command))
    subprocess.run(command, check=True)


def run_cli_runner_stdin_media(container_id: str,
                               media_path: str,
                               *runner_args: str) -> Dict[str, Any]:
    return run_cli_runner_stdin_media_and_env_vars(container_id, media_path, {}, *runner_args)


def run_cli_runner_stdin_media_and_env_vars(container_id: str,
                                            media_path: str,
                                            env_dict: Dict[str, str],
                                            *runner_args: str) -> Dict[str, Any]:
    env_params = (f'-e{k}={v}' for k, v in env_dict.items())
    command = ['docker', 'exec', '-i', *env_params, container_id, 'runner', *runner_args]
    print('Running job with command: ', shlex.join(command))

    with open(media_path) as media_file, \
            subprocess.Popen(command, stdin=media_file, stdout=subprocess.PIPE,
                             text=True) as proc:
        return json.load(proc.stdout)


def get_image_tracks(
        output_object: Dict[str, Any],
        detection_type: str) -> List[Dict[str, Any]]:
    return get_tracks(output_object, detection_type)


def get_tracks(
        output_object: Dict[str, Any],
        detection_type: str) -> List[Dict[str, Any]]:
    return output_object['media'][0]['output'][detection_type][0]['tracks']


def parse_cmd_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('docker_image',
                        help='Docker image name with registry and tag.')

    parser.add_argument('media_path',
                        help='Path to media to process.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
