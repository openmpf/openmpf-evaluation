import os
import json
import subprocess
import shlex
import time
from typing import Dict, Any, List


def get_test_media(file_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'data', file_name)


def _get_full_image_name(image_name):
    test_registry = os.getenv('TEST_REGISTRY', '')
    if len(test_registry) > 0 and test_registry[-1] != '/':
        test_registry += '/'

    test_img_tag = os.getenv('TEST_IMG_TAG')
    if not test_img_tag:
        test_img_tag = 'latest'
    if test_img_tag[0] == ':':
        test_img_tag = test_img_tag[1:]

    return f'{test_registry}{image_name}:{test_img_tag}'


def set_up_container(image_name):
    full_image_name = _get_full_image_name(image_name)
    command = ('docker', 'run', '--rm', '-d', full_image_name, '-d')
    print('Starting test container with command: ', shlex.join(command))
    proc = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
    return proc.stdout.strip()


def shut_down_container(_container_id):
    command = ('docker', 'stop', _container_id)
    print('Stopping test container with command: ', shlex.join(command))
    subprocess.run(command, check=True)


def run_cli_runner_stdin_media(_container_id, media_path: str, *runner_args: str) -> Dict[str, Any]:
    return run_cli_runner_stdin_media_and_env_vars(_container_id, media_path, {}, *runner_args)


def run_cli_runner_stdin_media_and_env_vars(_container_id, media_path: str, env_dict: Dict[str, str],
                                            *runner_args: str) -> Dict[str, Any]:
    env_params = (f'-e{k}={v}' for k, v in env_dict.items())
    command = ['docker', 'exec', '-i', *env_params, _container_id, 'runner', *runner_args]
    print('Running job with command: ', shlex.join(command))

    with open(media_path) as media_file, \
            subprocess.Popen(command, stdin=media_file, stdout=subprocess.PIPE,
                             text=True) as proc:
        return json.load(proc.stdout)


def get_image_tracks(
        expected_path,
        expected_mime_type: str,
        expected_job_props: Dict[str, str],
        expected_media_metadata: Dict[str, str],
        _output_object: Dict[str, Any],
        _detection_type) -> List[Dict[str, Any]]:

    return _get_tracks(_output_object, _detection_type)


def _get_tracks(_output_object: Dict[str, Any], _detection_type) -> List[Dict[str, Any]]:
    return _output_object['media'][0]['output'][_detection_type][0]['tracks']


if __name__ == "__main__":

    start_time = time.time()

    container_image_name = 'openmpf_ocv_face_detection'

    _face_image = get_test_media('meds-af-S419-01_40deg.jpg')
    _default_job_properties = {'MAX_FEATURE': '250',
                               'MAX_OPTICAL_FLOW_ERROR': '4.7',
                               'MIN_FACE_SIZE': '48',
                               'MIN_INITIAL_CONFIDENCE': '10',
                               'MIN_INIT_POINT_COUNT': '45',
                               'MIN_POINT_PERCENT': '0.70',
                               'VERBOSE': '0'}

    detection_type = "FACE"

    container_id = set_up_container(container_image_name)

    output_object = run_cli_runner_stdin_media(container_id, _face_image, '-t', 'image', '-')
    tracks = get_image_tracks(
        '/dev/stdin', 'image/octet-stream', _default_job_properties, {}, output_object, detection_type)

    shut_down_container(container_id)

    end_time = time.time()

    print("\nFound detections: ", len(tracks))
    print("Run time: ", end_time - start_time)
