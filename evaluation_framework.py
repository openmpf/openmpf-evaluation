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
import os
import subprocess
import shlex
from datetime import datetime
from typing import Dict, Any, List
import fiftyone as fo
import cv2


class EvalFramework:
    EVAL_JSON_JOB_RUNS = "jobRuns"
    EVAL_JSON_JOB_NAME = "jobName"
    EVAL_JSON_DOCKER_IMAGE = "dockerImage"
    EVAL_JSON_JOB_PROPS = "jobProperties"
    EVAL_JSON_DOCKER_ENV = "dockerEnvironment"

    def __init__(self,
                 docker_job_json: Dict = {},
                 detection_type: str = "FACE",
                 output_dir: str = None,
                 view_fiftyone: bool = True):

        self.container_dict = {}
        self.detection_type = detection_type
        self.docker_job_list = []
        self.update_docker_image_jobs(docker_job_json)
        self.view_fiftyone = view_fiftyone
        self.dataset = fo.Dataset()
        self.output_dir = output_dir

    def __del__(self):
        self._shut_down_all_containers()

    def process_image_jobs(self,
                           image_list: List[str],
                           ):
        """
        After initialization, run all jobs on given list of image file paths.
            Store any results in an optional output directory specified by the user.

        :param image_list: List of image file paths.
        :return: None
        """
        # Setup dataset
        for image in image_list:
            im = cv2.imread(image)
            image_sample = fo.Sample(filepath=image)
            image_sample.set_field("image_dim", im.shape)
            self.dataset.add_sample(sample=image_sample)

        for job_entry in self.docker_job_list:
            job_name = job_entry[self.EVAL_JSON_JOB_NAME]
            docker_image = job_entry[self.EVAL_JSON_DOCKER_IMAGE]

            job_props_dict = job_entry.get(self.EVAL_JSON_JOB_PROPS, {})
            docker_env_dict = job_entry.get(self.EVAL_JSON_DOCKER_ENV, {})

            container_id = self.container_dict[docker_image.strip()]
            self._process_images(job_name, job_props_dict, docker_env_dict, container_id)

    def launch_fiftyone_session(self):
        """
        Launch a FiftyOne session if prompted by the user.
        :return: None
        """
        session = fo.launch_app(self.dataset, remote=True)
        closing = input("Would you like to close this app (y/n):")
        while closing.lower()[0] != "y":
            closing = input("Would you like to close this app (y/n):")

    def update_docker_image_jobs(self, docker_job_json: Dict):
        """
        Initializes all unique Docker job containers referenced in job JSON file.

        :param docker_job_json: Job JSON file containing MPF Docker component images and associated job parameters.
        :return: None
        """
        for job_entry in docker_job_json[self.EVAL_JSON_JOB_RUNS]:
            docker_image = job_entry[self.EVAL_JSON_DOCKER_IMAGE]
            self._set_up_containers(docker_image)
        self.docker_job_list += docker_job_json[self.EVAL_JSON_JOB_RUNS]

    def _set_up_containers(self, image_name: str):
        if image_name.strip() not in self.container_dict:
            command = ('docker', 'run', '--rm', '-d', image_name, '-d')
            print('Starting test container with command: ', shlex.join(command))
            proc = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
            self.container_dict[image_name.strip()] = proc.stdout.strip()

    def _shut_down_all_containers(self):
        for container_name in self.container_dict:
            command = ('docker', 'stop', self.container_dict[container_name])
            print('Stopping test container with command: ', shlex.join(command))
            subprocess.run(command, check=True)

    def _process_images(self,
                        job_name: str,
                        job_props_dict: Dict[str, str],
                        docker_env_dict: Dict[str, str],
                        container_id: str):
        if len(self.dataset) > 1:
            print("Found {} images.".format(len(self.dataset)))

        index = 0

        dataset_start_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

        for image_sample in self.dataset:
            image = image_sample.filepath
            index += 1
            print("\nProcessing image {}: {}".format(index, image))
            start_time = datetime.now()
            output_obj = self._run_cli_runner_stdin_media(container_id,
                                                          job_props_dict,
                                                          docker_env_dict,
                                                          image,
                                                          '-t', 'image', '-')
            end_time = datetime.now()
            tracks = self._get_image_tracks(output_obj)
            print("Found detections: ", len(tracks))
            print("Run time: ", str(end_time - start_time))

            if self.view_fiftyone:
                # Convert detections to FiftyOne format
                detections = self._convert_detections(output_obj, image_sample)
                # Save predictions to dataset
                image_sample[job_name] = fo.Detections(detections=detections)
                print("Saving detection label")
                image_sample.save()

            if self.output_dir is not None:
                out_dir = os.path.join(self.output_dir, job_name + "_" + dataset_start_time)
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(out_dir, os.path.splitext(os.path.basename(image))[0])
                with open('{}.json'.format(output_path), 'w') as fp:
                    output_obj["media"][0]["path"] = image
                    json.dump(output_obj, fp, indent=4, sort_keys=True)

    def _convert_detections(self, tracks: Dict, image_sample: fo.Sample):
        detections = []

        for entry in tracks["media"][0]["output"][self.detection_type]:
            for track in entry["tracks"]:
                for detection in track["detections"]:
                    im_h, im_w, im_c = image_sample.get_field("image_dim")
                    x = float(detection["x"])
                    y = float(detection["y"])
                    w = float(detection["width"])
                    h = float(detection["height"])
                    rel_box = [x / im_w, y / im_h, w / im_w, h / im_h]
                    print("Image stats:" + str(image_sample.get_field("image_dim")))
                    print("Image stats:" + str([x, y, w, h]))
                    print("Bounding_box: " + str(rel_box))

                    detections.append(
                        fo.Detection(
                            label=self.detection_type,
                            bounding_box=rel_box,
                            confidence=detection["confidence"]
                        )
                    )
        return detections

    def _run_cli_runner_stdin_media(self,
                                    container_id: str,
                                    job_props_dict: Dict[str, str],
                                    docker_env_dict: Dict[str, str],
                                    media_path: str,
                                    *runner_args: str) -> Dict[str, Any]:
        return self._run_cli_runner_stdin_media_and_env_vars(container_id,
                                                             media_path,
                                                             docker_env_dict,
                                                             job_props_dict,
                                                             *runner_args)

    @staticmethod
    def _run_cli_runner_stdin_media_and_env_vars(container_id: str,
                                                 media_path: str,
                                                 env_dict: Dict[str, str],
                                                 job_props_dict: Dict[str, str],
                                                 *runner_args: str) -> Dict[str, Any]:

        env_params = (f'-e{k}={v}' for k, v in env_dict.items())
        job_props = (f'-P{k}={v}' for k, v in job_props_dict.items())
        command = ['docker', 'exec', '-i', *env_params, container_id, 'runner', *runner_args, *job_props]
        print('Running job with command: ', shlex.join(command))

        with open(media_path) as media_file, \
                subprocess.Popen(command, stdin=media_file, stdout=subprocess.PIPE,
                                 text=True) as proc:
            return json.load(proc.stdout)

    def _get_image_tracks(self,
                          output_object: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._get_tracks(output_object, self.detection_type)

    @staticmethod
    def _get_tracks(output_object: Dict[str, Any],
                    detection_type: str) -> List[Dict[str, Any]]:
        return output_object['media'][0]['output'][detection_type][0]['tracks']


def main():
    args = parse_cmd_line_args()

    if os.path.isfile(args.media_path):
        image_list = [args.media_path]
    elif os.path.isdir(args.media_path):
        image_list = [os.path.abspath(os.path.join(args.media_path, f))
                      for f in os.listdir(args.media_path) if os.path.isfile(os.path.join(args.media_path, f))]
    else:
        print("Error, please provide an image file or directory.")
        return

    docker_job_list = {}
    if os.path.isfile(args.docker_image_json):
        try:
            with open(args.docker_image_json, 'r') as json_f:
                docker_job_list = json.loads(json_f.read())
        except ValueError as e:
            print("Error: {} exists as a file in this directory, but is not a proper JSON."
                  .format(args.docker_image_json))
            print(e)
            return
    else:
        job_name = args.docker_image_json.replace(":", "_ver_")
        docker_job_list[EvalFramework.EVAL_JSON_JOB_RUNS] = [{EvalFramework.EVAL_JSON_JOB_NAME:
                                                              "run_{}".format(job_name),
                                                              EvalFramework.EVAL_JSON_DOCKER_IMAGE:
                                                              args.docker_image_json,
                                                              EvalFramework.EVAL_JSON_JOB_PROPS:
                                                              {}}]
    evaluator = EvalFramework(docker_job_list, args.label_type, args.out, args.view_fiftyone)
    evaluator.process_image_jobs(image_list)
    if args.view_fiftyone:
        evaluator.launch_fiftyone_session()


def parse_cmd_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # TODO: Swap between display only mode and live jobs
    parser.add_argument('docker_image_json',
                        help='JSON file listing Docker image names with registry, tag information, '
                             'and associated job parameters. '
                             '\nA single Docker image name can be provided in place of the JSON file. '
                             '\nPlease see README.md for details. ')

    parser.add_argument('media_path',
                        help='Path to media to process. Can be either an image file or directory of images.')

    parser.add_argument('--out', default=None,
                        help='Path to store output JSON results. If left blank no JSONs are created.')

    parser.add_argument('--label-type', default="FACE",
                        help='Specify label type for detection model.')

    parser.add_argument('--view-fiftyone', dest='view_fiftyone', action='store_true',
                        help='Load datasets and labels into FiftyOne for evaluation.')

    parser.set_defaults(view_fiftyone=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()
