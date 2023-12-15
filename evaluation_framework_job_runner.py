#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2023 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2023 The MITRE Corporation                                      #
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
import time
from typing import Dict, Any, List
import logging
#import cv2
#import pandas as pd
#from tqdm import tqdm
from xml.etree import ElementTree

class EvalFramework:
    EVAL_JSON_JOB_RUNS = "jobRuns"
    EVAL_JSON_JOB_NAME = "jobName"
    EVAL_JSON_DOCKER_IMAGE = "dockerImage"
    EVAL_JSON_JOB_PROPS = "jobProperties"
    EVAL_JSON_DOCKER_ENV = "dockerEnvironment"

    dummy_data_path = os.path.dirname(os.path.abspath(__file__))
    EVAL_DUMMY_FILE = dummy_data_path + "/data/images/meds-af-S419-01_40deg.jpg"

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._shut_down_all_containers()

    def __init__(self,
                 docker_job_json: str = None,
                 detection_type: str = None,
                 output_dir: str = None,
                 verbose: bool = False,
                 file_type: str = None,
                 gt_label_path: str = None,
                 dataset_subsample: int = None,
                 out_metrics: str = None,
                 sudo: bool = False,
                 dummy_jobs: bool = False,
                 repeat_run: int = 10,
                 fail_media: str = None,
                 blank_media: str = None,
                 seed: int = 51,
                 cpu: bool = False,
                 disable_shutdown: bool = False,
                 repeat_forever: bool = False,
                 ):

        self.disable_shutdown = disable_shutdown
        self.repeat_forever = repeat_forever
        self.repeat_run = repeat_run
        self.fail_media = fail_media
        self.blank_media = blank_media


        self.detection_type = detection_type
        self.dummy_jobs = dummy_jobs
        self.sudo = sudo
        self.seed = seed
        self.file_type = file_type
        self.gt_label_path = gt_label_path
        self.dataset_subsample = dataset_subsample
        self.verbose = verbose
        self.container_dict = {}
        self.docker_job_list = []
        self.past_jobs_list = []
        self.output_dir = output_dir
        self.out_metrics = out_metrics
        self.metrics = {}
        self.cpu = cpu
        self.docker_job_json = docker_job_json

        if self.blank_media is not None:
            os.makedirs(self.blank_media, exist_ok=True)
        if self.fail_media is not None:
            os.makedirs(self.fail_media, exist_ok=True)
        self.update_docker_image_jobs(docker_job_json)

    def process_media_jobs(self,
                           media_list: List[str],
                           ):
        """
        After initialization, run all jobs on given list of image file paths.
            Store any results in an optional output directory specified by the user.

        :param media_list: List of  file paths.
        :return: None
        """

        # Setup dataset
        for job_entry in self.docker_job_list:
            job_name = job_entry[self.EVAL_JSON_JOB_NAME]
            docker_image = job_entry[self.EVAL_JSON_DOCKER_IMAGE]

            self.metrics[job_name] = {'DOCKER_IMAGE': docker_image,
                                         'SUCCESSFUL_RUNS': 0,
                                         'RUNTIME': 0.0,
                                         'BLANK_OUTPUT':0,
                                         'FAILED_RUNS':0,
                                         'FAILED_MEDIA_RUNTIME':0.0,
                                         'BLANK_MEDIA_FILELIST':[],
                                         "FAILED_MEDIA_FILELIST":[]}

            job_props_dict = job_entry.get(self.EVAL_JSON_JOB_PROPS, {})
            docker_env_dict = job_entry.get(self.EVAL_JSON_DOCKER_ENV, {})

            env_params_list = [f'-e{k}={v}' for k, v in sorted(docker_env_dict.items())]
            job_props_list = [f'-P{k}={v}' for k, v in sorted(job_props_dict.items())]
            jobs_id_new = str(job_props_list)
            image_id = docker_image.strip() + str(env_params_list)
            container_id, jobs_id, *params = self.container_dict[image_id]
            if self.dummy_jobs and jobs_id_new != jobs_id:
                print("\n\n" + "=" * 80)
                print("\n\nNew job parameters detected for existing container, reinitializing dummy job.")
                self._run_dummy_job_container(container_id, job_props_dict)
                self.container_dict[image_id] = (container_id, jobs_id_new) + params

            self._process_media(job_name, media_list, job_props_dict, docker_env_dict, image_id)

    def _evaluate_fiftyone_job(self, job_name):
        results = self.dataset.evaluate_detections(
            pred_field=job_name,
            gt_field="ground_truth",
            eval_key="eval_" + job_name,
            compute_mAP=True,
        )
        print("\nEvaluation Results for Job: {}".format(job_name))
        results.print_report(classes=[self.detection_type.lower()])

        if self.out_metrics is not None:
            self.metrics[job_name]["FiftyOne_Eval"] = results.report(classes=[self.detection_type.lower()])
        return results

    # TODO append results if linked by run info.
    # TODO convert to JSON format.
    def generate_summary(self):
        if len(self.metrics) > 0:
            with open(self.out_metrics, "w+") as f:
                f.write("MPF Evaluation Summary:\n")
                f.write("Date: {}\n".format(datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")))
                for job_name, results in self.metrics.items():
                    f.write("\n\nRun Name: {}\n".format(job_name))
                    print(results)
                    if "TOTAL_RUNTIME" in results:
                        f.write("Total Processing Time: {} seconds.\n".format(results["TOTAL_RUNTIME"]))
                        f.write("Processing Time for Successful Runs: {} seconds\n".format(results["RUNTIME"]))
                        f.write("Number of Successful Runs: {} \n".format(results["SUCCESSFUL_RUNS"]))
                        f.write("Number of Blank JSONS: {} \n".format(results["BLANK_OUTPUT"]))
                        f.write("Processing Time for Failed Runs: {} seconds\n".format(results["FAILED_MEDIA_RUNTIME"]))
                        f.write("Number of Failed Runs: {} \n".format(results["FAILED_RUNS"]))
                        f.write("\nList of failed media:{}\n".format(", ".join(results["FAILED_MEDIA_FILELIST"])))
                        f.write("\nList of blank media:{}\n".format(", ".join(results["BLANK_MEDIA_FILELIST"])))

    def update_docker_image_jobs(self, docker_job_json: Dict):
        """
        Initializes all unique Docker job containers referenced in job JSON file.
        :param docker_job_json: Job JSON file containing MPF Docker component images and associated job parameters.
        :return: None
        """
        if len(docker_job_json) > 0:
            for job_entry in docker_job_json[self.EVAL_JSON_JOB_RUNS]:
                docker_image = job_entry[self.EVAL_JSON_DOCKER_IMAGE]
                docker_env_dict = job_entry.get(self.EVAL_JSON_DOCKER_ENV, {})
                job_dict = job_entry.get(self.EVAL_JSON_JOB_PROPS, {})
                self._set_up_containers(docker_image, docker_env_dict, job_dict)

            self.docker_job_list += docker_job_json[self.EVAL_JSON_JOB_RUNS]

    def _run_dummy_job_container(self, container_id: str,  job_dict: Dict):
        start_time = time.time()
        output_obj = self._run_cli_runner_stdin_media(container_id,
                                                      job_dict,
                                                      {},
                                                      self.EVAL_DUMMY_FILE,
                                                      '-t', self.file_type, '-')

        end_time = time.time()

        if self.verbose:
            if self.detection_type is not None:
                tracks = self._get_media_tracks(output_obj)
                print("Found detections (Dummy Run): ", len(tracks))
            print("Dummy run time: ", str(end_time - start_time))
        print("Dummy run complete!")
        print("="*80, "\n")

    def _set_up_containers(self, image_name: str, env_dict: Dict, job_dict: Dict):
        env_params = (f'-e{k}={v}' for k, v in sorted(env_dict.items()))
        env_params_list = [f'-e{k}={v}' for k, v in sorted(env_dict.items())]
        job_props_list = [f'-P{k}={v}' for k, v in sorted(job_dict.items())]

        image_id = image_name.strip() + str(env_params_list)
        job_id = str(job_props_list)

        if image_id not in self.container_dict:
            self.start_container(image_id, job_id, job_dict, env_params, image_name)


    def start_container(self, image_id, job_id, job_dict, env_params, image_name):
        if self.cpu:
            command = ['docker', 'run', '--runtime=runc', '--rm', *env_params, '-d', image_name, '-d']
        else:
            command = ['docker', 'run', '--rm', *env_params, '-d', image_name, '-d']
        if self.sudo:
            command = ['sudo'] + command
        if self.verbose:
            print('Starting test container with command: ', shlex.join(command))
        proc = subprocess.run(command, stdout=subprocess.PIPE, check=True)
        container_id = proc.stdout.strip()

        print(f'Container ID : {container_id}')
        self.container_dict[image_id] = (container_id, job_id, job_dict, env_params, image_name)

        print("\n\n"+"="*80)
        if self.dummy_jobs:
            print('\nRunning new dummy job for container: ', container_id)
            self._run_dummy_job_container(container_id, job_dict)

    def stop_container(self, container_key):
        if self.sudo:
            command = ('sudo', 'docker', 'stop', self.container_dict[container_key][0])
        else:
            command = ('docker', 'stop', self.container_dict[container_key][0])
        if self.verbose:
            str_command = [str(x) for x in command]
            print('Stopping test container with command: ', shlex.join(str_command))
        subprocess.run(command, check=True)


    def stop_and_restart_container(self, container_key):
        self.stop_container(container_key)

        job_id = self.container_dict[container_key][1]
        job_dict = self.container_dict[container_key][2]
        env_params = self.container_dict[container_key][3]
        image_name = self.container_dict[container_key][4]

        self.start_container(container_key, job_id, job_dict, env_params, image_name)


    def _shut_down_all_containers(self):
        if not self.disable_shutdown:
            for container_key in self.container_dict:
                if self.sudo:
                    command = ('sudo', 'docker', 'stop', self.container_dict[container_key][0])
                else:
                    command = ('docker', 'stop', self.container_dict[container_key][0])
                if self.verbose:
                    str_command = [str(x) for x in command]
                    print('Stopping test container with command: ', shlex.join(str_command))
                subprocess.run(command, check=True)

    def symlink_file(media, out_dir):
        out_media = os.path.join(out_dir, os.path.basename(media))
        os.symlink(media, out_media)

    def _process_media(self,
                       job_name: str,
                       media_list: List[str],
                       job_props_dict: Dict[str, str],
                       docker_env_dict: Dict[str, str],
                       image_id: str):

        if self.verbose:
            print("Found {} files.".format(len(media_list)))

        index = 0
        dataset_start_time = datetime.now()
        runtime_start = time.time()

        if self.output_dir is not None:
            out_dir = os.path.join(self.output_dir, job_name + "_" +
                                   dataset_start_time.strftime("%Y_%m_%d-%I_%M_%S_%p"))
        run_count = 0
        fail_count = 0
        while True:
            for media in media_list:
                index += 1
                run_count += 1
                if self.verbose:
                    print("\nProcessing file {}: {}".format(index, media))
                successful_run = False
                output_obj = None
                for i in range(self.repeat_run):
                    container_id = self.container_dict[image_id][0]
                    start_time = time.time()
                    try:
                        output_obj = self._run_cli_runner_stdin_media(container_id,
                                                                    job_props_dict,
                                                                    docker_env_dict,
                                                                    media,
                                                                    '-t', self.file_type, '-')

                    except ValueError:
                        end_time = time.time()
                        logging.exception("message")
                        print("\nError Processing file {}: {}".format(index, media))
                        print("Run time: ", str(end_time - start_time))
                        self.stop_and_restart_container(image_id)
                        continue
                    successful_run = True
                    end_time = time.time()
                    break

                if successful_run:
                    self.metrics[job_name]['SUCCESSFUL_RUNS'] += 1
                    self.metrics[job_name]['RUNTIME'] += end_time - start_time
                else:
                    self.metrics[job_name]['FAILED_RUNS'] += 1
                    self.metrics[job_name]['FAILED_MEDIA_RUNTIME'] += end_time - start_time
                    self.metrics[job_name]['FAILED_MEDIA_FILELIST'].append(media)
                    if self.fail_media is not None:
                        self.symlink_file(media, self.fail_media)


                if self.verbose:
                    if self.detection_type is not None:
                        tracks = self._get_media_tracks(output_obj)
                        print("Found detections: ", len(tracks[0]))
                        print("Run time: ", str(end_time - start_time))

                if output_obj is None:
                    self.metrics[job_name]['BLANK_OUTPUT'] += 1
                    self.metrics[job_name]['BLANK_MEDIA_FILELIST'].append(media)
                    if self.blank_media is not None:
                        self.symlink_file(media, self.blank_media)
                    print("Found no detections: ", len(tracks))
                    print("Run time: ", str(end_time - start_time))
                    continue

                tracks = self._get_media_tracks(output_obj)
                for track in tracks:
                    detections = track['detections']
                    print(f'Run #{run_count}: Number of detections = {len(detections)}')
                    if len(detections) < 1:
                        fail_count += 1
                        print(f"FAILED: found {len(detections)} detections ==> {fail_count} failures in {run_count} runs")

                # TODO: Modify to access and also store output info:
                # Output JSON is held in this variable: output_obj
                # Below the output JSON gets saved to the output directory.
                # Updated to allow for repeats.
                if self.output_dir is not None:
                    os.makedirs(out_dir, exist_ok=True)
                    output_path = os.path.join(out_dir, os.path.splitext(os.path.basename(media))[0])


                    # NOTE: If manual toggle to true, ALL run outputs are stored for every run in repeat_forever MODE
                    # From testing, this will create a huge output folder very quickly.
                    store_everything = False
                    if self.repeat_forever and store_everything:
                        # Mark output job files with a time label, as the same media gets reprocessed.
                        time_str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                        output_path = os.path.join(out_dir, os.path.splitext(os.path.basename(media))[0]+'_date_'+time_str)

                    with open('{}.json'.format(output_path), 'w') as fp:
                        if "media" in output_obj:
                            output_obj["media"][0]["path"] = media
                        else:
                            print("Error: Media path not specified in output JSON, adding description.")
                            output_obj["media"] = [{"path": media}]
                        json.dump(output_obj, fp, indent=4, sort_keys=True)

            if not self.repeat_forever:
                break

        runtime_end = time.time()
        if self.verbose:
            print("\nRun {} complete. Total image processing time: {} seconds.\n\n".format(job_name,
                                                                                       runtime_end - runtime_start))
        if self.out_metrics is not None:
            self.metrics[job_name]["TOTAL_RUNTIME"] =  runtime_end - runtime_start

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

    def _run_cli_runner_stdin_media_and_env_vars(self,
                                                 container_id: str,
                                                 media_path: str,
                                                 env_dict: Dict[str, str],
                                                 job_props_dict: Dict[str, str],
                                                 *runner_args: str) -> Dict[str, Any]:
        env_params = (f'-e{k}={v}' for k, v in env_dict.items())
        job_props = (f'-P{k}={v}' for k, v in job_props_dict.items())

        command = ['docker', 'exec', '-i', *env_params, container_id, 'runner', *runner_args, *job_props]
        if self.sudo:
            command = ['sudo'] + command
        if self.verbose:
            str_command = [str(x) for x in command]
            print('Running job with command: ', shlex.join(str_command))

        with open(media_path) as media_file, \
                subprocess.Popen(command, stdin=media_file, stdout=subprocess.PIPE,
                                 text=True) as proc:
            return json.load(proc.stdout)

    def _get_media_tracks(self,
                          output_object: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._get_tracks(output_object, self.detection_type)

    @staticmethod
    def _get_tracks(output_object: Dict[str, Any],
                    detection_type: str) -> List[Dict[str, Any]]:
        return output_object['media'][0]['output'][detection_type][0]['tracks']


def main():
    args = parse_cmd_line_args()
    if args.ground_truth_voc is not None:
        media_list = []
        voc_args_list = [os.path.join(args.ground_truth_voc, f)
                         for f in os.listdir(args.ground_truth_voc) if os.path.isfile(os.path.join(args.ground_truth_voc, f))]
        media_list = [os.path.join(media_path, extract_path_voc(f)) for f in voc_args_list]
    elif os.path.isfile(args.media_path):
        media_list = [args.media_path]
    elif os.path.isdir(args.media_path):
        media_list = [os.path.join(args.media_path, f)
                           for f in os.listdir(args.media_path) if os.path.isfile(os.path.join(args.media_path, f))]
    else:
        print("Error, please provide a file or directory.")
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

    with EvalFramework(docker_job_json=docker_job_list,
                       detection_type=args.detection_type,
                       output_dir=args.out_labels,
                       verbose=args.verbose,
                       file_type=args.file_type,
                       gt_label_path=args.ground_truth_voc,
                       dataset_subsample=args.dataset_subsample,
                       out_metrics=args.out_metrics,
                       dummy_jobs=args.dummy_jobs,
                       seed=args.seed,
                       sudo=args.sudo,
                       cpu=args.cpu,
                       fail_media = args.fail_media,
                       blank_media = args.blank_media,
                       repeat_run = args.repeat_run,
                       disable_shutdown = args.disable_shutdown,
                       repeat_forever = args.repeat_jobs_forever,
                       ) as evaluator:
        evaluator.process_media_jobs(media_list)
        evaluator.generate_summary()

    if args.run_fiftyone:
        if args.sudo:
            command_list = ['sudo','python3','docker_evaluation_framework.py', 'view', args.media_path, '--past-labels-dir', args.out_labels]
        else:
            command_list = ['sudo','python3','docker_evaluation_framework.py', 'view', args.media_path, '--past-labels-dir', args.out_labels]

        if "FiftyOneCommandLineArguments" in docker_job_list:
            command_list = command_list + [docker_job_list["FiftyOneCommandLineArguments"]]
        print(command_list)
        proc = subprocess.run(command_list, text=True, check=True)


def add_common_options(parser):
    parser.add_argument('--out-labels', default=None,
                        help='Path to store output JSON results. If left blank no JSONs are created.')

    parser.add_argument('--repeat-jobs-forever', action='store_true', default=False,
                        help='If set, the framework will loop FOREVER on available media. Runs will have a time-label added')

    parser.add_argument('--disable-shutdown', action='store_true', default=False,
                        help='If set, disable shutdown of MPF Docker components.')

    parser.add_argument('--cpu', dest='cpu', default=False, action='store_true',
                        help='Set runtime to runc.')

    parser.add_argument('--media-file-type', dest='file_type', default="image",
                        help='Specifies input media type. Currently supports `image` and `video`.')

    parser.add_argument('--ground-truth-voc', dest='ground_truth_voc', default=None,
                        help='Path to VOC labels directory or dataset ground truth.')

    parser.add_argument('--dataset-subsample', '--samples', type=int, dest='dataset_subsample', default=None,
                        help='Specify number of images to randomly select.')

    parser.add_argument('--seed', dest='seed', type=int, default=51,
                        help='Specify random seed parameter for database subset behavior.')

    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Display detection results for each image.')

    parser.add_argument('--run-dummy-jobs', dest='dummy_jobs', default=False, action='store_true',
                        help='Run a dummy job for each unique docker setup.')

    parser.add_argument('--out-metrics', dest='out_metrics', default="metrics_run.txt",
                        help='Specify output filename for evaluation and runtime metrics. '
                             'If left blank no metrics file will be generated.')

    parser.add_argument('--out-failed-media', dest='fail_media', default=None,
                        help='Specify optional directory for failed media. '
                        'Any media runs that consistently produce a system error are softlinked here.')

    parser.add_argument('--out-blank-media', dest='blank_media', default=None,
                        help='Specify optional directory for blank media. '
                        'Any media runs that produce an empty JSON are softlinked here.')

    parser.add_argument('--repeat-failed-runs', dest='repeat_run', default=10,
                        help='Specify number of times a container should be restarted on a failed run. '
                        'By default the evaluator will attempt to restart a run ten times.')

    parser.add_argument('--detection-type', dest='detection_type', default=None,
                        help='Specifies detection type for the given tracks. '
                             'If left blank no track metrics are reported.')

    parser.add_argument('--sudo', dest='sudo', action='store_true', default=False,
                        help='Enable sudo for Docker runs.')

    parser.add_argument('--run-fiftyone',  default=False, action='store_true',
                        help='Enable a follow-up run using FiftyOne.'
                             'Specify job parameters using the docker_image_json file.')

    parser.set_defaults(verbose=False)


def parse_cmd_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('docker_image_json',
                            help='JSON file listing Docker image names with registry, tag information, '
                            'and associated job parameters. '
                            '\nA single Docker image name can be provided in place of the JSON file. '
                            '\nPlease see README.md for details. ')

    parser.add_argument('media_path',
                            help='Path to media to process. Can be either an image file, directory of images, '
                                 'or target directory of a given dataset.'
                                 '\n A data or label file must provided for datasets.')
    add_common_options(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
