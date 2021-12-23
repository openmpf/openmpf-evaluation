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
import os, sys
import subprocess
import shlex
from datetime import datetime
import time
from typing import Dict, Any, List
import fiftyone as fo
import cv2
import pandas as pd
from tqdm import tqdm


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
                 docker_job_json: Dict = {},
                 detection_type: str = "FACE",
                 output_dir: str = None,
                 view_fiftyone: bool = True,
                 verbose: bool = False,
                 storage_format: str = "fiftyone",
                 dataset_mode: str = None,
                 gt_label_path: str = None,
                 dataset_subsample: int = None,
                 out_metrics: str = None,
                 sudo: bool = False,
                 dummy_jobs: bool=False,
                 seed: int = 51
                 ):
        self.dummy_jobs = dummy_jobs
        self.sudo = sudo
        self.seed = seed
        self.dataset_mode = dataset_mode
        self.gt_label_path = gt_label_path
        self.dataset_subsample = dataset_subsample
        self.verbose = verbose
        self.container_dict = {}
        self.detection_type = detection_type
        self.docker_job_list = []
        self.past_jobs_list = []
        self.view_fiftyone = view_fiftyone
        self.dataset = fo.Dataset()
        self.output_dir = output_dir
        self.out_metrics = out_metrics
        self.metrics = {}
        self.storage_format = storage_format

        self.update_docker_image_jobs(docker_job_json)


    def update_datasets(self, dir_path: str):
        """
        Given a dataset path, load prediction labels into FiftyOne dataset.
        Currently supports MPF JSON and FiftyOne labels.

        :param dir_path:
        :return:
        """

        if os.path.isdir(dir_path):
            dir_list = os.listdir(dir_path)

            for run_dir in dir_list:
                label_name = run_dir.replace('-', '_')
                self.past_jobs_list.append(label_name)
                self.metrics[label_name] = {}
                labels_path = os.path.join(dir_path, run_dir)

                if self.storage_format == "mpfjson":
                    file_list = [os.path.abspath(os.path.join(labels_path, f))
                                 for f in os.listdir(labels_path) if os.path.isfile(os.path.join(labels_path, f))]
                    for label_file in file_list:
                        with open(label_file, 'r') as json_f:
                            past_json_detection = json.loads(json_f.read())
                            filepath = past_json_detection["media"][0]["path"]

                            sample = self.dataset[filepath]
                            detections = self._convert_detections(past_json_detection, sample)
                            detections = fo.Detections(detections=detections)

                            # Save predictions to dataset
                            sample[label_name] = detections
                            if self.verbose:
                                print("Saving detection label")
                            sample.save()

                elif self.storage_format == "fiftyone":
                    data_file = os.path.abspath(os.path.join(labels_path, "data.json"))
                    labels_file = os.path.abspath(os.path.join(labels_path, "labels.json"))
                    with open(labels_file, 'r') as json_labels_f:
                        with open(data_file, 'r') as data_mapping_f:
                            json_labels = json.loads(json_labels_f.read())
                            data_mapping = json.loads(data_mapping_f.read())

                            for label in json_labels["labels"].keys():
                                filepath = data_mapping[label]
                                image_sample = self.dataset[filepath]
                                detections = []
                                ## TODO: check if converting to lower is acceptable in the long run.
                                for detection in json_labels["labels"][label]:
                                    fo_detection = fo.Detection(
                                        label=self.detection_type.lower(),
                                        bounding_box=detection["bounding_box"],
                                        confidence=detection["confidence"]
                                    )
                                    fo_detection.set_attribute_value("detectionProperties",
                                                                     detection["attributes"]["detectionProperties"])
                                    detections.append(fo_detection)
                                detections = fo.Detections(detections=detections)

                                # Save predictions to dataset
                                image_sample[label_name] = detections

                                if self.verbose:
                                    print("Saving detection label")
                                image_sample.save()

        return


    def process_image_jobs(self,
                           image_list: List[str],
                           image_path: str
                           ):
        """
        After initialization, run all jobs on given list of image file paths.
            Store any results in an optional output directory specified by the user.

        :param image_list: List of image file paths.
        :return: None
        """
        # Setup dataset
        if self.dataset_mode == "default":
            for image in image_list:
                image_sample = fo.Sample(filepath=image)
                self.dataset.add_sample(sample=image_sample)
        elif self.dataset_mode == "voc":
            original_path = os.getcwd()
            os.chdir(image_path)
            self.dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.VOCDetectionDataset,
                labels_path=self.gt_label_path
            )
            os.chdir(original_path)

        if self.dataset_subsample is not None:
            self.dataset = self.dataset.take(self.dataset_subsample, seed=self.seed).clone()

        self.dataset.default_classes = [self.detection_type.lower()]

        for image_sample in self.dataset:
            im = cv2.imread(image_sample.filepath)
            image_sample.set_field("image_dim", im.shape)
            image_sample.save()

        for job_entry in self.docker_job_list:
            job_name = job_entry[self.EVAL_JSON_JOB_NAME]
            docker_image = job_entry[self.EVAL_JSON_DOCKER_IMAGE]

            job_props_dict = job_entry.get(self.EVAL_JSON_JOB_PROPS, {})
            docker_env_dict = job_entry.get(self.EVAL_JSON_DOCKER_ENV, {})

            env_params_list = [f'-e{k}={v}' for k, v in sorted(docker_env_dict.items())]
            job_props_list = [f'-P{k}={v}' for k, v in sorted(job_props_dict.items())]
            jobs_id_new = str(job_props_list)
            image_id = docker_image.strip() + str(env_params_list)
            container_id, jobs_id = self.container_dict[image_id]
            if self.dummy_jobs and jobs_id_new != jobs_id:
                print("\n\n" + "=" * 80)
                print("\n\nNew job parameters detected for existing container, reinitializing dummy job.")
                self._run_dummy_job_container(container_id, job_props_dict)
                self.container_dict[image_id] = (container_id, jobs_id_new)

            self._process_images(job_name, job_props_dict, docker_env_dict, container_id)

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

    def launch_fiftyone_session(self, port: int = 5151):
        """
        Launch a FiftyOne session if prompted by the user.
        :return: None
        """
        if self.dataset_mode != "default":
            print("\n\nGenerating Evaluation Results:")
            for job in self.docker_job_list:
                job_name = job[self.EVAL_JSON_JOB_NAME]
                self._evaluate_fiftyone_job(job_name)

            for job_name in self.past_jobs_list:
                self._evaluate_fiftyone_job(job_name)

        session = fo.launch_app(self.dataset, remote=True, port=port)
        closing = input("Would you like to close this app (y/n):")
        while closing.lower()[0] != "y":
            closing = input("Would you like to close this app (y/n):")

    def generate_summary(self):
        if len(self.metrics) > 0:
            with open(self.out_metrics, "w+") as f:
                f.write("MPF Evaluation Summary:\n")
                f.write("Date: {}\n".format(datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")))
                for job_name, results in self.metrics.items():
                    f.write("\n\nRun Name: {}\n".format(job_name))
                    print(results)
                    if "Runtime" in results:
                        f.write("Total Image Processing Time: {} seconds.\n".format(results["Runtime"]))
                    if "FiftyOne_Eval" in results:
                        f.write("Prediction Metrics: \n\n")
                        metrics = pd.DataFrame(results["FiftyOne_Eval"]).transpose()
                        f.write(metrics.to_csv(sep='\t'))



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
                                                      '-t', 'image', '-')

        end_time = time.time()
        tracks = self._get_image_tracks(output_obj)
        if self.verbose:
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
            if self.sudo:
                command = ['sudo', 'docker', 'run', '--rm', *env_params, '-d', image_name, '-d']
            else:
                command = ['docker', 'run', '--rm', *env_params, '-d', image_name, '-d']

            if self.verbose:
                print('Starting test container with command: ', shlex.join(command))
            proc = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
            container_id = proc.stdout.strip()
            self.container_dict[image_id] = (container_id, job_id)

            print("\n\n"+"="*80)
            if self.dummy_jobs:
                print('\nRunning new dummy job for container: ', container_id)
                self._run_dummy_job_container(container_id, job_dict)


    def _shut_down_all_containers(self):
        for container_key in self.container_dict:
            if self.sudo:
                command = ('sudo', 'docker', 'stop', self.container_dict[container_key][0])
            else:
                command = ('docker', 'stop', self.container_dict[container_key][0])

            print('Stopping test container with command: ', shlex.join(command))
            subprocess.run(command, check=True)

    def _process_images(self,
                        job_name: str,
                        job_props_dict: Dict[str, str],
                        docker_env_dict: Dict[str, str],
                        container_id: str):

        if len(self.dataset) > 1 and self.verbose:
                print("Found {} images.".format(len(self.dataset)))

        index = 0

        self.dataset.classes[job_name] = [self.detection_type.lower()]
        self.dataset.save()

        dataset_start_time = datetime.now()
        runtime_start = time.time()
        if self.output_dir is not None:
            out_dir = os.path.join(self.output_dir, job_name + "_" +
                                   dataset_start_time.strftime("%Y_%m_%d-%I_%M_%S_%p"))

        for image_sample in tqdm(self.dataset, desc="{} Progress".format(job_name), disable=self.verbose):
            image = image_sample.filepath
            index += 1
            if self.verbose:
                print("\nProcessing image {}: {}".format(index, image))
            start_time = time.time()

            output_obj = self._run_cli_runner_stdin_media(container_id,
                                                          job_props_dict,
                                                          {},
                                                          image,
                                                          '-t', 'image', '-')
            end_time = time.time()
            tracks = self._get_image_tracks(output_obj)
            if self.verbose:
                print("Found detections: ", len(tracks))
                print("Run time: ", str(end_time - start_time))

            if self.view_fiftyone or self.storage_format != "mpfjson":
                # Convert detections to FiftyOne format
                detections = self._convert_detections(output_obj, image_sample)
                # Note: fo.Detections has custom attribute fields. However, sadly they are not guaranteed to be stored
                # in FiftyOne dataset format.
                detections = fo.Detections(detections=detections)

                # Save predictions to dataset
                image_sample[job_name] = detections
                if self.verbose:
                    print("Saving detection label")
                image_sample.save()

            if self.output_dir is not None and self.storage_format == "mpfjson":
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(out_dir, os.path.splitext(os.path.basename(image))[0])
                with open('{}.json'.format(output_path), 'w') as fp:
                    output_obj["media"][0]["path"] = image
                    json.dump(output_obj, fp, indent=4, sort_keys=True)

        runtime_end = time.time()

        print("\nRun {} complete. Total image processing time: {} seconds.\n\n".format(job_name,
                                                                                       runtime_end - runtime_start))
        if self.out_metrics is not None:
            self.metrics[job_name] = {"Runtime": runtime_end - runtime_start}

        if self.output_dir is not None and self.storage_format == "fiftyone":
            out_dir = os.path.join(self.output_dir,
                                   job_name + "_" + dataset_start_time.strftime("%Y_%m_%d-%I_%M_%S_%p"))
            self.dataset.export(dataset_type=fo.types.dataset_types.FiftyOneImageDetectionDataset,
                                labels_path="{}/labels.json".format(out_dir),
                                export_media="manifest",
                                data_path ="{}/data.json".format(out_dir),
                                label_field=job_name,
                                pretty_print=True,
                                overwrite=True)

    def _convert_detections(self, tracks: Dict, image_sample: fo.Sample):
        detections = []
        time_start = tracks["timeStart"]
        time_stop = tracks["timeStop"]
        job_properties = tracks["jobProperties"]
        media_metadata = tracks["media"][0]["mediaMetadata"]

        for entry in tracks["media"][0]["output"][self.detection_type]:
            for track in entry["tracks"]:
                track_properties = track["trackProperties"]
                new_track = True

                for detection in track["detections"]:
                    im_h, im_w, im_c = image_sample.get_field("image_dim")
                    x = float(detection["x"])
                    y = float(detection["y"])
                    w = float(detection["width"])
                    h = float(detection["height"])

                    rel_box = [x / im_w, y / im_h, w / im_w, h / im_h]
                    # TODO: check if converting to lowercase is acceptable in the long run.
                    fo_detection = fo.Detection(
                            label=self.detection_type.lower(),
                            bounding_box=rel_box,
                            confidence=detection["confidence"]
                        )

                    fo_detection.set_attribute_value("detectionProperties", detection["detectionProperties"])

                    # FiftyOne stores info per fo.Detection, any custom fields listed in fo.Detections are not written.
                    # TODO: Identify if another format/storage method can reduce redundancy.
                    # In the meantime, we will list this info once only per detection set.
                    if new_track:
                        new_track = False
                        fo_detection.set_attribute_value("trackProperties", track_properties)
                        fo_detection.set_attribute_value("jobProperties", job_properties)
                        fo_detection.set_attribute_value("mediaMetadata", media_metadata)
                        fo_detection.set_attribute_value("timeStart", time_start)
                        fo_detection.set_attribute_value("timeStop", time_stop)

                    detections.append(fo_detection)
        return detections

    def _run_cli_runner_stdin_media(self,
                                    container_id: str,
                                    job_props_dict: Dict[str, str],
                                    docker_env_dict: Dict[str, str],
                                    media_path: str,
                                    *runner_args: str) -> Dict[str, Any]:
        return self._run_cli_runner_stdin_media_and_env_vars(container_id,
                                                             media_path,
                                                             {},
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
        if self.sudo:
            command = ['sudo', 'docker', 'exec', '-i', *env_params, container_id, 'runner', *runner_args, *job_props]
        else:
            command = ['docker', 'exec', '-i', *env_params, container_id, 'runner', *runner_args, *job_props]
        if self.verbose:
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
    dataset_mode = "default"
    gt_label_path = ""

    if args.ground_truth_voc is not None:
        gt_label_path = args.ground_truth_voc
        dataset_mode = "voc"

    if dataset_mode != "default":
        image_list = []
    elif os.path.isfile(args.media_path):
        image_list = [args.media_path]
    elif os.path.isdir(args.media_path):
        image_list = [os.path.abspath(os.path.join(args.media_path, f))
                      for f in os.listdir(args.media_path) if os.path.isfile(os.path.join(args.media_path, f))]
    else:
        print("Error, please provide an image file or directory.")
        return

    docker_job_list = {}

    if args.subparser_name == "run":
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
                       detection_type=args.label_type,
                       output_dir=args.out_labels,
                       view_fiftyone=args.view_fiftyone,
                       verbose=args.verbose,
                       storage_format=args.prediction_storage_format,
                       dataset_mode=dataset_mode,
                       gt_label_path=gt_label_path,
                       dataset_subsample=args.dataset_subsample,
                       out_metrics=args.out_metrics,
                       dummy_jobs=args.dummy_jobs,
                       seed=args.seed,
                       sudo=args.sudo
                       ) as evaluator:

        evaluator.process_image_jobs(image_list, args.media_path)

        if args.past_labels_dir is not None:
            evaluator.update_datasets(dir_path=args.past_labels_dir)

        if args.view_fiftyone:
            evaluator.launch_fiftyone_session(args.fo_port)

        evaluator.generate_summary()


def add_common_options(parser):
    parser.add_argument('--out-labels', default=None,
                        help='Path to store output JSON results. If left blank no JSONs are created.')

    parser.add_argument('--label-type', default="FACE",
                        help='Specify label type for detection model.')

    parser.add_argument('--view-fiftyone', dest='view_fiftyone', action='store_true',
                        help='Load datasets and labels into FiftyOne for evaluation.')

    parser.add_argument('--view-fiftyone-port', '--fo-port', dest='fo_port', default=5151,
                        help='Specify port number for FiftyOne viewing app. Default port is 5151')

    parser.add_argument('--prediction-storage-format', default="fiftyone",
                        help='Storage format for predicted labels for all label directories. Current options are `fiftyone` and `mpfjson`')

    parser.add_argument('--past-labels-dir', default=None,
                        help='Optional: Storage directory for past labels. All past labels must share the same format.')

    parser.add_argument('--ground-truth-voc', dest='ground_truth_voc', default=None,
                        help='Path to VOC labels directory or dataset ground truth.')

    parser.add_argument('--dataset-subsample', '--samples', type=int, dest='dataset_subsample', default=None,
                        help='Specify number of images to randomly select.')

    parser.add_argument('--seed', dest='seed', type=int, default=51,
                        help='Specify random seed parameter for database subset behavior.')

    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Display detection results for each image.')

    parser.add_argument('--out-metrics', dest='out_metrics', default=None,
                        help='Specify output filename for evaluation and runtime metrics. '
                             'If left blank no metrics file will be generated.')

    parser.add_argument('--run-dummy-jobs', dest='dummy_jobs', default=False, action='store_true',
                        help='Run a dummy job for each unique docker setup.')

    parser.add_argument('--sudo', dest='sudo', action='store_true', default=False,
                        help='Enable sudo for Docker runs.')


def parse_cmd_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # TODO: Swap between display only mode and live jobs
    subparsers = parser.add_subparsers(title='Evaluation Framework Subcommands',
                                       dest="subparser_name",
                                       description='Valid subcommands are `run` and `view`',
                                       help='Examples:\n'
                                            '- `mpf-eval run <docker_image_json> <media_path> [optional_parameters]`'
                                            ' : Run new docker jobs on given media path.\n'
                                            '- `mpf-eval view <media_path> [optional_parameters]`'
                                            ' : View images or past mpf-eval runs. FiftyOne app is auto-enabled.\n')

    job_parser = subparsers.add_parser('run',
                                       formatter_class=argparse.RawTextHelpFormatter)
    add_common_options(job_parser)
    job_parser.add_argument('docker_image_json',
                            help='JSON file listing Docker image names with registry, tag information, '
                            'and associated job parameters. '
                            '\nA single Docker image name can be provided in place of the JSON file. '
                            '\nPlease see README.md for details. ')

    job_parser.add_argument('media_path',
                            help='Path to media to process. Can be either an image file, directory of images, '
                                 'or target directory of a given dataset.'
                                 '\n A data or label file must provided for datasets.')

    job_parser.set_defaults(view_fiftyone=False)
    job_parser.set_defaults(verbose=False)

    view_parser = subparsers.add_parser('view',
                                        formatter_class=argparse.RawTextHelpFormatter)
    add_common_options(view_parser)
    view_parser.add_argument('media_path',
                             help='Path to media to process. Can be either an image file, directory of images, '
                                  'or target directory of a given dataset.'
                                  '\n A data or label file must provided for datasets.')
    view_parser.add_argument('--disable-fiftyone', dest='view_fiftyone', action='store_false',
                             help='Disables FiftyOne viewing service.')
    view_parser.set_defaults(verbose=False)
    view_parser.set_defaults(view_fiftyone=True)

    return parser.parse_args()


if __name__ == "__main__":
    main()
