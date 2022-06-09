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
import fiftyone as fo
import csv
import numpy as np
import json
import os
import sys
import random
import datetime
import pandas as pd

from pathlib import Path


class EvalFramework:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return

    def __init__(self,
                 dataset_dir: str = None,
                 annotations_file: str = None,
                 predictions_dir: str = None,
                 dataset_type: str = None,
                 output_metrics: str = None,
                 view_fiftyone: bool = True,
                 class_mapping_file: str = None,
                 case_sensitive: bool = True,
                 dataset_sample: float = -1,
                 verbose: bool = False,
                 ):
        self.dataset_dir = dataset_dir
        self.annotations_file = annotations_file
        self.predictions_dir = predictions_dir
        # TODO: Add support for other dataset types
        self.dataset_type = dataset_type
        self.class_mapping_file = class_mapping_file

        self.view_fiftyone = view_fiftyone
        self.case_sensitive = case_sensitive
        self.output_metrics = output_metrics

        self.prediction_tags = []

        self.dataset_sample = dataset_sample

        self.class_to_string = {}
        self.class_id_list = []
        self.annotations = []
        self.dataset = fo.Dataset()
        self.results = {}
        self.verbose = verbose

        if self.dataset_type == "imagenet":
            self._init_imagenet_data()

    def _init_imagenet_data(self):
        """
        Initialize the ImageNet class labels list and ImageNet ground truth annotations.
        :return:
        """

        self._setup_class_lists_imagenet()
        self._setup_imagenet_annotations()

    def _setup_class_lists_imagenet(self):
        """
        Initialize the ImageNet class labels list.
        :return:
        """

        with open(self.class_mapping_file, "r+") as f:
            for id_string in f.read().split("\n"):
                id_list = id_string.split(" ")
                self.class_to_string[id_list[0]] = id_string[len(id_list[0]):].strip()
                if not self.case_sensitive:
                    self.class_to_string[id_list[0]] = self.class_to_string[id_list[0]].lower()
                self.class_id_list.append(self.class_to_string[id_list[0]])

    def _setup_imagenet_annotations(self):
        """
        Initialize the ImageNet groundtruth dataset annotations.
        Based on code by Zachary Cafego.
        :return:
        """
        with open(self.annotations_file) as f:
            csv_reader = csv.reader(f, delimiter=",")
            line_count = 0
            for row in csv_reader:
                line_count += 1
                if line_count > 1:
                    filename = row[0]
                    classification = row[1].split()[0]
                    filepath = os.path.join(self.dataset_dir, filename+'.JPEG')
                    if not os.path.exists(filepath):
                        continue
                    self.annotations.append(dict(filename=filename, filepath=filepath, classification=classification))

    def setup_dataset(self):
        """
        Setup ground truth dataset for image classification.
        :return:
        """
        samples = []
        for annotation in self.annotations:
            filepath = annotation["filepath"]
            label = annotation["classification"]
            sample = fo.Sample(filepath=filepath)

            # Store classification in a field name of your choice
            label = self.class_to_string[label]
            if not self.case_sensitive:
                label = label.lower()

            sample["ground_truth"] = fo.Classification(label=label)
            samples.append(sample)

        dataset_samples = samples

        # Trim dataset if user enters a fraction or non-negative number of files.
        if self.dataset_sample > 0:
            if self.dataset_sample < 1:
                dataset_sample = len(samples) * self.dataset_sample
            dataset_samples = random.sample(samples, dataset_sample)
        self.dataset.add_samples(dataset_samples)

    def _create_logits(self, top_preds, class_list):
        """
        Create a logits prediction vector given a list of top predictions.

        :param top_preds:
        :param class_list:
        :return:
        """
        logits = np.full(len(class_list), sys.float_info.min)
        for pred in top_preds:
            # print(pred)
            logits[class_list.index(pred[0])] = pred[1]
        return np.log(logits)

    def add_predictions(self):
        """
        Add predictions from each job to dataset.
        :return:
        """
        # Construct a logits array and fill in top predictions by model.
        # All missing predictions are reconstructed with lowest possible values.
        predictions_list = [f.path for f in os.scandir(self.predictions_dir) if f.is_dir()]

        for pred in predictions_list:

            pred_tag = pred.split("/")[-1]
            self.prediction_tags.append(pred_tag)
            for sample in self.dataset:
                filepath = sample["filepath"]
                filename = filepath.split("/")[-1][:-5]
                jfile = os.path.join(pred, filename + ".json")
                if os.path.exists(jfile):
                    if str(filepath) in self.dataset:
                        with open(jfile) as f:
                            output = json.load(f)
                            top_classes = output["media"][0]["output"]["CLASS"][0]["tracks"][0]["detections"][0][
                                "detectionProperties"]["CLASSIFICATION LIST"]
                            top_conf = output["media"][0]["output"]["CLASS"][0]["tracks"][0]["detections"][0][
                                "detectionProperties"]["CLASSIFICATION CONFIDENCE LIST"]
                            top_conf = [float(f.strip(" tensor()")) for f in top_conf.split(";")]
                            if not self.case_sensitive:
                                top_classes = [x.lower() for x in top_classes.split("; ")]
                            else:
                                top_classes = top_classes.split("; ")

                            logits = self._create_logits(zip(top_classes, top_conf), self.class_id_list)

                            sample = self.dataset[str(filepath)]
                            sample[pred_tag] = fo.Classification(
                                label=top_classes[0],
                                confidence=top_conf[0],
                                logits=logits,
                            )
                            sample.save()
                    else:
                        print("Warning: Missing Prediction for " + filepath)
                        self.dataset.remove(filepath)

    def evaluate_fiftyone(self):
        """
        Evaluate top k predictions for each job run.
        K values range from 1-10 and can be adjusted for further inferences.
        :return:
        """

        #TODO: Continue updating evaluation framework metrics.
        k_pred = [1, 3, 5, 10]
        for pred in self.prediction_tags:
            results_dict = {}
            for k in k_pred:
                results = self.dataset.evaluate_classifications(
                    pred,
                    gt_field="ground_truth",
                    eval_key="eval_" + pred.replace("-","_") + "_k_{}".format(k),
                    method="top-k",
                    classes=self.class_id_list,
                    k=k,
                )
                results_dict[k] = results.report(classes=self.class_id_list)

            if self.output_metrics is not None:
                self.results[pred] = results_dict

    def launch_fiftyone_session(self, port: int = 5151):
        """
        Launch a FiftyOne session if prompted by the user.
        :return: None
        """
        session = fo.launch_app(self.dataset, remote=True, port=port)
        closing = input("Would you like to close this app (y/n):")
        while closing.lower()[0] != "y":
            closing = input("Would you like to close this app (y/n):")

    def generate_summary(self):
        """
        Print out summary of metric results.
        :return:
        """
        Path(self.output_metrics).parent.mkdir(parents=True, exist_ok=True)
        if len(self.results) > 0:
            with open(self.output_metrics, "w+") as f:
                f.write("MPF Evaluation Summary:\n")
                f.write("Date: {}\n".format(datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")))

                for job_name, result in self.results.items():
                    f.write("\n\nRun Name: {}\n".format(job_name))
                    for k in result:
                        f.write("Prediction Metrics (top-k = {}): \n\n".format(k))
                        if self.verbose:
                            metrics = pd.DataFrame(result[k]).transpose()
                        else:
                            aggregate_res = ["micro avg", "macro avg", "weighted avg"]
                            metrics = {}
                            for agg in aggregate_res:
                                metrics[agg] = result[k][agg]
                            metrics = pd.DataFrame(metrics).transpose()
                        f.write(metrics.to_csv(sep="\t"))
                        f.write("\n")


def main():
    args = parse_cmd_line_args()
    with EvalFramework(dataset_dir=args.dataset_dir,
                       annotations_file=args.annotations_file,
                       predictions_dir=args.predictions_dir,
                       dataset_type=args.dataset_type,
                       class_mapping_file=args.class_mapping_file,
                       output_metrics=args.output_metrics,
                       view_fiftyone=args.view_fiftyone,
                       case_sensitive=args.case_sensitive,
                       dataset_sample=args.dataset_subsample,
                       verbose=args.verbose
                       ) as evaluator:

        evaluator.setup_dataset()
        evaluator.add_predictions()
        evaluator.evaluate_fiftyone()
        if args.output_metrics is not None:
            evaluator.generate_summary()
        if args.view_fiftyone:
            evaluator.launch_fiftyone_session(port=args.fo_port)


def parse_cmd_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataset_dir", help="Path to dataset directory containing image files.")
    parser.add_argument("annotations_file", help="Path to dataset annotations file or directory (csv for ImageNet).")
    parser.add_argument("class_mapping_file", help="Path to class labels file (classes.txt for ImageNet).")
    parser.add_argument("predictions_dir", help="Path to predictions directory containing job run subdirectories.")

    parser.add_argument("--dataset-type", dest="dataset_type", default="imagenet",
                        help="Specifies supported dataset type (currently imagenet).")
    parser.add_argument("--dataset-subsample", dest="dataset_subsample", default=-1,
                        help="Specifies number of files to process in dataset. "
                             "Set below 0 to process the entire dataset, or between 0-1 to process a percentage of the dataset.")

    parser.add_argument("--output-metrics", default=None,
                        help="Path to store output prediction results.")

    parser.add_argument("--view-fiftyone", dest="view_fiftyone", action="store_true",
                        help="Enables FiftyOne viewing service.")
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                        help="Enables reporting of all individual class metrics.")
    parser.add_argument("--case-sensitive", dest="case_sensitive", action="store_true",
                        help="Enables FiftyOne viewing service.")

    parser.add_argument('--view-fiftyone-port', '--fo-port', dest='fo_port', default=5151,
                        help='Specify port number for FiftyOne viewing app. Default port is 5151')

    parser.set_defaults(view_fiftyone=False)
    parser.set_defaults(verbose=False)
    parser.set_defaults(case_sensitive=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()
