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
import shlex
import os, sys
import subprocess
from pathlib import Path


def get_parent_directory(path):
    """
    Returns the parent path of the given file or directory path.
    If path is invalid, the current directory is returned.

    :param path: Input file or directory path.
    :return: Parent of input path.
    """
    path_var = Path(path)
    if os.path.exists(path):
        if os.path.isfile(path):
            return path_var.parent.absolute()
        return path_var.absolute()
    else:
        return path_var.parent.absolute()


def swap_parent_path(path, new_parent):
    """
    Updates the given path to follow a new parent path.
    :param path: Input file or directory path.
    :param new_parent: New parent path.

    :return: Updated path.
    """
    abs_path = Path(path).absolute()
    parent = abs_path.parent
    return str(abs_path).replace(str(parent), str(new_parent))


def add_docker_vol(src, dst, vol_list):
    """
    Helper function to setup mount commands for each given source and destination path.

    :param src: Source path in host.
    :param dst: Destination path in Docker container.
    :param vol_list: A list of mount commands to update.
    :return:
    """
    vol_list.append("--mount")
    vol_list.append("type=bind,source={},target={}".format(Path(src).absolute(), dst))


def update_vol_command(arg_dict, argv, data_dir, docker_vol_list, param):
    '''
    Helper function to update job parameters.
    Sets up the Docker mount command and updates the job parameter to point to Docker mounted filepath.

    :param arg_dict: Argument dictionary generated from argparse.
    :param argv: Direct command line inputs.
    :param data_dir: List of mappings to Docker directories for each specific command line parameter.
    :param docker_vol_list: List of Docker mount commands.
    :param param: Specified parameter to update.
    :return:
    '''

    command = "--{}".format(param.replace('_', '-'))
    try:
        index = argv.index(command)
    except ValueError:
        return

    path = arg_dict[param]
    add_docker_vol(get_parent_directory(path), data_dir[param], docker_vol_list)
    path = swap_parent_path(path, data_dir[param])
    argv[index+1] = path


def add_docker_vol_command(media_path, docker_vol_list, vol_set):
    if get_parent_directory(media_path) not in vol_set:
        vol_set.add(get_parent_directory(media_path))
        add_docker_vol(get_parent_directory(media_path), get_parent_directory(media_path), docker_vol_list)


def main(argv):
    verbose = False
    argv[0] = 'evaluation_framework.py'

    data_dir_list = ['dataset_dir', 'annotations_file', 'output_metrics', 'class_mapping_file', 'predictions_dir']

    docker_vol_list = ['--mount',
                       'type=bind,source={},target={}'.format('/var/run/docker.sock', '/var/run/docker.sock'),
                       '--mount',
                       'type=bind,source={},target={}'.format('/usr/bin/docker', '/usr/bin/docker')]

    args = parse_cmd_line_args()
    arg_dict = vars(args)
    vol_set = set()
    for command in arg_dict:
        if command in data_dir_list:
            index = argv.index(arg_dict[command])
            argv[index] = Path(arg_dict[command]).absolute()

            add_docker_vol_command(arg_dict[command], docker_vol_list, vol_set)

    if args.sudo:
        command = ['sudo', 'docker', 'run', '-a', 'STDIN', '-a', 'STDOUT', '-a', 'STDERR', '-it', '-w', '/usr/src/app']
    else:
        command = ['docker', 'run', '-a', 'STDIN', '-a', 'STDOUT', '-a', 'STDERR', '-it', '-w', '/usr/src/app']

    if args.view_fiftyone:
        command.append('-p')
        command.append('{}:{}'.format(args.fo_port, args.fo_port))
    subcommand = ['mpf-eval'] + argv[1:]
    command = command + docker_vol_list + ['mpf_evaluation_framework:latest'] + subcommand
    if verbose:
        for arg in vars(args):
            print("Parameter {} : {}".format(arg, arg_dict[arg]))
        print(argv)
        print(docker_vol_list)
        print(shlex.join(command))

    proc = subprocess.run(command, text=True, check=True)


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
    parser.add_argument('--sudo', dest='sudo', action='store_true', default=False,
                        help='Enable sudo for Docker runs.')
    parser.add_argument('--view-fiftyone-port', '--fo-port', dest='fo_port', default=5151,
                        help='Specify port number for FiftyOne viewing app. Default port is 5151')

    parser.set_defaults(view_fiftyone=False)
    parser.set_defaults(verbose=False)
    parser.set_defaults(case_sensitive=False)
    return parser.parse_args()


if __name__ == "__main__":
    main(sys.argv)
