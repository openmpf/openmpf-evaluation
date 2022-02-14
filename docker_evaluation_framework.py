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

def get_directory(path):
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
    add_docker_vol(get_directory(path), data_dir[param], docker_vol_list)
    path = swap_parent_path(path, data_dir[param])
    argv[index+1] = path


def main(argv):
    verbose = False
    argv[0] = 'evaluation_framework.py'

    data_dir = {'out_labels': '/out/labels',
                'past_labels_dir': '/out/data/past_labels',
                'out_metrics': '/out/metrics',
                'docker_image_json': '/in/job_json',
                'media_path': '/in/media'}

    docker_vol_list = ['--mount',
                       'type=bind,source={},target={}'.format('/var/run/docker.sock', '/var/run/docker.sock'),
                       '--mount',
                       'type=bind,source={},target={}'.format('/usr/bin/docker', '/usr/bin/docker')]

    args = parse_cmd_line_args()
    arg_dict = vars(args)

    if arg_dict['subparser_name'] == 'run':
        docker_json = arg_dict['docker_image_json']
        if os.path.exists(docker_json) and os.path.isfile(docker_json):
            add_docker_vol(get_directory(docker_json), data_dir['docker_image_json'], docker_vol_list)
            new_docker_json = swap_parent_path(docker_json, data_dir['docker_image_json'])
            argv[2] = new_docker_json

    if "media_path" in arg_dict:
        media_path = arg_dict["media_path"]
        if os.path.exists(media_path) and os.path.isfile(media_path):
            # If file swap parents.
            add_docker_vol(get_directory(media_path), get_directory(media_path), docker_vol_list)
        else:
            # If directory, swap directories.
            add_docker_vol(media_path, media_path, docker_vol_list)

        if arg_dict['subparser_name'] == 'run':
            argv[3] = media_path
        else:
            argv[2] = media_path

    update_vol_command(arg_dict, argv, data_dir, docker_vol_list, "out_labels")
    update_vol_command(arg_dict, argv, data_dir, docker_vol_list, 'past_labels_dir')
    update_vol_command(arg_dict, argv, data_dir, docker_vol_list, 'out_metrics')

    subcommand = ['mpf-eval'] + argv[1:]

    if args.sudo:
        command = ['sudo', 'docker', 'run', '-a', 'STDIN', '-a', 'STDOUT', '-a', 'STDERR', '-it', '-w', '/usr/src/app']
    else:
        command = ['docker', 'run', '-a', 'STDIN', '-a', 'STDOUT', '-a', 'STDERR', '-it', '-w', '/usr/src/app']

    if args.view_fiftyone:
        command.append('-p')
        command.append('{}:{}'.format(args.fo_port, args.fo_port))

    command = command + docker_vol_list + ['mpf-evaluation-framework:latest'] + subcommand

    if verbose:
        for arg in vars(args):
            print("Parameter {} : {}".format(arg, arg_dict[arg]))
        print(argv)
        print(docker_vol_list)
        print(shlex.join(command))

    proc = subprocess.run(command, text=True, check=True)


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
    main(sys.argv)
