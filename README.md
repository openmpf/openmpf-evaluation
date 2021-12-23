# OpenMPF Evaluation Framework

Welcome to the Open Media Processing Framework (OpenMPF) Evaluation Framework Project!

## What is the OpenMPF?

OpenMPF provides a platform to perform content detection and extraction on bulk multimedia, enabling users to analyze,
search, and share information through the extraction of objects, keywords, thumbnails, and other contextual data.

OpenMPF enables users to build configurable media processing pipelines, enabling the rapid development and deployment of
analytic algorithms and large-scale media processing applications.

### Search and Share

Simplify large-scale media processing and enable the extraction of meaningful content

### Open API

Apply cutting-edge algorithms such as face detection and object classification

### Flexible Architecture

Integrate into your existing environment or use OpenMPF as a standalone application

## Overview

This repository contains code for the OpenMPF evaluation framework and related files. The framework is used for
generating metrics such as speed, accuracy, precision, recall, etc., of OpenMPF component algorithms, depending on the
algorithm type (e.g. face detection, text detection, speech detection, object detection and classification, etc.).

## Where Am I?

- [Parent OpenMPF Project](https://github.com/openmpf/openmpf-projects)
- [OpenMPF Core](https://github.com/openmpf/openmpf)
- Components
    * [OpenMPF Standard Components](https://github.com/openmpf/openmpf-components)
    * [OpenMPF Contributed Components](https://github.com/openmpf/openmpf-contrib-components)
- Component APIs:
    * [OpenMPF C++ Component SDK](https://github.com/openmpf/openmpf-cpp-component-sdk)
    * [OpenMPF Java Component SDK](https://github.com/openmpf/openmpf-java-component-sdk)
    * [OpenMPF Python Component SDK](https://github.com/openmpf/openmpf-python-component-sdk)
- [OpenMPF Build Tools](https://github.com/openmpf/openmpf-build-tools)
- [OpenMPF Docker](https://github.com/openmpf/openmpf-docker)
- [OpenMPF Web Site Source](https://github.com/openmpf/openmpf.github.io)
- [OpenMPF Evaluation](https://github.com/openmpf/openmpf-evaluation) ( **You are here** )

## Getting Started

The evaluation framework uses the [CLI Runner](https://github.com/openmpf/openmpf-docker/blob/master/CLI_RUNNER.md) to
execute jobs using pre-built OpenMPF Docker images, such as those on [Docker Hub](https://hub.docker.com/u/openmpf).
This framework can be run in two ways: as a standalone script (assuming FiftyOne is properly installed) or
within a custom built Docker image. Instructions for building and running the evaluation framework in Docker can be
found below.


### Quick Start

To directly use the evaluation framework script (located in evaluation_framework directory) run:

`python3 evaluation_framework.py <docker-image-name-with-registry-and-tag> <path-to-media-file-or-dir>`

For example, to run OCV face detection on the sample image file, run:

`python3 evaluation_framework.py openmpf_ocv_face_detection:latest /home/mpf/openmpf-projects/openmpf-evaluation/data/meds-af-S419-01_40deg.jpg`

### Building and Running the Evaluation Framework in Docker:

To build the mpf_evaluation_framework image run the following commands:

```
chmod +x build.sh
sudo build.sh
```
Afterwards you can use the `docker_evaluation_framework.py` script in the same way as the original `evaluation_framework.py`:
`python3 docker_evaluation_framework.py openmpf_ocv_face_detection:latest /home/mpf/openmpf-projects/openmpf-evaluation/data/meds-af-S419-01_40deg.jpg`


### Running with Multiple Docker Images

This component will also accept multiple docker images with customized job parameters and can process image directories
using a job JSON file in place of a single docker image input:

`python3 evaluation_framework.py <docker-job-json-file> <path-to-media-file-or-dir>`

For example:

`python3 evaluation_framework.py ./sample_jobs.json ./data/images`

### Job JSON File Overview:
Each custom JSON job file should be organized as follows:
- All job runs should be stored in the `jobRuns` array.
  Each job run is a JSON object with the following three fields:
    - `jobName` : Specifies name of particular job run. All `jobNames` in this JSON file must be unique.
    - `dockerImage`: Specifies Docker image and registry tag. The same image can be reused across job run objects.
    - `jobProperties`: A JSON object containing custom job properties for the associated OpenMPF docker component.

As mentioned, users can specify the same Docker image for multiple job runs. The component will initialize
a Docker container for each unique image listed across the job JSON file and reuse them as needed.

Example Job JSON below:
```
{
    "jobRuns": [
        {
            "jobName": "<Docker_Image_1_Run_Name>",
            "dockerImage": "<openmpf_docker_image_1>:<image_version>"
        },
        {
            "jobName": "<Docker_Image_1_With_Custom_Parameters>",
            "dockerImage": "<openmpf_docker_image_1>:<image_version>"
            "jobProperties": {
                "<SAMPLE_PARAMETER_1>": "<input_parameter>",
            }
        },
                {
            "jobName": "<Docker_Image_n_Run_Name>",
            "dockerImage": "<openmpf_docker_image_n>:<image_version>"
            "jobProperties": {
                "<SAMPLE_PARAMETER_1>": "<input_parameter>",
            }
        },
    ]
}
```

### Storing Results and Viewing Labels on FiftyOne:

To store the output JSON labels into a custom labels folder, add the following parameter to the job run:
`--out <output_label_dir>`

The outputs JSONs will be directed to `<output_label_dir>/<job_run_with_timestamp_dir>` under unique subdirectories
referencing the individual `jobRun` names and ending timestamps.

To view these results using the FiftyOne application add the following parameter:
"--view-fiftyone`

This will open the FiftyOne app which can be viewed in an open browser at `localhost:5151` by default.

## Project Website

For more information about OpenMPF, including documentation, guides, and other material, visit
our [website](https://openmpf.github.io/).

## Project Workboard

For a latest snapshot of what tasks are being worked on, what's available to pick up, and where the project stands as a
whole, check out our [workboard](https://github.com/orgs/openmpf/projects/3).

