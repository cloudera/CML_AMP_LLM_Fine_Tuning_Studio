# CML Fine Tuning Studio

## Resources

* [Technical Overview](docs/techinical_overview.md)
* [User Guide](docs/user_guide/index.md)

## About this AMP

The CML Fine Tuning Studio is a Cloudera-developed AMP that provides users with an all-encompassing application and “ecosystem” for managing, fine tuning, and evaluating LLMs. This application is a launcher that helps users organize and dispatch other CML Workloads (primarily CML Jobs) that are configured specifically for LLM training and evaluation type tasks.

![Fine Tuning Studio Homepage](resources/images/fts_home.png)

## High Level Features

* Import datasets, models, model PEFT adapters, and other components into the Application's UI for easy tracking of project resources
* Design training prompt and inference prompt templates to be used to augment datasets for training, inference, and evaluation purposes
* Launch fine tuning jobs and MLFlow evaluation jobs directly from the UI, which in turn kicks off appropriate CML Workloads
* Locally compare generation results between different PEFT adapters for a given base model