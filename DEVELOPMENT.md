# Development

This covers some development prerequisites for contributers to the Fine Tuning Studio. This document is for users who want to actively contribute changes to the Fine Tuning Studio's source code.

Developing on the Fine Tuning Studio requires some prerequisite knowledge, depending on the area of work of the development. If you're unfamiliar with any of these topics completely, it's encouraged that developers build a general sense of what all of these technologies are.
* Python
* [SQLite](https://www.sqlite.org/cli.html)
* [Streamlit](https://streamlit.io/)
* [gRPC](https://grpc.io/docs/what-is-grpc/introduction/)
* [Protobuf](https://protobuf.dev/)
* [Huggingface Transformers](https://huggingface.co/docs/transformers/en/index)
* [Huggingface TRL](https://huggingface.co/docs/trl/en/index)
* [Huggingface Accelerate](https://huggingface.co/docs/accelerate/en/index)
* [MLFlow](https://mlflow.org/docs/latest/index.html)
* [pytest](https://docs.pytest.org/en/stable/)

Some other prerequisites for working with the Fine Tuning Studio
* Download and configure [Visual Studio Code](https://code.visualstudio.com/)
* Read through the [Technical Overview](docs/techinical_overview.md) of the Fine Tuning Studio
* [Configure VS Code as a Local IDE for Cloudera ML](https://docs.cloudera.com/cdsw/1.10.5/editors/topics/ml-editors-vs-code.html)

## Development Workflow

For development, it's typical to deploy the AMP to a Cloudera ML project, and do all work remotely.

Do once and never do again for a Project:
* Set up all project prerequisites
* Import the AMP into a Cloudera ML project, and wait for the AMP to complete loading
* Run the `./bin/get-dev-tools.sh` from within a session to enhance remote developer experience

Do once per development session:
* Start up an ssh session using `cdswctl`. An example looks like `cdswctl ssh-endpoint -p "<user>/<project>" -c 2 -m 24 -g 0 -r 108`
* Connect to the remote session following the [Setting up VS Code](https://docs.cloudera.com/cdsw/1.10.5/editors/topics/ml-setting-up-vs-code.html) guide
* Pull down any recent changes into your development branch if necessary, i.e., `git pull origin dev`

Do every time you'd like to test out your changes:
* Restart the Application from the Cloudera ML **Applications** page.


## Testing

This package uses a combination of `pytest` and `unittest` features to test the main `ft` package. If you're writing application intensive workloads, consider writing some tests as a courtesy to those who pick up work after you.

Running tests and seeing coverage reports locally:

```
./bin/run-tests
```

## Making Pull Requests

The typical procedure for making a new pull request is as follows:
* Make your code changes.
* Restart the Fine Tuning Studio Application and test your changes.
* **Write unit tests for any applicable piece of your code changes.**
* Run `./bin/run-tests.sh` to run the tests and make sure your tests are passing.
* **Run `./bin/format.sh` to format your code following PEP formatting guidelines.**
* Create a new branch.
* Add the relevant changed files to the staging area.
* Commit your changes to the new branch.
* Push a new upstream branch to the origin.
* Open a pull request from `github.com` for your new branch into the `dev` branch (which is the active development branch).
* Wait for approvals, incorporate any feedback if necessary.
* Merge the change into dev once you have approvals.
* Checkout `dev` branch again and `git pull` your changes which are now in `dev`.