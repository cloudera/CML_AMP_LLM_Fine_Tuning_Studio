# Development

## Testing AMP Locally

Most UI features can be tested locally. To start the Streamlit UI server from your
local machine, just run:

```
streamlit run main.py
```


## Testing AMP in CML Workspace

The team is currently using the [CML Team GenAI GPU Cluster](https://docs.google.com/document/d/1wsxWV3P6dtBacxOZgE5YVPQaps6ZSkL6HsaaI5JXMF0/) for testing. If you're ready to test your AMP in a CML workspace environment:
* Push changes to a branch (optional, if you have local changes)
* Log in to the GenAI cluster
* Open a Workspace, and import the AMP by using the `github.infra.com` URL (optionally specifying the branch with your feature updates)


## Testing

This package uses a combination of `pytest` and `unittest` features to test the main `ft` package. If you're writing application intensive workloads, consider writing some tests as a courtesy to those who pick up work after you.

Running tests and seeing coverage reports locally:

```
pytest --cov=ft --cov-report=html tests/
python -m http.server --directory htmlcov/
```