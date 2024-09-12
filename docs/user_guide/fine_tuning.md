# Fine Tuning

[User Guide Home](../user_guide.md)

----


## Debugging Training Jobs

### Viewing Training Job Log Output

You can access the log output of a training job directly from the Cloudera ML Job that is spawned for the training job:
* From **Fine Tuning Studio**, go to the **Monitor Training Jobs** page.
* From the row of the training job in question, open the corresponding Cloudera ML Job by selecting the **Open CML Job** link.
* From the Job page, select **History** and select the most recent **Run**.
* This will open the **Session** logs where the entire log output of the training job is recorded.

-----

## Common Issues

### `MlflowException`

Occasionally starting a fine tuning job leads to an `MlflowException` error such as this one:

```
File /opt/cmladdons/python/site-packages/tracking_server/utils.py:34, in raise_mlflow_exception.<locals>.inner(*args, **kwargs)
     32     raise MlflowException(str(e), HTTP_STATUS_TO_ERRORCODE[e.status])
     33 except Exception as e:
---> 34     raise MlflowException(str(e))

MlflowException: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))
```

This is a known bug in Cloudera Machine Learning's MLFLow tracking server and is being worked on. Right now, this is a transient error, and can usually be fixed by just retrying the actual Cloudera ML Job without having to send an entire new job training request from the Studio UI.
* From **Fine Tuning Studio**, go to the **Monitor Training Jobs** page.
* From the failed training job, open the corresponding Cloudera ML Job by selecting the **Open CML Job** link.
* From the Job page, select the **Run as me** option to re-run the failed Job with the same exact arguments.
* Refresh this page to show the new run in the Job history.
* Monitor the log outputs of this new run to see if this transient MLFLow error has disappeared.

### `ValueError: Cannot find a suitable padding token to use, which is mandatory for TRL training.`

Currently, the Studio uses the `trl` package and corresponding `*Trainer` classes for training. There are two important notes about this trainer:
* The trainer uses a `DataCollatorForLanguageModeling` colator, which necessitates a padding token is available.
* This training scheme uses `trl`'s generation methods during training, which mask out the padding token from training.

Given the two notes above, if a model's tokenizer does not have a *dedicated* padding token that can be used that is separate from the EOS token, then **training will not produce valid results**. Particularly, in the case where a padding token is set equal to the EOS token, this means that EOS tokens are masked from the training job as well, which will eliminate the model learning when to stop an output (and which leads to the model rambling on without stopping).

The Studio has a simple heuristic to determine if there are any special reserved tokens available that can be used as the dedicated unique padding token. However, if the Studio cannot find a token suitable for the padding token, this error will also be presented. In the future, we would like to expose a custom padding token option for models that don't have a padding token. As of right now, **the padding token needs to be available in the tokenizer's vocabulary already**. Adding another token to the vocabulary would necessitate changing the base model's token embedding layer, which currently is not supported in the Studio.

Examples of base models where no padding token exists (or is set to EOS token), and henceforth cannot be used within the Studio:
* `distilbert/distilgpt2`
* `openai-community/gpt2`
* `Qwen/Qwen2-0.5B-Instruct`