
from google.protobuf.json_format import ParseDict


from ft.api import TrainingArguments


def test_protobuf_message_defaults():

    input_dict = {
        "output_dir": "out-dir",
        "num_train_epochs": 0,
        "warmup_ratio": 0.03,
        "learning_rate": 0.0,
        "fp16": True,
        "disable_tqdm": False
    }

    ta: TrainingArguments = ParseDict(input_dict, TrainingArguments())

    # Assert that strings are set properly.
    assert ta.output_dir == "out-dir"

    # In protobuf, message fields have default
    # values, and the concept of None does not
    # exist. Therefore we need to find a way to
    # see "unset" fields versus "zero-set" fields.
    # Once this behavior changes, or once we find
    # a new way to handle this, we should address
    # it here. Perhaps a wrapper at the interception
    # layer of messages during deserialization?
    assert ta.num_train_epochs == TrainingArguments().num_train_epochs

    # Assert floats are set but only to within the
    # accuracy of a 32-bit float.
    assert abs(ta.warmup_ratio - 0.03) < 1e-6
    assert abs(ta.warmup_ratio - 0.03) > 0

    # Demonstrating the same behavior described above,
    # but for floats.
    assert ta.learning_rate == TrainingArguments().learning_rate

    # Assert booleans are being set.
    assert ta.fp16

    # Demonstrate same beahvior described above,
    # but for booleans.
    assert ta.disable_tqdm == TrainingArguments().disable_tqdm
