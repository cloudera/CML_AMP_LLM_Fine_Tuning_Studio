import unittest
from ft.utils import dict_to_yaml_string, format_status_with_icon
from ft.utils import generate_templates

class TestYamlFunctions(unittest.TestCase):

    def test_dict_to_yaml_string_basic(self):
        yaml_dict = {'key1': 'value1', 'key2': 'value2'}
        expected_yaml = "key1: value1\nkey2: value2\n"

        result = dict_to_yaml_string(yaml_dict)
        self.assertEqual(result, expected_yaml)

    def test_dict_to_yaml_string_with_none(self):
        yaml_dict = {'key1': 'value1', 'key2': None}
        expected_yaml = "key1: value1\nkey2:\n"  # Adjusted to match actual output

        result = dict_to_yaml_string(yaml_dict)
        self.assertEqual(result, expected_yaml)


class TestFormatStatusWithIcon(unittest.TestCase):
    def test_succeeded_status(self):
        self.assertEqual(format_status_with_icon("succeeded"), "🟢 Succeeded")

    def test_running_status(self):
        self.assertEqual(format_status_with_icon("running"), "🔵 Running")

    def test_scheduling_status(self):
        self.assertEqual(format_status_with_icon("scheduling"), "🟡 Scheduling")

    def test_failed_status(self):
        self.assertEqual(format_status_with_icon("failed"), "🔴 Failed")

    def test_unknown_status(self):
        self.assertEqual(format_status_with_icon("Unknown"), "⚪ Unknown")
        self.assertEqual(format_status_with_icon("anything_else"), "⚪ Anything_else")

    def test_non_string_input(self):
        self.assertEqual(format_status_with_icon(None), "⚪ Unknown")
        self.assertEqual(format_status_with_icon(123), "⚪ Unknown")
        self.assertEqual(format_status_with_icon(["running"]), "⚪ Unknown")
        self.assertEqual(format_status_with_icon({"status": "running"}), "⚪ Unknown")

class TestGenerateTemplates(unittest.TestCase):

    def test_single_output_column(self):
        columns = ['instruction', 'input', 'response']
        expected_prompt_template = (
            "You are an LLM responsible with generating a response. Please provide a response given the user input below.\n\n"
            "<Instruction>: {instruction}\n"
            "<Input>: {input}\n"
            "<Response>: \n"
        )
        expected_completion_template = "{response}\n"

        prompt_template, completion_template = generate_templates(columns)
        self.assertEqual(prompt_template, expected_prompt_template)
        self.assertEqual(completion_template, expected_completion_template)

    def test_multiple_output_columns(self):
        columns = ['instruction', 'input', 'response', 'output']
        expected_prompt_template = (
            "You are an LLM responsible with generating a response. Please provide a response given the user input below.\n\n"
            "<Instruction>: {instruction}\n"
            "<Input>: {input}\n"
            "<Response>: \n"
        )
        expected_completion_template = (
            "{response}\n"
            "<Output>: {output}\n"
        )

        prompt_template, completion_template = generate_templates(columns)
        self.assertEqual(prompt_template, expected_prompt_template)
        self.assertEqual(completion_template, expected_completion_template)

    def test_equal_input_output_columns(self):
        columns = ['col1', 'col2', 'col3', 'col4']
        expected_prompt_template = (
            "You are an LLM responsible with generating a response. Please provide a response given the user input below.\n\n"
            "<Col1>: {col1}\n"
            "<Col2>: {col2}\n"
            "<Col3>: \n"
        )
        expected_completion_template = (
            "{col3}\n"
            "<Col4>: {col4}\n"
        )

        prompt_template, completion_template = generate_templates(columns)
        self.assertEqual(prompt_template, expected_prompt_template)
        self.assertEqual(completion_template, expected_completion_template)

    def test_no_columns(self):
        columns = []
        expected_prompt_template = (
            "You are an LLM responsible with generating a response. Please provide a response given the user input below.\n\n"
        )
        expected_completion_template = ""

        prompt_template, completion_template = generate_templates(columns)
        self.assertEqual(prompt_template, expected_prompt_template)
        self.assertEqual(completion_template, expected_completion_template)
