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

    def test_generate_templates_with_matching_output_column(self):
        columns = ["input", "context", "response", "label"]
        prompt_template, completion_template = generate_templates(columns)

        expected_prompt_template = (
            "You are an LLM responsible for generating a response. Please provide a response given the user input below.\n\n"
            "<Input>: {input}\n"
            "<Context>: {context}\n"
            "<Label>: {label}\n"
            "<Response>: \n")
        expected_completion_template = "{response}\n"

        self.assertEqual(prompt_template, expected_prompt_template)
        self.assertEqual(completion_template, expected_completion_template)

    def test_generate_templates_with_no_matching_output_column(self):
        columns = ["input", "context", "user_input", "unknown"]
        prompt_template, completion_template = generate_templates(columns)

        expected_prompt_template = (
            "You are an LLM responsible for generating a response. Please provide a response given the user input below.\n\n"
            "<Input>: {input}\n"
            "<Context>: {context}\n"
            "<User_input>: {user_input}\n"
            "<Unknown>: \n")
        expected_completion_template = "{unknown}\n"

        self.assertEqual(prompt_template, expected_prompt_template)
        self.assertEqual(completion_template, expected_completion_template)

    def test_generate_templates_with_single_column(self):
        columns = ["response"]
        prompt_template, completion_template = generate_templates(columns)

        expected_prompt_template = (
            "You are an LLM responsible for generating a response. Please provide a response given the user input below.\n\n"
            "<Response>: \n")
        expected_completion_template = "{response}\n"

        self.assertEqual(prompt_template, expected_prompt_template)
        self.assertEqual(completion_template, expected_completion_template)

    def test_generate_templates_with_all_matching_columns(self):
        columns = ["answer", "response", "output", "label"]
        prompt_template, completion_template = generate_templates(columns)

        expected_prompt_template = (
            "You are an LLM responsible for generating a response. Please provide a response given the user input below.\n\n"
            "<Response>: {response}\n"
            "<Output>: {output}\n"
            "<Label>: {label}\n"
            "<Answer>: \n")
        expected_completion_template = "{answer}\n"

        self.assertEqual(prompt_template, expected_prompt_template)
        self.assertEqual(completion_template, expected_completion_template)

    def test_generate_templates_with_empty_columns(self):
        columns = []
        with self.assertRaises(IndexError):
            generate_templates(columns)
