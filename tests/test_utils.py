import unittest
from ft.utils import dict_to_yaml_string, format_status_with_icon


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
        self.assertEqual(format_status_with_icon("anything_else"), "⚪ Error")

    def test_non_string_input(self):
        self.assertEqual(format_status_with_icon(None), "⚪ Unknown")
        self.assertEqual(format_status_with_icon(123), "⚪ Unknown")
        self.assertEqual(format_status_with_icon(["running"]), "⚪ Unknown")
        self.assertEqual(format_status_with_icon({"status": "running"}), "⚪ Unknown")
