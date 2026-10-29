# vitstrain
# Filename: tests/test_args.py
# Description: Tests for vitstrain cli.

import unittest

from args import parse_args


class TestBooleanArguments(unittest.TestCase):
    def test_add_rotations_default(self):
        args = parse_args([])
        self.assertIs(args.add_rotations, True)

    def test_add_rotations_boolean_values(self):
        test_cases = [
            # value, expected
            ("True", True),
            ("False", False),
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
        ]
        for value, expected in test_cases:
            args = parse_args(["--add-rotations", value])
            self.assertIs(
                args.add_rotations,
                expected,
                msg=f"exp {expected}, got {args.add_rotations} from {value!r}",
            )

    def test_remove_long_tail_default(self):
        args = parse_args([])
        self.assertIs(args.remove_long_tail, False)

    def test_remove_long_tail_boolean_values(self):
        test_cases = [
            # value, expected
            ("True", True),
            ("False", False),
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
        ]
        for value, expected in test_cases:
            args = parse_args(["--remove-long-tail", value])
            self.assertIs(
                args.remove_long_tail,
                expected,
                msg=f"exp {expected}, got {args.remove_long_tail} from {value!r}",
            )


if __name__ == "__main__":
    unittest.main()
