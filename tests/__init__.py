import os


_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_TEST_ROOT = _TEST_ROOT[:-18]
_PROJECT_ROOT = os.path.join(_TEST_ROOT, "src")  # root of project
_PATH_DATA = os.path.join(_TEST_ROOT, "data/processed")  # root of data