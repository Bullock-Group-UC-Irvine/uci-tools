import os
import pathlib


def pytest_configure(config):
    os.environ.setdefault(
        'UCI_TOOLS_CONFIG',
        str(
            pathlib.Path(__file__).parent
            / '.github'
            / 'config_ci.ini'
        ),
    )
