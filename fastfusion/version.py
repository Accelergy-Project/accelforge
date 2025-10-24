from typing import Annotated

__version__ = "0.5"


def assert_version(version: str):
    if str(version) != str(__version__):
        raise ValueError(
            f"Version {str(version)} is not supported. "
            f"Only version {__version__} is supported."
        )
    return str(version)


version:  Annotated[str, assert_version] = __version__