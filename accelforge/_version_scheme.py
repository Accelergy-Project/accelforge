"""Version scheme for setuptools-scm - creates post-release versions."""

from setuptools_scm.version import guess_next_version


def post_version(version):
    """Create post-release versions instead of dev versions."""
    if version.exact:
        return version.format_with("{tag}").lstrip("v")

    base = (
        str(version.tag).lstrip("v")
        if version.tag
        else (guess_next_version(version) or "1.0")
    )
    distance = version.distance or 0

    return f"{base}.{distance}" if distance > 0 else base


def no_local(version):
    """No local version identifier."""
    return ""
