import re

def grab_from_yaml_file(
    yaml_file: str,
    startfrom: str | None = None,
    same_indent: bool = True,
    include_lines_before: int = 0,
) -> str:
    """
    Grab a section from a YAML file.

    Args:
        yaml_file:
            The path to the YAML file.
        startfrom:
            The string to start from. If None, return the entire file.
        same_indent:
            Whether to include lines with the same indentation as the startfrom line, or
            only lines with >= indentation.
        include_lines_before:
            The number of lines to include before the startfrom line.

    Returns:
        The section of the YAML file as a string.
    """
    with open(yaml_file, "r") as f:
        contents = f.readlines()
    start, end = 0, len(contents)
    n_whitespace = 0

    if startfrom is not None:
        for i, line in enumerate(contents):
            if re.findall(r"\b\s*" + startfrom + r"\b", line):
                start = i
                n_whitespace = len(re.findall(r"^\s*", line)[0])
                break
        else:
            raise ValueError(f"{startfrom} not found in {yaml_file}")
        for i, line in enumerate(contents[start + 1 :]):
            if not line.strip():
                continue
            ws = len(re.findall(r"^\s*", line)[0])
            if ws < n_whitespace or (not same_indent and ws == n_whitespace):
                end = start + i + 1
                break

    contents = [c[n_whitespace:] for c in contents[start - include_lines_before : end]]
    return "".join(contents)
