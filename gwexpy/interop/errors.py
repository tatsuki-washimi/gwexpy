class IoNotImplementedError(NotImplementedError):
    """Exception raised when an I/O format is not fully implemented."""

    pass


def raise_unimplemented_io(format_name, hint=None, refs=None, plugin_help=None):
    """
    Raise a standardized NotImplementedError for missing I/O readers.

    Parameters
    ----------
    format_name : str
        Name of the format (e.g. 'WIN', 'SDB').
    hint : str, optional
        Suggestion for alternative approaches.
    refs : str, optional
        References to external tools or scripts.
    plugin_help : str, optional
        Instructions on how to add a plugin.
    """
    msg = [
        f"The reader for format '{format_name}' is currently unimplemented in gwexpy."
    ]

    if hint:
        msg.append(f"Hint: {hint}")
    if refs:
        msg.append(f"Reference: {refs}")
    if plugin_help:
        msg.append(f"To implement this, please check {plugin_help}")
    else:
        msg.append("Contributions to implement this format are welcome.")

    raise IoNotImplementedError("\n".join(msg))
