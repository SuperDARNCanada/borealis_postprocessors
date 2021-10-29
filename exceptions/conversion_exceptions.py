# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This files contains the exceptions generated when an impossible
conversion is attempted.
"""

import logging
postprocessing_logger = logging.getLogger('borealis_postprocessing')


class GeneralConversionError(Exception):
    """
    Raised when an unexpected error occurs.

    Parameters
    ----------
    error_str: str
        explanation for why the error was raised.

    Attributes
    ----------
    message: str
        The message to display with the error

    See Also
    --------
    conversion.py
    """

    def __init__(self, error_str: str):
        self.message = "Unexpected error converting files: {error_str}"\
            "".format(error_str=error_str)
        postprocessing_logger.error(self.message)
        Exception.__init__(self, self.message)


class FileDeletionError(Exception):
    """
    Raised when a file cannot be deleted.

    Parameters
    ----------
    error_str: str
        explanation for why the error was raised.

    Attributes
    ----------
    message: str
        The message to display with the error

    See Also
    --------
    conversion.py
    """

    def __init__(self, error_str: str):
        self.message = "Error deleting file: {error_str}"\
            "".format(error_str=error_str)
        postprocessing_logger.error(self.message)
        Exception.__init__(self, self.message)


class FileDoesNotExistError(Exception):
    """
    Raised when the file does not exist.

    Parameters
    ----------
    error_str: str
        explanation for why the error was raised.

    Attributes
    ----------
    message: str
        The message to display with the error

    See Also
    --------
    conversion.py
    """

    def __init__(self, error_str: str):
        self.message = "File does not exist: {error_str}"\
            "".format(error_str=error_str)
        postprocessing_logger.error(self.message)
        Exception.__init__(self, self.message)


class NoConversionNecessaryError(Exception):
    """
    Raised when the file types and structures specified
    are the same for input and output files.

    Parameters
    ----------
    error_str: str
        explanation for why the error was raised.

    Attributes
    ----------
    message: str
        The message to display with the error

    See Also
    --------
    conversion.py
    """

    def __init__(self, error_str: str):
        self.message = "File type and structure are identical: "\
            "{error_str}".format(error_str=error_str)
        postprocessing_logger.error(self.message)
        Exception.__init__(self, self.message)


class ImproperFileStructureError(Exception):
    """
    Raised when the file structure is not a valid structure
    for the SuperDARN data file type.

    Parameters
    ----------
    error_str: str
        explanation for why the error was raised.

    Attributes
    ----------
    message: str
        The message to display with the error

    See Also
    --------
    conversion.py
    """

    def __init__(self, error_str: str):
        self.message = "File structure is not a valid "\
            "SuperDARN data file structure: {error_str}"\
            "".format(error_str=error_str)
        postprocessing_logger.error(self.message)
        Exception.__init__(self, self.message)


class ImproperFileTypeError(Exception):
    """
    Raised when the file type is not a valid SuperDARN data file type.

    Parameters
    ----------
    error_str: str
        explanation for error was raised.

    Attributes
    ----------
    message: str
        The message to display with the error

    See Also
    --------
    conversion.py
    """

    def __init__(self, error_str: str):
        self.message = "File type is not a valid "\
            "SuperDARN data file type: {error_str}"\
            "".format(error_str=error_str)
        postprocessing_logger.error(self.message)
        Exception.__init__(self, self.message)


class ConversionUpstreamError(Exception):
    """
    Raised when the file cannot be converted because the desired
    conversion is upstream.

    Parameters
    ----------
    error_str: str
        explanation for why the file cannot be converted.

    Attributes
    ----------
    message: str
        The message to display with the error

    See Also
    --------
    conversion.py
    """

    def __init__(self, error_str: str):
        self.message = "The file cannot be converted due to the "\
            " following error: {error_str}"\
            "".format(error_str=error_str)
        postprocessing_logger.error(self.message)
        Exception.__init__(self, self.message)
