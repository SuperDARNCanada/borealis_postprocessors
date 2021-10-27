# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This files contains the exceptions generated when an impossible
conversion is attempted.
"""

import logging
postprocessing_logger = logging.getLogger('borealis_postprocessing')


class ImproperFileTypeError(Exception):
    """
    Raised when the file type is not a valid SuperDARN data file type.

    Parameters
    ----------
    file_type: str
        file type which is not supported.
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
