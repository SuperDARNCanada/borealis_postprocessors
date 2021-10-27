# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This files contains the exceptions generated when an impossible
conversion is attempted.
"""

import logging
postprocessing_logger = logging.getLogger('borealis_postprocessing')


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
