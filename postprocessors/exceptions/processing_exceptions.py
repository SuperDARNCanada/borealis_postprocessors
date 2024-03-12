# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This files contains the exceptions generated when an impossible
conversion is attempted.
"""

import logging
postprocessing_logger = logging.getLogger('borealis_postprocessing')


class FileCreationError(Exception):
    """
    Raised when a file cannot be created.

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
    antennas_iq_to_bfiq.py
    bfiq_to_rawacf.py
    """

    def __init__(self, error_str: str):
        self.message = "File cannot be made: {error_str}"\
            "".format(error_str=error_str)
        postprocessing_logger.error(self.message)
        Exception.__init__(self, self.message)
