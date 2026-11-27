"""Two exact speedups of the gluonts evaluation of GIFT-Eval (#425).

cProfile of the eval of loop_seattle/5T/short (6,460 windows, 96 s) found the
time in gluonts, not in the model (1.8 s):

* The seasonal error. The MASE divides by the seasonal error of each window:
  the mean absolute difference of the history of the window and the same
  history shifted by the seasonality. ``gluonts.ev.ts_stats.seasonal_error``
  takes the two shifted copies with ``np.take`` and ``range`` indices. On a
  history of 105,000 values this takes 21 ms a window, 82 s in all. Slices
  take the same values in 0.4 ms, and the arithmetic after the two copies
  does not change.
* The quantile names. The metrics ask each forecast for its quantiles about
  27 times, and each request parses the quantile again into a pydantic
  dataclass (``Quantile.parse``): 5 s in all. A parse is a function of its
  argument only, so each float or str is parsed once and the result is kept.

Thus each value of the eval is the same float, in a fraction of the time.
"""

import gluonts.model.evaluation
import numpy as np
from gluonts.model.forecast import Quantile

# The parsers of gluonts, before use_fast_gluonts replaces them.
_GLUONTS_PARSE = Quantile.parse
_GLUONTS_FROM_STR = Quantile.from_str
_PARSED = {}


def seasonal_error(time_series, seasonality, time_axis=0):
    """``gluonts.ev.ts_stats.seasonal_error`` with slices: the mean absolute
    difference of ``time_series`` and ``time_series`` shifted by
    ``seasonality`` along ``time_axis``. A seasonality longer than the series
    becomes 1, as in gluonts."""
    time_length = time_series.shape[time_axis]
    if seasonality > time_length:
        seasonality = 1
    y_t = _time_slice(time_series, seasonality, time_length, time_axis)
    y_tm = _time_slice(time_series, 0, time_length - seasonality, time_axis)
    return np.abs(y_t - y_tm).mean(axis=time_axis, keepdims=True)


def _time_slice(array, start, stop, axis):
    """``array[start:stop]`` along ``axis``."""
    index = [slice(None)] * array.ndim
    index[axis] = slice(start, stop)
    return array[tuple(index)]


def _parsed_once(parser, quantile):
    """``parser(quantile)``, parsed once for each float or str. The key holds
    the type and the repr of the argument: 0.5 and "0.5" give different
    names, and so do 0.0 and -0.0."""
    if type(quantile) not in (float, str):
        return parser(quantile)
    key = (parser, type(quantile), repr(quantile))
    if key not in _PARSED:
        _PARSED[key] = parser(quantile)
    return _PARSED[key]


def parse_quantile(cls, quantile):
    """``Quantile.parse`` of gluonts, parsed once for each float or str."""
    return _parsed_once(_GLUONTS_PARSE, quantile)


def quantile_from_str(cls, quantile):
    """``Quantile.from_str`` of gluonts, parsed once for each str."""
    return _parsed_once(_GLUONTS_FROM_STR, quantile)


def use_fast_gluonts():
    """Makes ``gluonts.model.evaluate_model`` compute the seasonal errors with
    :func:`seasonal_error` and parse each quantile once."""
    gluonts.model.evaluation.seasonal_error = seasonal_error
    Quantile.parse = classmethod(parse_quantile)
    Quantile.from_str = classmethod(quantile_from_str)
