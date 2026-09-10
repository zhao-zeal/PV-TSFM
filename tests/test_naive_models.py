import numpy as np

from pv_tsfm.models import DailySeasonalPersistence, LastValue, SevenDaySameHourMean


def test_e0_formulas():
    past = np.arange(336, dtype=float)
    mask = np.ones(336, dtype=bool)
    np.testing.assert_array_equal(LastValue().predict(past, mask, 24), np.repeat(335.0, 24))
    np.testing.assert_array_equal(DailySeasonalPersistence().predict(past, mask, 24), past[-24:])
    expected = past[-168:].reshape(7, 24).mean(axis=0)
    np.testing.assert_array_equal(SevenDaySameHourMean().predict(past, mask, 24), expected)
