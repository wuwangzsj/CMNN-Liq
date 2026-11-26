"""Read-only, support-component-aware path diagnostics.

Mw/PGA: sign=+1 (probability must not decrease).
qt1N: sign=-1 (probability must not increase).
No differencing or running maximum crosses a support gap.
"""
import numpy as np


def continuous_runs(mask):
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []


def component_metrics(components, sign=1):
    """Combine contiguous supported probability arrays for one case and seed."""
    assert sign in (1, -1)
    components = [np.asarray(p, dtype=float) for p in components if len(p) >= 2]
    assert all(p.ndim == 1 and np.isfinite(p).all() for p in components)
    ans = dict(eligible=bool(components), components=len(components),
               eligible_steps=sum(len(p) - 1 for p in components))
    for prefix, orient in [('', sign), ('raw_probability_', 1)]:
        if not components:
            ans.update({prefix + key: np.nan for key in
                        ['max_adverse_step', 'peak_drawdown', 'negative_step_sum']})
            continue
        negative_steps = [np.maximum(0., -np.diff(orient * p)) for p in components]
        peaks = [np.max(np.maximum.accumulate(orient * p) - orient * p) for p in components]
        ans[prefix + 'max_adverse_step'] = float(max(x.max() for x in negative_steps))
        ans[prefix + 'peak_drawdown'] = float(max(peaks))
        ans[prefix + 'negative_step_sum'] = float(sum(x.sum() for x in negative_steps))
    return ans


def case_metrics(probability, mask, sign=1):
    probability = np.asarray(probability, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    assert probability.ndim == mask.ndim == 1 and probability.shape == mask.shape
    return component_metrics([probability[run] for run in continuous_runs(mask)], sign)


def self_check():
    # A gap must not create a spurious 0.6 drawdown.
    r = case_metrics([.8, .7, .6, .2, .3], [1, 1, 0, 1, 1], 1)
    assert r['components'] == 2 and r['eligible_steps'] == 2
    assert np.isclose(r['peak_drawdown'], .1) and np.isclose(r['negative_step_sum'], .1)
    # Three individually small drops can accumulate; recovery does not cancel their sum.
    r = case_metrics([.8, .76, .72, .78, .70], [1] * 5, 1)
    assert np.isclose(r['max_adverse_step'], .08)
    assert np.isclose(r['peak_drawdown'], .1)
    assert np.isclose(r['negative_step_sum'], .16)
    r = case_metrics([.1, .2, .3], [1, 1, 1], -1)
    assert np.isclose(r['peak_drawdown'], .2) and r['raw_probability_peak_drawdown'] == 0
    r = case_metrics([.9, .1, .8], [1, 0, 1], 1)
    assert not r['eligible'] and np.isnan(r['peak_drawdown'])
    return True


if __name__ == '__main__':
    print('Segmented path metric self-check:', self_check())
