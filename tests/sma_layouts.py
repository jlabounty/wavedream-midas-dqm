"""The channel cabling of the SMA frames the tests feed.

The real runs in ``tests/data`` (342, 682, 1008) and the synthetic frames
modelled on them are all from before run 1015: S1..S5 on ch 1-5, RF 6, the
proton current on 7, delayed channels 8-10, and S1..S5 plus the RF raising the
mismatch flags, and no NIM copies. The plugin's defaults are the run-1015
cabling (TOT_1015/NIM_1015 below), so the tests on these frames name theirs;
the defaults have tests of their own.

Plain module (numpy and mdqm only), so the fixture generators can import it.
"""

from __future__ import annotations

import json

import numpy as np

from mdqm.plugins import sma_words as W

#: The ``/DQM/SMA`` settings for the pre-1015 cabling.
OLD_LAYOUT = {
    "Channel roles": {"s1": 1, "counters": [1, 2, 3, 4, 5], "rf": 6, "current": 7,
                      "delayed": [8, 9, 10]},
    "Self check": {"mismatch flag channels": [1, 2, 3, 4, 5, 6]},
    #: No NIM copies before run 1015.
    "NIM": {"channels": [-1, -1, -1, -1, -1]},
}
OLD_ROLES = W.Roles(s1=1, counters=(1, 2, 3, 4, 5), rf=6, current=7, delayed=(8, 9, 10))


def old_layout(settings=None) -> dict:
    """`settings` over OLD_LAYOUT (a given key wins, section by section)."""
    out = json.loads(json.dumps(OLD_LAYOUT))
    for k, v in (settings or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


# --- the run-1015 cabling (the plugin's defaults) ---------------------------------

#: Counter k's TOT channel and its NIM copy ("S*k*L") since run 1015, S1..S5.
TOT_1015 = (1, 2, 7, 4, 5)
NIM_1015 = (3, 9, 10, 11, 12)
RF_1015 = 6


def words_of(t, ch, tot, shift=14, fine_lag=None):
    """Trigger words for true times ``t`` (any order; sorted here, stably).

    ``fine_lag``: ``{channel: ns}`` added to those words' fine field only (mod
    2^20), the NIM fine-time lag fault: the time moves by the lag and the
    coarse field no longer agrees with the fine one.
    """
    t = np.asarray(t, dtype=np.int64)
    ch = np.asarray(ch)
    tot = np.broadcast_to(np.asarray(tot), t.shape)
    o = np.argsort(t, kind="stable")
    t, ch, tot = t[o], ch[o], tot[o]
    coarse, fine = W.fields_of(t, shift)
    for c, lag in (fine_lag or {}).items():
        m = ch == c
        fine[m] = ((fine[m].astype(np.int64) + int(lag)) & (W.FINE_WRAP_NS - 1)).astype(np.uint32)
    return W.encode(ch, tot, coarse, fine)


def dense_1015(n_events=3000, span_ns=30_000_000, t0=10**12, seed=0, eff_tot=0.97,
               eff_nim=0.97, nim_dt=2, rf_words=10000, echo_frac=0.01, shift=14,
               fine_lag=None):
    """A dense 1015-cabled frame (about 40000 words at the defaults), as words.

    ``n_events`` particles at random times over ``span_ns``: each counter's TOT
    word (S_k at t + k ns, ToT 20-60) with probability ``eff_tot``, its NIM
    copy ``nim_dt`` +- 1 ns later (ToT 10) with ``eff_nim``; a share
    ``echo_frac`` of S3's TOT words is followed by a late echo word (ToT 200);
    ``rf_words`` RF words at random times. ``fine_lag``: see :func:`words_of`.
    """
    rng = np.random.default_rng(seed)
    te = t0 + np.sort(rng.integers(0, span_ns, n_events))
    ts, chs, tots = [], [], []
    for k, (ct, cn) in enumerate(zip(TOT_1015, NIM_1015, strict=True)):
        tk = te + k
        m = rng.random(n_events) < eff_tot
        ts.append(tk[m])
        chs.append(np.full(int(m.sum()), ct))
        tots.append(rng.integers(20, 60, int(m.sum())))
        m = rng.random(n_events) < eff_nim
        ts.append(tk[m] + nim_dt + rng.integers(-1, 2, int(m.sum())))
        chs.append(np.full(int(m.sum()), cn))
        tots.append(np.full(int(m.sum()), 10))
        if ct == TOT_1015[2] and echo_frac > 0:
            m = rng.random(n_events) < echo_frac
            ts.append(tk[m] + 400)
            chs.append(np.full(int(m.sum()), ct))
            tots.append(np.full(int(m.sum()), 200))
    ts.append(t0 + rng.integers(0, span_ns, rf_words))
    chs.append(np.full(rf_words, RF_1015))
    tots.append(np.full(rf_words, 5))
    return words_of(np.concatenate(ts), np.concatenate(chs), np.concatenate(tots), shift,
                    fine_lag)
