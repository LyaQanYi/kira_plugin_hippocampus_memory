"""Per-session sender info cache.

KiraAI's SessionManager stores chunks as `{role, content}` only, dropping any
sender metadata. The hippocampus extractor needs `(user_id, nickname)` per
turn to route facts to the correct entity. This cache fills the gap by
listening on `@on.im_batch_message` and remembering the most recent N user
messages per session.
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Dict, Optional


class SenderCache:
    def __init__(self, per_session: int = 32, max_sessions: int = 2000) -> None:
        self._per_session = per_session
        # Bound the number of tracked sessions so a long-running process with
        # many distinct chats can't grow _data / _consumed_seq without limit.
        # Oldest-inserted session is evicted FIFO when the cap is exceeded.
        self._max_sessions = max_sessions
        self._data: Dict[str, "deque[dict]"] = {}
        # Monotonic record counter (NOT wall-clock): coarse OS timers (~15ms on
        # Windows) let two rapid messages share a ts, which would make a
        # ts-based watermark silently drop the second one. A strictly-increasing
        # seq is collision-free.
        self._seq = 0
        # Per-session consumption watermark (max seq already paired with an
        # assistant reply). Lets take_unconsumed() hand the hippocampus EXACT
        # sender identity per turn instead of guessing by text match.
        self._consumed_seq: Dict[str, int] = {}
        self._lock = Lock()

    def record(
        self,
        sid: str,
        user_id: str,
        nickname: str,
        text: str,
    ) -> None:
        if not sid:
            return
        with self._lock:
            dq = self._data.get(sid)
            if dq is None:
                # Evict the oldest-inserted session (dicts preserve insertion
                # order) before adding a new one past the cap. Keep _consumed_seq
                # in lockstep so it never outlives its _data entry.
                while len(self._data) >= self._max_sessions:
                    oldest = next(iter(self._data))
                    self._data.pop(oldest, None)
                    self._consumed_seq.pop(oldest, None)
                dq = deque(maxlen=self._per_session)
                self._data[sid] = dq
            self._seq += 1
            dq.append(
                {
                    "ts": time.time(),
                    "seq": self._seq,
                    "user_id": user_id or "",
                    "nickname": nickname or "",
                    "text": text or "",
                }
            )

    def get_recent(self, sid: str, *, max_age_sec: float = 600.0) -> list[dict]:
        """Return recent (ts, user_id, nickname, text) entries for a session."""
        if not sid:
            return []
        cutoff = time.time() - max_age_sec
        with self._lock:
            dq = self._data.get(sid)
            if not dq:
                return []
            return [dict(item) for item in dq if item["ts"] >= cutoff]

    def take_unconsumed(self, sid: str, *, max_age_sec: float = 600.0) -> list[dict]:
        """Return user records recorded since the last take, advancing the
        watermark so each turn is paired exactly once.

        Used to pair a completed assistant reply with the precise user message(s)
        of this turn — every record carries its own ``user_id``/``nickname``, so
        the hippocampus routes facts to the right entity even in busy
        multi-speaker group bursts (no text-matching guesswork). Unlike
        ``pop_pending_users`` this does NOT delete records, so ``get_recent`` and
        the sender map still see the full window.
        """
        if not sid:
            return []
        cutoff = time.time() - max_age_sec
        with self._lock:
            dq = self._data.get(sid)
            if not dq:
                return []
            last = self._consumed_seq.get(sid, 0)
            fresh = [
                dict(item)
                for item in dq
                if item["seq"] > last and item["ts"] >= cutoff
            ]
            if fresh:
                self._consumed_seq[sid] = max(item["seq"] for item in fresh)
            return fresh

    def get_sender_map(self, sid: str) -> Dict[str, str]:
        """Return {user_id -> latest nickname} for the session."""
        sender_map: Dict[str, str] = {}
        for item in self.get_recent(sid, max_age_sec=24 * 3600):
            uid = item["user_id"]
            if uid:
                sender_map[uid] = item["nickname"] or uid
        return sender_map

    def pop_pending_users(
        self, sid: str, *, max_age_sec: float = 600.0
    ) -> list[dict]:
        """Pop and return recent user messages (used when pairing with an
        assistant turn to form a chunk for the hippocampus).
        """
        if not sid:
            return []
        cutoff = time.time() - max_age_sec
        with self._lock:
            dq = self._data.get(sid)
            if not dq:
                return []
            kept = deque(maxlen=self._per_session)
            popped = []
            for item in dq:
                if item["ts"] >= cutoff:
                    popped.append(dict(item))
                else:
                    kept.append(item)
            self._data[sid] = kept
            return popped

    def clear(self, sid: Optional[str] = None) -> None:
        with self._lock:
            if sid is None:
                self._data.clear()
                self._consumed_seq.clear()
            else:
                self._data.pop(sid, None)
                self._consumed_seq.pop(sid, None)
