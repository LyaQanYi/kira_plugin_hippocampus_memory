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
    def __init__(self, per_session: int = 32) -> None:
        self._per_session = per_session
        self._data: Dict[str, "deque[dict]"] = {}
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
                dq = deque(maxlen=self._per_session)
                self._data[sid] = dq
            dq.append(
                {
                    "ts": time.time(),
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
            else:
                self._data.pop(sid, None)
