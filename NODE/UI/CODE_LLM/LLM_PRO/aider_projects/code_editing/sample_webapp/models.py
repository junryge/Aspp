"""데이터 모델 정의"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Todo:
    """할일 항목"""
    id: int
    title: str
    done: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    priority: int = 0  # 0=보통, 1=중요, 2=긴급

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "done": self.done,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "priority": self.priority,
        }

    def mark_done(self):
        self.done = True
        self.updated_at = datetime.now()

    def mark_undone(self):
        self.done = False
        self.updated_at = datetime.now()


@dataclass
class User:
    """사용자"""
    id: int
    username: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
