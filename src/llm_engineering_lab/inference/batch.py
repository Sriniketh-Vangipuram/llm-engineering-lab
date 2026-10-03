from dataclasses import dataclass
from enum import Enum
from typing import List

from .request import Request


class WorkType(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class BatchItem:
    request: Request
    work_type: WorkType
    tokens: int


@dataclass
class Batch:
    items: List[BatchItem]

    @property
    def total_tokens(self) -> int:
        return sum(item.tokens for item in self.items)

    @property
    def prefill_tokens(self) -> int:
        return sum(
            item.tokens
            for item in self.items
            if item.work_type == WorkType.PREFILL
        )

    @property
    def decode_tokens(self) -> int:
        return sum(
            item.tokens
            for item in self.items
            if item.work_type == WorkType.DECODE
        )

    def __repr__(self) -> str:
        return (
            f"Batch("
            f"items={len(self.items)}, "
            f"total_tokens={self.total_tokens}, "
            f"prefill={self.prefill_tokens}, "
            f"decode={self.decode_tokens}"
            f")"
        )