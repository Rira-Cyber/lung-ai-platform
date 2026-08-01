from abc import ABC, abstractmethod
from typing import Any


class BaseLogger(ABC):
    @abstractmethod
    def log_epoch(
        self,
        epoch: int,
        epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        pass

    def close(self) -> None:
        """
        Cleanup resources.
        """
        pass
