from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ParsedChunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> List[ParsedChunk]:
        """Parse a file and return a list of ParsedChunk objects."""
        pass

    def _base_metadata(self, file_path: str, file_type: str) -> Dict[str, Any]:
        from datetime import datetime, timezone
        import os
        return {
            "source_file": os.path.basename(file_path),
            "file_path": file_path,
            "file_type": file_type,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }