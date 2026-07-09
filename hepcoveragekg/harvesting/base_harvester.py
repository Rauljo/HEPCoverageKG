# =============================================================================
# HEPCoverageKG: BaseHarvester interface
# Adapted from DeepCollector's base_harvester.py. There, execute_harvest(state)
# mutated one shared CatalogState across a live, incrementally-discovered
# catalog. Here there's no such shared state at harvest time - each fixed
# paper URL is fetched and parsed independently; the retrieval index over the
# parsed text is only built later, per paper, in extraction/state.py.
# =============================================================================
from abc import ABC, abstractmethod
from typing import Any


class BaseHarvester(ABC):
    """Abstract interface for fetching and parsing a single fixed source page."""

    def __init__(self, config: Any):
        self.config = config
        self.verbosity = getattr(config, "VERBOSITY_LEVEL", 1)

    @abstractmethod
    def fetch(self, url: str) -> str:
        """Fetches raw HTML for a single fixed page. Returns the raw HTML text."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, html: str, source_url: str) -> Any:
        """Parses raw HTML into a structured paper document."""
        raise NotImplementedError
