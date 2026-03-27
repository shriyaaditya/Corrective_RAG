"""
retriever/document_watcher.py

Watches a folder for new or modified documents and automatically
ingests them into the VectorStore.

Uses the `watchdog` library for cross-platform filesystem events.
Runs the watcher in a background daemon thread so the main CLI
stays responsive.

Usage
-----
    store   = VectorStore()
    parser  = DocumentParser()
    watcher = DocumentWatcher(docs_dir="docs", store=store, parser=parser)
    watcher.start()          # non-blocking — runs in background thread
    ...
    watcher.stop()
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from retriever.document_parser import DocumentParser, SUPPORTED_EXTENSIONS
from retriever.vector_store import VectorStore


# ── Event handler ─────────────────────────────────────────────────────────────

class _DocEventHandler(FileSystemEventHandler):
    """
    Called by watchdog whenever files are added into the watched directory.
    Ingests new supported files into the vector store.
    """

    def __init__(self, store: VectorStore, parser: DocumentParser) -> None:
        super().__init__()
        self.store  = store
        self.parser = parser
        # Debounce: avoid processing the same file twice in quick succession
        self._recently_processed: dict[str, float] = {}
        self._debounce_seconds = 2.0

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.dest_path)

    def _handle(self, path: str) -> None:
        path = os.path.abspath(path)
        ext  = Path(path).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            return

        # Debounce
        now = time.time()
        last = self._recently_processed.get(path, 0)
        if now - last < self._debounce_seconds:
            return
        self._recently_processed[path] = now

        # Small delay so the file is fully written before we read it
        time.sleep(0.5)

        filename = os.path.basename(path)
        fingerprint = _file_fingerprint(path)
        if not self.store.should_ingest(filename, fingerprint):
            return

        print(f"\n[Watcher] Detected: '{filename}' — ingesting …")
        self._ingest(path, fingerprint)

    def _ingest(self, path: str, fingerprint: str | None = None) -> None:
        filename = os.path.basename(path)
        try:
            chunks = self.parser.parse(path)
            if chunks:
                self.store.add_chunks(
                    chunks,
                    source=filename,
                    source_fingerprint=fingerprint,
                    suppress_unchanged_log=True,
                )
            else:
                print(f"[Watcher] No text extracted from '{filename}' — skipping.")
        except Exception as e:
            print(f"[Watcher] Error ingesting '{filename}': {e}")


# ── DocumentWatcher ───────────────────────────────────────────────────────────

class DocumentWatcher:
    """
    Watches *docs_dir* recursively and ingests every new/modified
    supported file into *store*.

    Also performs an initial scan on startup so any documents already
    present in the folder are indexed before queries begin.

    Parameters
    ----------
    docs_dir : str
        Path to the folder to watch (created if it doesn't exist).
    store : VectorStore
        The vector store to ingest into.
    parser : DocumentParser
        The parser used to extract text.
    """

    def __init__(
        self,
        docs_dir: str,
        store: VectorStore,
        parser: DocumentParser,
    ) -> None:
        self.docs_dir = os.path.abspath(docs_dir)
        self.store    = store
        self.parser   = parser
        self._observer: Observer | None = None

        os.makedirs(self.docs_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Perform an initial scan of existing documents, then start the
        background file-system watcher.
        """
        print(f"[Watcher] Watching folder: '{self.docs_dir}'")
        self._initial_scan()
        self._start_observer()

    def stop(self) -> None:
        """Stop the background watcher gracefully."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            print("[Watcher] Stopped.")

    def ingest_file(self, path: str) -> None:
        """Manually ingest a single file (useful for testing)."""
        filename = os.path.basename(path)
        print(f"[Watcher] Manual ingest: '{filename}'")
        fingerprint = _file_fingerprint(path)
        chunks = self.parser.parse(path)
        if chunks:
            self.store.add_chunks(
                chunks,
                source=filename,
                source_fingerprint=fingerprint,
                suppress_unchanged_log=True,
            )
        else:
            print(f"[Watcher] No text extracted from '{filename}'.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _initial_scan(self) -> None:
        """Index all supported files already in docs_dir."""
        files = [
            os.path.join(root, fname)
            for root, _, fnames in os.walk(self.docs_dir)
            for fname in fnames
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            print("[Watcher] No existing documents found — drop files into "
                  f"'{self.docs_dir}' to index them.")
            return

        print(f"[Watcher] Found {len(files)} existing file(s). Indexing …")
        for path in files:
            filename = os.path.basename(path)
            fingerprint = _file_fingerprint(path)
            if not self.store.should_ingest(filename, fingerprint):
                continue
            chunks = self.parser.parse(path)
            if chunks:
                self.store.add_chunks(
                    chunks,
                    source=filename,
                    source_fingerprint=fingerprint,
                    suppress_unchanged_log=True,
                )
            else:
                print(f"[Watcher] No text from '{filename}' — skipping.")

    def _start_observer(self) -> None:
        handler = _DocEventHandler(store=self.store, parser=self.parser)
        self._observer = Observer()
        self._observer.schedule(handler, path=self.docs_dir, recursive=True)
        self._observer.daemon = True   # dies when main thread exits
        self._observer.start()
        print("[Watcher] Watching for new files … (running in background)")


def _file_fingerprint(path: str) -> str | None:
    """
    Build a stable fingerprint for change detection using mtime + SHA256.
    Returns None if the file cannot be read.
    """
    try:
        stat = os.stat(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return f"{stat.st_mtime_ns}:{stat.st_size}:{h.hexdigest()}"
    except Exception:
        return None