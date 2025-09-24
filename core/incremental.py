import os
import time
from pathlib import Path
from typing import Dict, List, Set, Callable, TYPE_CHECKING, Any
from threading import Thread, Event
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

from core.parser import MultiLanguageParser
from core.vectorstore import TextBasedVectorStore
from models import CodeFile


class CodeFileWatcher(FileSystemEventHandler):
    """File system event handler for code files."""
    
    def __init__(self, 
                 parser: MultiLanguageParser,
                 vectorstore: TextBasedVectorStore,
                 include_patterns: List[str],
                 exclude_patterns: List[str],
                 callback: Callable[[str, str], None] = None):
        self.parser = parser
        self.vectorstore = vectorstore
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.callback = callback
        
        # Debouncing to prevent excessive processing
        self.debounce_delay = 1.0  # seconds
        self.pending_changes: Dict[str, float] = {}
        self.processing_thread = None
        self.stop_event = Event()
        
        self._start_processing_thread()
    
    def _start_processing_thread(self):
        """Start the background thread for processing file changes."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = Thread(target=self._process_pending_changes, daemon=True)
            self.processing_thread.start()
    
    def _process_pending_changes(self):
        """Background thread to process pending file changes with debouncing."""
        while not self.stop_event.is_set():
            current_time = time.time()
            files_to_process = []
            
            # Find files that have been stable for the debounce period
            for file_path, timestamp in list(self.pending_changes.items()):
                if current_time - timestamp >= self.debounce_delay:
                    files_to_process.append(file_path)
                    del self.pending_changes[file_path]
            
            # Process stable files
            for file_path in files_to_process:
                self._process_file_change(file_path)
            
            # Sleep briefly before checking again
            time.sleep(0.5)
    
    def _should_process_file(self, file_path: str) -> bool:
        """Check if a file should be processed based on include/exclude patterns."""
        path = Path(file_path)
        
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if pattern in str(path) or path.match(pattern):
                return False
        
        # Check include patterns
        for pattern in self.include_patterns:
            if path.match(pattern):
                return True
        
        return False
    
    def _add_pending_change(self, file_path: str):
        """Add a file to the pending changes queue with timestamp."""
        if self._should_process_file(file_path):
            self.pending_changes[file_path] = time.time()
    
    def _process_file_change(self, file_path: str):
        """Process a single file change."""
        try:
            if os.path.exists(file_path):
                # File created or modified
                print(f"Processing file change: {file_path}")
                
                # Remove old version if it exists
                self.vectorstore.remove_file(file_path)
                
                # Parse and add new version
                code_file = self.parser.parse_file(file_path)
                if code_file:
                    success = self.vectorstore.add_file(code_file)
                    if success:
                        print(f"Updated index for: {file_path}")
                        if self.callback:
                            self.callback(file_path, "updated")
                    else:
                        print(f"Failed to update index for: {file_path}")
                else:
                    print(f"Failed to parse file: {file_path}")
            else:
                # File deleted
                print(f"Processing file deletion: {file_path}")
                success = self.vectorstore.remove_file(file_path)
                if success:
                    print(f"Removed from index: {file_path}")
                    if self.callback:
                        self.callback(file_path, "deleted")
                else:
                    print(f"Failed to remove from index: {file_path}")
        
        except Exception as e:
            print(f"Error processing file change {file_path}: {e}")
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._add_pending_change(event.src_path)
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._add_pending_change(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            self._add_pending_change(event.src_path)
    
    def on_moved(self, event):
        """Handle file move/rename events."""
        if not event.is_directory:
            # Handle as delete old + create new
            self._add_pending_change(event.src_path)
            self._add_pending_change(event.dest_path)
    
    def stop(self):
        """Stop the file watcher and processing thread."""
        self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)


class IncrementalIndexer:
    """Manages incremental indexing with file watching."""
    
    def __init__(self, 
                 parser: MultiLanguageParser,
                 vectorstore: TextBasedVectorStore,
                 include_patterns: List[str] = None,
                 exclude_patterns: List[str] = None):
        self.parser = parser
        self.vectorstore = vectorstore
        
        # Default patterns
        self.include_patterns = include_patterns or [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
            "*.java", "*.go", "*.rs", "*.cpp", "*.c", "*.h", "*.hpp"
        ]
        
        self.exclude_patterns = exclude_patterns or [
            "node_modules/*", "__pycache__/*", ".git/*", "*.pyc", 
            "*.pyo", "*.so", "*.dll", ".venv/*", "venv/*",
            "build/*", "dist/*", "*.egg-info/*"
        ]
        
        self.observers: Dict[str, Any] = {}  # Dict[str, Observer] - using Any to avoid Pylance type error
        self.watchers: Dict[str, CodeFileWatcher] = {}
        self.change_callbacks: List[Callable[[str, str], None]] = []
        
        # Statistics
        self.files_processed = 0
        self.last_update = datetime.now()
    
    def add_change_callback(self, callback: Callable[[str, str], None]):
        """Add a callback to be called when files change."""
        self.change_callbacks.append(callback)
    
    def _on_file_change(self, file_path: str, change_type: str):
        """Handle file change notifications."""
        self.files_processed += 1
        self.last_update = datetime.now()
        
        # Notify all callbacks
        for callback in self.change_callbacks:
            try:
                callback(file_path, change_type)
            except Exception as e:
                print(f"Error in change callback: {e}")
    
    def start_watching(self, directory: str, recursive: bool = True) -> bool:
        """Start watching a directory for file changes."""
        try:
            if not os.path.isdir(directory):
                print(f"Directory does not exist: {directory}")
                return False
            
            abs_dir = os.path.abspath(directory)
            
            # Stop existing watcher for this directory if any
            if abs_dir in self.observers:
                self.stop_watching(abs_dir)
            
            # Create file watcher
            watcher = CodeFileWatcher(
                parser=self.parser,
                vectorstore=self.vectorstore,
                include_patterns=self.include_patterns,
                exclude_patterns=self.exclude_patterns,
                callback=self._on_file_change
            )
            
            # Create observer
            observer = Observer()
            observer.schedule(watcher, abs_dir, recursive=recursive)
            observer.start()
            
            # Store references
            self.observers[abs_dir] = observer
            self.watchers[abs_dir] = watcher
            
            print(f"Started watching directory: {abs_dir}")
            return True
            
        except Exception as e:
            print(f"Error starting file watcher for {directory}: {e}")
            return False
    
    def stop_watching(self, directory: str) -> bool:
        """Stop watching a specific directory."""
        try:
            abs_dir = os.path.abspath(directory)
            
            if abs_dir in self.observers:
                # Stop observer
                observer = self.observers[abs_dir]
                observer.stop()
                observer.join(timeout=2.0)
                del self.observers[abs_dir]
                
                # Stop watcher
                if abs_dir in self.watchers:
                    watcher = self.watchers[abs_dir]
                    watcher.stop()
                    del self.watchers[abs_dir]
                
                print(f"Stopped watching directory: {abs_dir}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error stopping file watcher for {directory}: {e}")
            return False
    
    def stop_all_watchers(self):
        """Stop all file watchers."""
        for directory in list(self.observers.keys()):
            self.stop_watching(directory)
    
    def get_watched_directories(self) -> List[str]:
        """Get list of currently watched directories."""
        return list(self.observers.keys())
    
    def is_watching(self, directory: str) -> bool:
        """Check if a directory is currently being watched."""
        abs_dir = os.path.abspath(directory)
        return abs_dir in self.observers and self.observers[abs_dir].is_alive()
    
    def force_reindex_directory(self, directory: str, recursive: bool = True) -> int:
        """Force reindexing of all files in a directory."""
        processed_count = 0
        
        try:
            if not os.path.isdir(directory):
                print(f"Directory does not exist: {directory}")
                return 0
            
            print(f"Force reindexing directory: {directory}")
            
            # Find all files matching patterns
            files_to_process = []
            
            if recursive:
                for root, dirs, files in os.walk(directory):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        if self._should_process_file(file_path):
                            files_to_process.append(file_path)
            else:
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path) and self._should_process_file(file_path):
                        files_to_process.append(file_path)
            
            # Process each file
            for file_path in files_to_process:
                try:
                    # Remove existing version
                    self.vectorstore.remove_file(file_path)
                    
                    # Parse and add new version
                    code_file = self.parser.parse_file(file_path)
                    if code_file:
                        success = self.vectorstore.add_file(code_file)
                        if success:
                            processed_count += 1
                            print(f"Reindexed: {file_path}")
                        else:
                            print(f"Failed to reindex: {file_path}")
                    else:
                        print(f"Failed to parse: {file_path}")
                
                except Exception as e:
                    print(f"Error reindexing {file_path}: {e}")
            
            print(f"Reindexed {processed_count} files in {directory}")
            return processed_count
            
        except Exception as e:
            print(f"Error during force reindex: {e}")
            return processed_count
    
    def _should_process_file(self, file_path: str) -> bool:
        """Check if a file should be processed based on patterns."""
        path = Path(file_path)
        
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern in str(path) or path.match(pattern):
                return False
        
        # Check include patterns
        for pattern in self.include_patterns:
            if path.match(pattern):
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, any]:
        """Get incremental indexer statistics."""
        return {
            'watched_directories': len(self.observers),
            'active_watchers': sum(1 for obs in self.observers.values() if obs.is_alive()),
            'files_processed': self.files_processed,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'include_patterns': self.include_patterns,
            'exclude_patterns': self.exclude_patterns
        }