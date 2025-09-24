import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

import numpy as np
import faiss
from cachetools import LRUCache
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from models import CodeFile, Symbol, SearchResult, IndexStats, Language


class TextBasedVectorStore:
    """FAISS-based vector store for fast text-based code search."""
    
    def __init__(self, index_dir: str = "index_data", cache_size: int = 1000):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # FAISS indices for different search types
        self.symbol_index = None
        self.file_index = None
        self.content_index = None
        
        # Metadata storage
        self.symbol_metadata: Dict[int, Symbol] = {}
        self.file_metadata: Dict[int, CodeFile] = {}
        self.content_metadata: Dict[int, Dict] = {}
        
        # String to vector mappings for text-based search
        self.symbol_strings: Dict[int, str] = {}
        self.file_strings: Dict[int, str] = {}
        self.content_strings: Dict[int, str] = {}
        
        # Reverse lookup for string to ID mapping
        self.string_to_id: Dict[str, int] = {}
        
        # LRU caches
        self.symbol_cache = LRUCache(maxsize=cache_size)
        self.file_cache = LRUCache(maxsize=cache_size)
        
        # Current index counters
        self.symbol_counter = 0
        self.file_counter = 0
        self.content_counter = 0
        
        # Initialize embedding model
        self.embed_model = None
        self._init_embedding_model()
        
        # Load existing indices if they exist
        self._load_indices()
    
    def _init_embedding_model(self):
        """Initialize the HuggingFace embedding model."""
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                trust_remote_code=True
            )
            print("HuggingFace embeddings initialized successfully")
        except Exception as e:
            print(f"Failed to initialize HuggingFace embeddings: {e}")
            print("Falling back to simple hash-based vectors")
            self.embed_model = None
    
    def add_file(self, code_file: CodeFile) -> bool:
        """Add a code file and its symbols to the index."""
        try:
            # Generate searchable text for the file
            file_text = self._generate_file_searchtext(code_file)
            file_vector = self._text_to_vector(file_text)
            
            # Add to file index
            if self.file_index is None:
                self.file_index = faiss.IndexFlatIP(len(file_vector))  # Inner product for text similarity
            
            file_id = self.file_counter
            self.file_index.add(np.array([file_vector], dtype=np.float32))
            
            # Store metadata
            self.file_metadata[file_id] = code_file
            self.file_strings[file_id] = file_text
            self.string_to_id[f"file:{code_file.path}"] = file_id
            self.file_counter += 1
            
            # Add symbols from this file
            for symbol in code_file.symbols:
                self._add_symbol(symbol, code_file)
            
            # Cache the file
            self.file_cache[code_file.path] = code_file
            
            return True
            
        except Exception as e:
            print(f"Error adding file {code_file.path}: {e}")
            return False
    
    def _add_symbol(self, symbol: Symbol, parent_file: CodeFile):
        """Add a symbol to the symbol index."""
        try:
            # Generate searchable text for the symbol
            symbol_text = self._generate_symbol_searchtext(symbol, parent_file)
            symbol_vector = self._text_to_vector(symbol_text)
            
            # Add to symbol index
            if self.symbol_index is None:
                self.symbol_index = faiss.IndexFlatIP(len(symbol_vector))
            
            symbol_id = self.symbol_counter
            self.symbol_index.add(np.array([symbol_vector], dtype=np.float32))
            
            # Store metadata
            self.symbol_metadata[symbol_id] = symbol
            self.symbol_strings[symbol_id] = symbol_text
            self.string_to_id[f"symbol:{symbol.name}:{symbol.file_path}"] = symbol_id
            self.symbol_counter += 1
            
            # Cache the symbol
            self.symbol_cache[f"{symbol.file_path}:{symbol.name}"] = symbol
            
        except Exception as e:
            print(f"Error adding symbol {symbol.name}: {e}")
    
    def search_symbols(self, query: str, limit: int = 20, symbol_type: Optional[str] = None) -> List[SearchResult]:
        """Search for symbols using text-based matching."""
        if self.symbol_index is None or self.symbol_index.ntotal == 0:
            return []
        
        try:
            # Convert query to vector
            query_vector = self._text_to_vector(query.lower())
            
            # Search in FAISS index
            scores, indices = self.symbol_index.search(
                np.array([query_vector], dtype=np.float32), 
                min(limit * 2, self.symbol_index.ntotal)  # Get more results to filter
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                symbol = self.symbol_metadata.get(idx)
                if not symbol:
                    continue
                
                # Filter by symbol type if specified
                if symbol_type and symbol.type.value != symbol_type:
                    continue
                
                # Create search result
                result = SearchResult(
                    id=symbol.id,
                    type="symbol",
                    name=symbol.name,
                    file_path=symbol.file_path,
                    line_number=symbol.line_number,
                    score=float(score),
                    symbol=symbol
                )
                results.append(result)
            
            # Sort by score (descending) and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"Error searching symbols: {e}")
            return []
    
    def search_files(self, query: str, limit: int = 20, language: Optional[str] = None) -> List[SearchResult]:
        """Search for files using text-based matching."""
        if self.file_index is None or self.file_index.ntotal == 0:
            return []
        
        try:
            # Convert query to vector
            query_vector = self._text_to_vector(query.lower())
            
            # Search in FAISS index
            scores, indices = self.file_index.search(
                np.array([query_vector], dtype=np.float32),
                min(limit * 2, self.file_index.ntotal)
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                
                code_file = self.file_metadata.get(idx)
                if not code_file:
                    continue
                
                # Filter by language if specified
                if language and code_file.language.value != language:
                    continue
                
                # Create search result
                result = SearchResult(
                    id=code_file.id,
                    type="file",
                    name=os.path.basename(code_file.path),
                    file_path=code_file.path,
                    score=float(score),
                    file=code_file
                )
                results.append(result)
            
            # Sort by score and limit
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"Error searching files: {e}")
            return []
    
    def search_combined(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Combined search across symbols and files."""
        symbol_results = self.search_symbols(query, limit // 2)
        file_results = self.search_files(query, limit // 2)
        
        # Combine and sort by score
        all_results = symbol_results + file_results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:limit]
    
    def get_symbol_by_name(self, name: str, file_path: Optional[str] = None) -> Optional[Symbol]:
        """Get a symbol by name, optionally filtered by file path."""
        # Check cache first
        cache_key = f"{file_path}:{name}" if file_path else name
        if cache_key in self.symbol_cache:
            return self.symbol_cache[cache_key]
        
        # Search through all symbols
        for symbol in self.symbol_metadata.values():
            if symbol.name == name:
                if file_path is None or symbol.file_path == file_path:
                    self.symbol_cache[cache_key] = symbol
                    return symbol
        
        return None
    
    def get_file_symbols(self, file_path: str) -> List[Symbol]:
        """Get all symbols in a specific file."""
        symbols = []
        for symbol in self.symbol_metadata.values():
            if symbol.file_path == file_path:
                symbols.append(symbol)
        
        # Sort by line number
        symbols.sort(key=lambda s: s.line_number)
        return symbols
    
    def remove_file(self, file_path: str) -> bool:
        """Remove a file and its symbols from the index."""
        try:
            # This is a simplified removal - in a full implementation,
            # you'd need to rebuild the FAISS indices after removal
            # For now, we'll mark items as removed in metadata
            
            # Find and remove file
            file_id_to_remove = None
            for file_id, code_file in self.file_metadata.items():
                if code_file.path == file_path:
                    file_id_to_remove = file_id
                    break
            
            if file_id_to_remove is not None:
                del self.file_metadata[file_id_to_remove]
                if file_id_to_remove in self.file_strings:
                    del self.file_strings[file_id_to_remove]
            
            # Find and remove associated symbols
            symbols_to_remove = []
            for symbol_id, symbol in self.symbol_metadata.items():
                if symbol.file_path == file_path:
                    symbols_to_remove.append(symbol_id)
            
            for symbol_id in symbols_to_remove:
                del self.symbol_metadata[symbol_id]
                if symbol_id in self.symbol_strings:
                    del self.symbol_strings[symbol_id]
            
            # Clear from caches
            if file_path in self.file_cache:
                del self.file_cache[file_path]
            
            # Remove symbol cache entries for this file
            keys_to_remove = [k for k in self.symbol_cache.keys() if k.startswith(f"{file_path}:")]
            for key in keys_to_remove:
                del self.symbol_cache[key]
            
            return True
            
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")
            return False
    
    def get_stats(self) -> IndexStats:
        """Get current index statistics."""
        # Count symbols by type
        symbol_types = {}
        for symbol in self.symbol_metadata.values():
            symbol_type = symbol.type.value
            symbol_types[symbol_type] = symbol_types.get(symbol_type, 0) + 1
        
        # Count files by language
        languages = {}
        for code_file in self.file_metadata.values():
            lang = code_file.language.value
            languages[lang] = languages.get(lang, 0) + 1
        
        # Estimate index size (rough approximation)
        index_size_mb = 0.0
        if self.symbol_index:
            index_size_mb += (self.symbol_index.ntotal * self.symbol_index.d * 4) / (1024 * 1024)
        if self.file_index:
            index_size_mb += (self.file_index.ntotal * self.file_index.d * 4) / (1024 * 1024)
        
        return IndexStats(
            total_files=len(self.file_metadata),
            total_symbols=len(self.symbol_metadata),
            total_dependencies=sum(len(f.dependencies) for f in self.file_metadata.values()),
            languages=languages,
            symbol_types=symbol_types,
            index_size_mb=round(index_size_mb, 2),
            last_updated=datetime.now()
        )
    
    def save_indices(self):
        """Save FAISS indices and metadata to disk."""
        try:
            # Save FAISS indices
            if self.symbol_index and self.symbol_index.ntotal > 0:
                faiss.write_index(self.symbol_index, str(self.index_dir / "symbol_index.faiss"))
            
            if self.file_index and self.file_index.ntotal > 0:
                faiss.write_index(self.file_index, str(self.index_dir / "file_index.faiss"))
            
            # Save metadata
            metadata = {
                'symbol_metadata': {k: self._symbol_to_dict(v) for k, v in self.symbol_metadata.items()},
                'file_metadata': {k: self._file_to_dict(v) for k, v in self.file_metadata.items()},
                'symbol_strings': self.symbol_strings,
                'file_strings': self.file_strings,
                'string_to_id': self.string_to_id,
                'symbol_counter': self.symbol_counter,
                'file_counter': self.file_counter,
                'content_counter': self.content_counter
            }
            
            with open(self.index_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Indices saved to {self.index_dir}")
            
        except Exception as e:
            print(f"Error saving indices: {e}")
    
    def _load_indices(self):
        """Load FAISS indices and metadata from disk."""
        try:
            # Load FAISS indices
            symbol_index_path = self.index_dir / "symbol_index.faiss"
            if symbol_index_path.exists():
                loaded_index = faiss.read_index(str(symbol_index_path))
                # Check if dimensions match (384 for BGE embeddings)
                if loaded_index.d != 384:
                    print(f"Symbol index has wrong dimensions ({loaded_index.d}), rebuilding...")
                    self.symbol_index = None
                else:
                    self.symbol_index = loaded_index
            
            file_index_path = self.index_dir / "file_index.faiss"
            if file_index_path.exists():
                loaded_index = faiss.read_index(str(file_index_path))
                if loaded_index.d != 384:
                    print(f"File index has wrong dimensions ({loaded_index.d}), rebuilding...")
                    self.file_index = None
                else:
                    self.file_index = loaded_index
            
            # Load metadata
            metadata_path = self.index_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Restore metadata with proper type conversion
                self.symbol_metadata = {
                    int(k): self._dict_to_symbol(v) 
                    for k, v in metadata.get('symbol_metadata', {}).items()
                }
                self.file_metadata = {
                    int(k): self._dict_to_file(v) 
                    for k, v in metadata.get('file_metadata', {}).items()
                }
                
                self.symbol_strings = {int(k): v for k, v in metadata.get('symbol_strings', {}).items()}
                self.file_strings = {int(k): v for k, v in metadata.get('file_strings', {}).items()}
                self.string_to_id = metadata.get('string_to_id', {})
                
                self.symbol_counter = metadata.get('symbol_counter', 0)
                self.file_counter = metadata.get('file_counter', 0)
                self.content_counter = metadata.get('content_counter', 0)
                
                print(f"Loaded indices from {self.index_dir}")
                
        except Exception as e:
            print(f"Error loading indices: {e}")
    
    def _generate_file_searchtext(self, code_file: CodeFile) -> str:
        """Generate searchable text representation of a file."""
        parts = [
            code_file.path,
            os.path.basename(code_file.path),
            code_file.language.value,
            ' '.join(code_file.imports),
            ' '.join(code_file.exports),
            ' '.join([symbol.name for symbol in code_file.symbols])
        ]
        
        return ' '.join(filter(None, parts)).lower()
    
    def _generate_symbol_searchtext(self, symbol: Symbol, parent_file: CodeFile) -> str:
        """Generate searchable text representation of a symbol."""
        parts = [
            symbol.name,
            symbol.type.value,
            os.path.basename(symbol.file_path),
            symbol.signature or '',
            symbol.docstring or '',
            ' '.join(symbol.parameters),
            symbol.return_type or '',
            parent_file.language.value
        ]
        
        return ' '.join(filter(None, parts)).lower()
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector using embeddings or fallback to hash-based approach."""
        # Try to use HuggingFace embeddings if available
        if self.embed_model is not None:
            try:
                return np.array(self.embed_model.get_text_embedding(text), dtype=np.float32)
            except Exception as e:
                print(f"Embedding failed, falling back to hash-based: {e}")
        
        # Fallback: Create a fixed-size vector (384 dimensions to match BGE) based on text characteristics
        vector = np.zeros(384, dtype=np.float32)
        
        if not text:
            return vector
        
        # Character frequency features
        for char in text:
            if char.isalnum():
                vector[ord(char) % 384] += 1
        
        # Word-based features
        words = text.split()
        for i, word in enumerate(words):
            if word:
                hash_val = hash(word) % 384
                vector[hash_val] += 1
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _symbol_to_dict(self, symbol: Symbol) -> Dict:
        """Convert Symbol to dictionary for JSON serialization."""
        data = symbol.model_dump()
        # Convert any datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    def _dict_to_symbol(self, data: Dict) -> Symbol:
        """Convert dictionary back to Symbol object."""
        # Convert ISO strings back to datetime objects
        for key, value in data.items():
            if isinstance(value, str) and 'T' in value:  # Simple check for ISO datetime format
                try:
                    data[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass  # Not a datetime string, keep as is
        return Symbol(**data)
    
    def _file_to_dict(self, code_file: CodeFile) -> Dict:
        """Convert CodeFile to dictionary for JSON serialization."""
        data = code_file.model_dump()
        # Convert any datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    def _dict_to_file(self, data: Dict) -> CodeFile:
        """Convert dictionary back to CodeFile object."""
        # Convert ISO strings back to datetime objects
        for key, value in data.items():
            if isinstance(value, str) and 'T' in value:  # Simple check for ISO datetime format
                try:
                    data[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass  # Not a datetime string, keep as is
        return CodeFile(**data)