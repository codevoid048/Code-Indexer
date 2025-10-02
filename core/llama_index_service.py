from calendar import c
import os
from typing import List, Dict, Any, Optional, Tuple

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    QueryBundle,
    Settings
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from core.vectorstore import TextBasedVectorStore
from core.parser import MultiLanguageParser
from models import CodeFile, Symbol, SearchResult, SearchQuery
from config import Config

config = Config()

class LlamaIndexService:
    """LlamaIndex wrapper for enhanced RAG capabilities over existing FAISS indices."""

    def __init__(self, vectorstore: TextBasedVectorStore):
        self.vectorstore = vectorstore
        self.llm = None
        self.embed_model = None
        self.index: Optional[VectorStoreIndex] = None
        self.documents: List[Document] = []

        # Initialize LLM and embeddings
        self._setup_llm_and_embeddings()

        # Clean up metadata for missing files
        self.cleanup_missing_files()

        # Build LlamaIndex from existing data
        self._build_index_from_existing_data()
        
        # Scan for any new files in data directory
        self.scan_and_index_data_directory()

    def scan_and_index_data_directory(self, data_dir: str = "data"):
        """Scan data directory and index any new files."""
        try:
            if not os.path.exists(data_dir):
                print(f"Data directory {data_dir} does not exist")
                return

            parser = MultiLanguageParser()
            
            # File patterns to include/exclude (same as IncrementalIndexer defaults)
            include_patterns = [
                "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
                "*.java", "*.go", "*.rs", "*.cpp", "*.c", "*.h", "*.hpp"
            ]
            exclude_patterns = [
                "node_modules/*", "__pycache__/*", ".git/*", "*.pyc", 
                "*.pyo", "*.so", "*.dll", ".venv/*", "venv/*",
                "build/*", "dist/*", "*.egg-info/*"
            ]
            
            indexed_paths = {cf.absolute_path for cf in self.vectorstore.file_metadata.values()}
            
            def should_process_file(file_path: str) -> bool:
                """Check if a file should be processed."""
                from pathlib import Path
                path = Path(file_path)
                
                # Check exclude patterns first
                for pattern in exclude_patterns:
                    if pattern in str(path) or path.match(pattern):
                        return False
                
                # Check include patterns
                for pattern in include_patterns:
                    if path.match(pattern):
                        return True
                
                return False
            
            new_files = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path not in indexed_paths and should_process_file(file_path):
                        new_files.append(file_path)
            
            if new_files:
                print(f"Found {len(new_files)} new files to index")
                for file_path in new_files:
                    try:
                        code_file = parser.parse_file(file_path)
                        if code_file:
                            success = self.vectorstore.add_file(code_file)
                            if success:
                                # Add to LlamaIndex documents
                                doc = self._codefile_to_document(code_file)
                                if doc:
                                    self.documents.append(doc)
                                print(f"Indexed new file: {file_path}")
                            else:
                                print(f"Failed to index: {file_path}")
                    except Exception as e:
                        print(f"Error indexing {file_path}: {e}")
                
                # Rebuild index with new documents
                if self.documents:
                    try:
                        if hasattr(self.vectorstore, 'symbol_index') and self.vectorstore.symbol_index is not None:
                            faiss_store = FaissVectorStore(faiss_index=self.vectorstore.symbol_index)
                            storage_context = StorageContext.from_defaults(vector_store=faiss_store)
                            self.index = VectorStoreIndex.from_documents(
                                self.documents,
                                storage_context=storage_context,
                                embed_model=self.embed_model
                            )
                        else:
                            self.index = VectorStoreIndex.from_documents(
                                self.documents,
                                embed_model=self.embed_model
                            )
                        print("Index updated with new files")
                    except Exception as e:
                        print(f"Failed to rebuild index: {e}")
            else:
                print("No new files found")

        except Exception as e:
            print(f"Error scanning data directory: {e}")

    def cleanup_missing_files(self):
        """Remove metadata for files that no longer exist."""
        try:
            files_to_remove = []
            symbols_to_remove = []
            
            # Check files
            for file_id, code_file in self.vectorstore.file_metadata.items():
                if not os.path.exists(code_file.absolute_path):
                    files_to_remove.append(file_id)
            
            # Check symbols
            for symbol_id, symbol in self.vectorstore.symbol_metadata.items():
                if not os.path.exists(symbol.file_path):
                    symbols_to_remove.append(symbol_id)
            
            # Remove missing files and their symbols
            for file_id in files_to_remove:
                del self.vectorstore.file_metadata[file_id]
                del self.vectorstore.file_strings[file_id]
                # Remove from string_to_id
                keys_to_remove = [k for k, v in self.vectorstore.string_to_id.items() if v == file_id]
                for key in keys_to_remove:
                    del self.vectorstore.string_to_id[key]
            
            for symbol_id in symbols_to_remove:
                del self.vectorstore.symbol_metadata[symbol_id]
                del self.vectorstore.symbol_strings[symbol_id]
                keys_to_remove = [k for k, v in self.vectorstore.string_to_id.items() if v == symbol_id]
                for key in keys_to_remove:
                    del self.vectorstore.string_to_id[key]
            
            if files_to_remove or symbols_to_remove:
                print(f"Cleaned up {len(files_to_remove)} missing files and {len(symbols_to_remove)} missing symbols")
                # Save the cleaned metadata
                self.vectorstore.save_indices()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _setup_llm_and_embeddings(self):
        """Setup LLM and embedding models."""
        try:
            # Try Groq first (since they already use it)
            groq_api_key = config.groq_api_key
            if groq_api_key:
                self.llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)
            else:
                print("No Groq API key found")

            # Setup local HuggingFace embeddings
            try:
                self.embed_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    trust_remote_code=True
                )
                print("HuggingFace embeddings initialized successfully")
            except Exception as e:
                print(f"Failed to initialize HuggingFace embeddings: {e}")
                print("Falling back to mock embeddings")
                self.embed_model = None

            # Configure LlamaIndex settings
            if self.llm:
                Settings.llm = self.llm
            if self.embed_model:
                Settings.embed_model = self.embed_model
            else:
                # Fallback to mock embeddings if HuggingFace fails
                Settings.embed_model = None

        except Exception as e:
            print(f"Warning: Could not initialize LLM/embeddings: {e}")
            print("LlamaIndex will work in basic mode")

    def _build_index_from_existing_data(self):
        """Convert existing FAISS data to LlamaIndex documents."""
        try:
            # Convert files to documents - only process files that still exist
            for file_id, code_file in self.vectorstore.file_metadata.items():
                if os.path.exists(code_file.absolute_path):
                    doc = self._codefile_to_document(code_file)
                    if doc:
                        self.documents.append(doc)
                else:
                    print(f"Skipping missing file: {code_file.absolute_path}")

            # Convert symbols to documents (as separate docs for better granularity)
            for symbol_id, symbol in self.vectorstore.symbol_metadata.items():
                # Only include symbols from files that still exist
                if os.path.exists(symbol.file_path):
                    doc = self._symbol_to_document(symbol)
                    if doc:
                        self.documents.append(doc)

            # Create index from documents
            if self.documents:
                try:
                    # Try to use FAISS vector store if available
                    if hasattr(self.vectorstore, 'symbol_index') and self.vectorstore.symbol_index is not None:
                        faiss_store = FaissVectorStore(
                            faiss_index=self.vectorstore.symbol_index
                        )
                        storage_context = StorageContext.from_defaults(vector_store=faiss_store)
                        self.index = VectorStoreIndex.from_documents(
                            self.documents,
                            storage_context=storage_context,
                            embed_model=self.embed_model
                        )
                    else:
                        # Fallback: create index without FAISS wrapper
                        self.index = VectorStoreIndex.from_documents(
                            self.documents,
                            embed_model=self.embed_model
                        )
                except Exception as e:
                    print(f"FAISS integration failed, using basic index: {e}")
                    self.index = VectorStoreIndex.from_documents(
                        self.documents,
                        embed_model=self.embed_model
                    )

        except Exception as e:
            print(f"Error building LlamaIndex from existing data: {e}")
            # Create empty index as fallback
            self.index = None

    def _codefile_to_document(self, code_file: CodeFile) -> Optional[Document]:
        """Convert a CodeFile to a LlamaIndex Document."""
        try:
            # Read file content from disk since CodeFile doesn't store content
            file_content = ""
            try:
                with open(code_file.absolute_path, 'r', encoding=getattr(code_file, 'encoding', 'utf-8'), errors='ignore') as f:
                    file_content = f.read()
            except Exception as e:
                print(f"Warning: Could not read file content for {code_file.absolute_path}: {e}")
                file_content = f"[Could not read file content: {e}]"

            # Create comprehensive text content
            content_parts = [
                f"File: {code_file.path}",
                f"Language: {code_file.language.value}",
                f"Size: {len(file_content)} characters",
                "",
                "Content:",
                file_content
            ]

            # Add symbol information
            if code_file.symbols:
                content_parts.append("\nSymbols:")
                for symbol in code_file.symbols:
                    content_parts.append(f"- {symbol.type.value}: {symbol.name} (line {symbol.line_number})")

            content = "\n".join(content_parts)

            # Create document with metadata
            doc = Document(
                text=content,
                metadata={
                    "type": "file",
                    "path": code_file.path,
                    "language": code_file.language.value,
                    "symbol_count": len(code_file.symbols),
                    "file_size": len(file_content),
                    "last_modified": code_file.last_modified.isoformat() if code_file.last_modified else None
                },
                id_=f"file:{code_file.path}"
            )

            return doc

        except Exception as e:
            print(f"Error converting file {code_file.path} to document: {e}")
            return None

    def _symbol_to_document(self, symbol: Symbol) -> Optional[Document]:
        """Convert a Symbol to a LlamaIndex Document."""
        try:
            # Get language from the file that contains this symbol
            language = "unknown"
            try:
                # Try to get language from file metadata
                for code_file in self.vectorstore.file_metadata.values():
                    if code_file.path == symbol.file_path or code_file.absolute_path == symbol.file_path:
                        language = code_file.language.value
                        break
            except:
                pass

            # Create symbol content
            content_parts = [
                f"Symbol: {symbol.name}",
                f"Type: {symbol.type.value}",
                f"Language: {language}",
                f"File: {symbol.file_path}",
                f"Line: {symbol.line_number}",
            ]

            if hasattr(symbol, 'docstring') and symbol.docstring:
                content_parts.extend(["", "Documentation:", symbol.docstring])

            if hasattr(symbol, 'signature') and symbol.signature:
                content_parts.extend(["", "Signature:", symbol.signature])

            # Add context information
            if hasattr(symbol, 'parent_symbol') and symbol.parent_symbol:
                content_parts.append(f"Parent: {symbol.parent_symbol}")
            if hasattr(symbol, 'parameters') and symbol.parameters:
                content_parts.append(f"Parameters: {', '.join(symbol.parameters)}")
            if hasattr(symbol, 'return_type') and symbol.return_type:
                content_parts.append(f"Return Type: {symbol.return_type}")
            if hasattr(symbol, 'access_modifier') and symbol.access_modifier:
                content_parts.append(f"Access: {symbol.access_modifier}")
            if hasattr(symbol, 'is_async') and symbol.is_async:
                content_parts.append("Async: Yes")

            content = "\n".join(content_parts)

            # Create document with metadata
            doc = Document(
                text=content,
                metadata={
                    "type": "symbol",
                    "name": symbol.name,
                    "symbol_type": symbol.type.value,
                    "language": language,
                    "file_path": symbol.file_path,
                    "start_line": symbol.line_number,
                    "end_line": getattr(symbol, 'end_line', symbol.line_number),
                    "parent_symbol": getattr(symbol, 'parent_symbol', None),
                    "has_docstring": bool(getattr(symbol, 'docstring', None)),
                    "has_signature": bool(getattr(symbol, 'signature', None))
                },
                id_=f"symbol:{symbol.file_path}:{symbol.name}:{symbol.line_number}"
            )


            return doc

        except Exception as e:
            print(f"Error converting symbol {symbol.name} to document: {e}")
            return None

    def query(self, query_str: str, **kwargs) -> str:
        """Perform a natural language query using LlamaIndex RAG."""
        if not self.index:
            return "LlamaIndex not initialized. Check your configuration and API keys."

        try:
            # Use the new query API for LlamaIndex 0.14.0
            query_engine = self.index.as_query_engine(
                similarity_top_k=kwargs.get('top_k', 10)
            )
            response = query_engine.query(query_str)
            return str(response)

        except Exception as e:
            return f"Error performing query: {str(e)}"

    def query_with_sources(self, query_str: str, **kwargs) -> Dict[str, Any]:
        """Query with source information and metadata."""
        if not self.index:
            return {"error": "LlamaIndex not initialized"}

        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=kwargs.get('top_k', 10)
            )

            response = query_engine.query(query_str)

            # Extract sources and metadata (API might have changed in 0.14.0)
            sources = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    sources.append({
                        "content": node.node.text if hasattr(node, 'node') else str(node),
                        "metadata": node.node.metadata if hasattr(node, 'node') and hasattr(node.node, 'metadata') else {},
                        "score": getattr(node, 'score', None)
                    })

            return {
                "answer": str(response),
                "sources": sources,
                "query": query_str
            }

        except Exception as e:
            return {"error": str(e), "query": query_str}

    def rebuild_index(self):
        """Rebuild the LlamaIndex from current vectorstore data."""
        self.documents = []
        self._build_index_from_existing_data()

    def get_stats(self) -> Dict[str, Any]:
        """Get LlamaIndex statistics."""
        return {
            "documents_count": len(self.documents),
            "has_index": self.index is not None,
            "has_llm": self.llm is not None,
            "has_embeddings": self.embed_model is not None,
            "llm_type": type(self.llm).__name__ if self.llm else None,
            "embedding_type": type(self.embed_model).__name__ if self.embed_model else None
        }

    def query_code_explanation(self, query_str: str, context_lines: int = 3) -> Dict[str, Any]:
        """Query for code explanations with enhanced context."""
        if not self.index:
            return {"error": "LlamaIndex not initialized"}

        try:
            # Enhance query for better code explanation
            enhanced_query = f"Explain this code concept or functionality: {query_str}. Include relevant code examples and context."

            response = self.query_with_sources(enhanced_query, top_k=15, response_mode="tree_summarize")

            # Add code-specific enhancements
            if "sources" in response:
                for source in response["sources"]:
                    if source.get("metadata", {}).get("type") == "symbol":
                        # Add surrounding context for symbols
                        self._add_symbol_context(source, context_lines)

            return response

        except Exception as e:
            return {"error": str(e)}

    def query_similar_patterns(self, code_snippet: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Find similar code patterns across the codebase."""
        if not self.index:
            return {"error": "LlamaIndex not initialized"}

        try:
            query = f"Find code patterns similar to: {code_snippet}"
            if language:
                query += f" in {language}"

            return self.query_with_sources(query, top_k=20, response_mode="simple_summarize")

        except Exception as e:
            return {"error": str(e)}

    def query_architecture_overview(self, component: str) -> Dict[str, Any]:
        """Get architectural overview of a component or system."""
        if not self.index:
            return {"error": "LlamaIndex not initialized"}

        try:
            query = f"Provide an architectural overview of the {component} component. Include main classes, functions, and their relationships."

            return self.query_with_sources(query, top_k=25, response_mode="tree_summarize")

        except Exception as e:
            return {"error": str(e)}

    def query_best_practices(self, topic: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Find examples of best practices for a specific topic."""
        if not self.index:
            return {"error": "LlamaIndex not initialized"}

        try:
            query = f"Show examples of best practices for {topic}"
            if language:
                query += f" in {language}"

            return self.query_with_sources(query, top_k=15, response_mode="simple_summarize")

        except Exception as e:
            return {"error": str(e)}

    def _add_symbol_context(self, source: Dict[str, Any], context_lines: int):
        """Add surrounding code context to a symbol source."""
        try:
            metadata = source.get("metadata", {})
            file_path = metadata.get("file_path")
            start_line = metadata.get("start_line")

            if file_path and start_line and os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                # Get context around the symbol
                start_idx = max(0, start_line - context_lines - 1)
                end_idx = min(len(lines), start_line + context_lines)

                context_lines_text = lines[start_idx:end_idx]
                source["context"] = {
                    "lines": context_lines_text,
                    "start_line": start_idx + 1,
                    "end_line": end_idx
                }

        except Exception as e:
            print(f"Error adding context to symbol: {e}")

    def conversational_query(self, conversation_history: List[Dict[str, str]], current_query: str) -> Dict[str, Any]:
        """Perform a conversational query with context from previous interactions."""
        if not self.index:
            return {"error": "LlamaIndex not initialized"}

        try:
            # Build context from conversation history
            context_parts = []
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context_parts.append(f"{role}: {content}")

            context = "\n".join(context_parts)
            enhanced_query = f"Context from previous conversation:\n{context}\n\nCurrent question: {current_query}"

            return self.query_with_sources(enhanced_query, top_k=12, response_mode="tree_summarize")

        except Exception as e:
            return {"error": str(e)}

    def analyze_code_complexity(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze code complexity patterns in the codebase."""
        if not self.index:
            return {"error": "LlamaIndex not initialized"}

        try:
            query = "Analyze the code complexity patterns in this codebase."
            if file_path:
                query += f" Focus on the file: {file_path}"

            query += " Look for complex functions, nested structures, and potential refactoring opportunities."

            return self.query_with_sources(query, top_k=30, response_mode="tree_summarize")

        except Exception as e:
            return {"error": str(e)}