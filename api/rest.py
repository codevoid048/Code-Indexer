import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.parser import MultiLanguageParser
from core.vectorstore import TextBasedVectorStore  
from core.search import CodeSearchEngine
from core.incremental import IncrementalIndexer
from core.groq_analyzer import GroqCodeAnalyzer
from models import (
    SearchQuery, SearchResult, IndexRequest, IndexStats, 
    CodeFile, Symbol, AnalysisRequest, AnalysisResult,
    Language, SymbolType
)


class CodeIndexerAPI:
    """Main API class for the code indexer."""
    
    def __init__(self):
        self.parser = MultiLanguageParser()
        self.vectorstore = TextBasedVectorStore()
        self.search_engine = CodeSearchEngine(self.vectorstore, self.parser)
        self.incremental_indexer = IncrementalIndexer(self.parser, self.vectorstore)
        self.groq_analyzer = GroqCodeAnalyzer()
        
        # Track indexing status
        self.indexing_status = {
            'is_indexing': False,
            'current_file': None,
            'progress': 0,
            'total_files': 0,
            'start_time': None,
            'last_error': None
        }


# Global instance
api_instance = CodeIndexerAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("Starting Code Indexer API...")
    
    # Load existing indices
    try:
        stats = api_instance.vectorstore.get_stats()
        print(f"Loaded index with {stats.total_files} files and {stats.total_symbols} symbols")
    except Exception as e:
        print(f"Error loading existing indices: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down Code Indexer API...")
    
    # Save indices
    try:
        api_instance.vectorstore.save_indices()
        print("Indices saved successfully")
    except Exception as e:
        print(f"Error saving indices: {e}")
    
    # Stop file watchers
    api_instance.incremental_indexer.stop_all_watchers()
    print("File watchers stopped")


# Create FastAPI app
app = FastAPI(
    title="Code Indexer API",
    description="Advanced RAG system for code parsing, indexing and searching using FAISS",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = api_instance.vectorstore.get_stats()
    indexer_stats = api_instance.incremental_indexer.get_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "index_stats": stats.dict(),
        "indexer_stats": indexer_stats,
        "indexing_status": api_instance.indexing_status
    }


# Index management endpoints
@app.post("/index", response_model=Dict[str, Any])
async def index_repository(request: IndexRequest, background_tasks: BackgroundTasks):
    """Index a repository or directory."""
    if not os.path.exists(request.path):
        raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")
    
    if not os.path.isdir(request.path):
        raise HTTPException(status_code=400, detail="Path must be a directory")
    
    # Start background indexing
    background_tasks.add_task(
        _index_directory_background,
        request.path,
        request.recursive,
        request.include_patterns,
        request.exclude_patterns,
        request.force_reindex
    )
    
    return {
        "message": "Indexing started",
        "path": request.path,
        "recursive": request.recursive,
        "status": "processing"
    }


async def _index_directory_background(
    path: str,
    recursive: bool,
    include_patterns: List[str],
    exclude_patterns: List[str],
    force_reindex: bool
):
    """Background task for indexing a directory."""
    try:
        api_instance.indexing_status['is_indexing'] = True
        api_instance.indexing_status['start_time'] = datetime.now()
        api_instance.indexing_status['last_error'] = None
        
        # Update indexer patterns
        api_instance.incremental_indexer.include_patterns = include_patterns
        api_instance.incremental_indexer.exclude_patterns = exclude_patterns
        
        # Count total files first
        total_files = 0
        for root, dirs, files in os.walk(path) if recursive else [(path, [], os.listdir(path))]:
            for file in files:
                file_path = os.path.join(root, file)
                if api_instance.incremental_indexer._should_process_file(file_path):
                    total_files += 1
        
        api_instance.indexing_status['total_files'] = total_files
        api_instance.indexing_status['progress'] = 0
        
        # Index files
        processed = 0
        for root, dirs, files in os.walk(path) if recursive else [(path, [], os.listdir(path))]:
            for file in files:
                file_path = os.path.join(root, file)
                
                if api_instance.incremental_indexer._should_process_file(file_path):
                    api_instance.indexing_status['current_file'] = file_path
                    
                    try:
                        # Remove existing if force reindex
                        if force_reindex:
                            api_instance.vectorstore.remove_file(file_path)
                        
                        # Parse and add file
                        code_file = api_instance.parser.parse_file(file_path)
                        if code_file:
                            api_instance.vectorstore.add_file(code_file)
                            processed += 1
                    
                    except Exception as e:
                        print(f"Error indexing {file_path}: {e}")
                        api_instance.indexing_status['last_error'] = str(e)
                    
                    api_instance.indexing_status['progress'] = processed
        
        # Start watching the directory
        api_instance.incremental_indexer.start_watching(path, recursive)
        
    except Exception as e:
        api_instance.indexing_status['last_error'] = str(e)
        print(f"Error during background indexing: {e}")
    
    finally:
        api_instance.indexing_status['is_indexing'] = False
        api_instance.indexing_status['current_file'] = None
        
        # Save indices after indexing
        try:
            api_instance.vectorstore.save_indices()
        except Exception as e:
            print(f"Error saving indices after indexing: {e}")


@app.get("/index/status")
async def get_indexing_status():
    """Get current indexing status."""
    return api_instance.indexing_status


@app.get("/index/stats", response_model=IndexStats)
async def get_index_stats():
    """Get index statistics."""
    return api_instance.vectorstore.get_stats()


@app.delete("/index/file")
async def remove_file_from_index(file_path: str):
    """Remove a specific file from the index."""
    success = api_instance.vectorstore.remove_file(file_path)
    
    if success:
        return {"message": f"File removed from index: {file_path}"}
    else:
        raise HTTPException(status_code=404, detail=f"File not found in index: {file_path}")


@app.post("/index/save")
async def save_index():
    """Manually save the current index to disk."""
    try:
        api_instance.vectorstore.save_indices()
        return {"message": "Index saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving index: {str(e)}")


# Search endpoints
@app.post("/search", response_model=List[SearchResult])
async def search(query: SearchQuery):
    """Search across indexed code."""
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        results = api_instance.search_engine.search(query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/search/symbols", response_model=List[SearchResult])
async def search_symbols(
    q: str = Query(..., description="Search query"),
    symbol_type: Optional[SymbolType] = Query(None, description="Filter by symbol type"),
    language: Optional[Language] = Query(None, description="Filter by language"),
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    limit: int = Query(20, description="Maximum results", ge=1, le=100)
):
    """Search for symbols."""
    query = SearchQuery(
        query=q,
        symbol_type=symbol_type,
        language=language,
        file_path=file_path,
        limit=limit
    )
    
    try:
        results = api_instance.search_engine.search_symbols(query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symbol search error: {str(e)}")


@app.get("/search/files", response_model=List[SearchResult])
async def search_files(
    q: str = Query(..., description="Search query"),
    language: Optional[Language] = Query(None, description="Filter by language"),
    limit: int = Query(20, description="Maximum results", ge=1, le=100)
):
    """Search for files."""
    query = SearchQuery(
        query=q,
        language=language,
        limit=limit
    )
    
    try:
        results = api_instance.search_engine.search_files(query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File search error: {str(e)}")


@app.get("/search/dependencies", response_model=List[SearchResult])
async def search_dependencies(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, description="Maximum results", ge=1, le=100)
):
    """Search for dependencies."""
    query = SearchQuery(query=q, limit=limit)
    
    try:
        results = api_instance.search_engine.search_dependencies(query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dependency search error: {str(e)}")


@app.get("/search/regex", response_model=List[SearchResult])
async def search_regex(
    pattern: str = Query(..., description="Regex pattern"),
    search_type: str = Query("symbol", description="Search type: symbol, file, or combined")
):
    """Search using regular expressions."""
    try:
        from core.search import SearchType
        
        if search_type == "symbol":
            type_enum = SearchType.SYMBOL
        elif search_type == "file":
            type_enum = SearchType.FILE
        elif search_type == "combined":
            type_enum = SearchType.COMBINED
        else:
            raise HTTPException(status_code=400, detail="Invalid search_type")
        
        results = api_instance.search_engine.search_by_regex(pattern, type_enum)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regex search error: {str(e)}")


# Symbol and file information endpoints
@app.get("/symbol/{symbol_name}", response_model=Optional[Symbol])
async def get_symbol(symbol_name: str, file_path: Optional[str] = Query(None)):
    """Get detailed information about a specific symbol."""
    symbol = api_instance.vectorstore.get_symbol_by_name(symbol_name, file_path)
    
    if not symbol:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {symbol_name}")
    
    return symbol


@app.get("/symbol/{symbol_name}/references", response_model=List[SearchResult])
async def get_symbol_references(symbol_name: str, file_path: Optional[str] = Query(None)):
    """Get all references to a specific symbol."""
    try:
        results = api_instance.search_engine.find_symbol_references(symbol_name, file_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding references: {str(e)}")


@app.get("/symbol/{symbol_name}/hierarchy")
async def get_symbol_hierarchy(symbol_name: str, file_path: Optional[str] = Query(None)):
    """Get the hierarchy of a symbol (parents, children, siblings)."""
    try:
        hierarchy = api_instance.search_engine.get_symbol_hierarchy(symbol_name, file_path)
        
        # Convert to serializable format
        return {
            "parents": [s.dict() for s in hierarchy["parents"]],
            "children": [s.dict() for s in hierarchy["children"]],
            "siblings": [s.dict() for s in hierarchy["siblings"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hierarchy: {str(e)}")


@app.get("/symbol/{symbol_name}/similar", response_model=List[SearchResult])
async def get_similar_symbols(
    symbol_name: str,
    limit: int = Query(10, description="Maximum results", ge=1, le=50)
):
    """Get symbols similar to the given name."""
    try:
        results = api_instance.search_engine.suggest_similar_symbols(symbol_name, limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar symbols: {str(e)}")


@app.get("/file/symbols", response_model=List[Symbol])
async def get_file_symbols(file_path: str):
    """Get all symbols in a specific file."""
    symbols = api_instance.vectorstore.get_file_symbols(file_path)
    
    if not symbols:
        # Check if file exists in index
        file_found = False
        for code_file in api_instance.vectorstore.file_metadata.values():
            if code_file.path == file_path:
                file_found = True
                break
        
        if not file_found:
            raise HTTPException(status_code=404, detail=f"File not found in index: {file_path}")
    
    return symbols


# File watcher management endpoints
@app.post("/watch/start")
async def start_watching(
    directory: str,
    recursive: bool = Query(True, description="Watch subdirectories recursively")
):
    """Start watching a directory for file changes."""
    if not os.path.isdir(directory):
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
    
    success = api_instance.incremental_indexer.start_watching(directory, recursive)
    
    if success:
        return {"message": f"Started watching: {directory}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to start watching: {directory}")


@app.post("/watch/stop")
async def stop_watching(directory: str):
    """Stop watching a specific directory."""
    success = api_instance.incremental_indexer.stop_watching(directory)
    
    if success:
        return {"message": f"Stopped watching: {directory}"}
    else:
        raise HTTPException(status_code=404, detail=f"Not watching directory: {directory}")


@app.get("/watch/status")
async def get_watch_status():
    """Get status of all file watchers."""
    directories = api_instance.incremental_indexer.get_watched_directories()
    
    status = {}
    for directory in directories:
        status[directory] = {
            "is_watching": api_instance.incremental_indexer.is_watching(directory)
        }
    
    return {
        "watched_directories": status,
        "total_watchers": len(directories),
        "indexer_stats": api_instance.incremental_indexer.get_stats()
    }


# Utility endpoints
@app.get("/languages")
async def get_supported_languages():
    """Get list of supported programming languages."""
    return [lang.value for lang in Language]


@app.get("/symbol-types")
async def get_symbol_types():
    """Get list of supported symbol types."""
    return [st.value for st in SymbolType]


# Code analysis endpoints (Groq integration)
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_code(request: AnalysisRequest):
    """Analyze code using Groq AI."""
    if not api_instance.groq_analyzer.is_available():
        raise HTTPException(status_code=503, detail="Groq API not configured. Set GROQ_API_KEY environment variable.")
    
    try:
        result = await api_instance.groq_analyzer.analyze_code(request, api_instance.vectorstore)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/analyze/symbol/{symbol_name}/explain", response_model=AnalysisResult)
async def explain_symbol(
    symbol_name: str,
    file_path: Optional[str] = Query(None, description="Specific file path to search in")
):
    """Explain what a specific symbol does."""
    if not api_instance.groq_analyzer.is_available():
        raise HTTPException(status_code=503, detail="Groq API not configured.")
    
    # Find the symbol
    symbol = api_instance.vectorstore.get_symbol_by_name(symbol_name, file_path)
    if not symbol:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {symbol_name}")
    
    try:
        result = await api_instance.groq_analyzer.explain_symbol(symbol, api_instance.vectorstore)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/analyze/symbol/{symbol_name}/improve", response_model=AnalysisResult)
async def suggest_symbol_improvements(
    symbol_name: str,
    file_path: Optional[str] = Query(None, description="Specific file path to search in")
):
    """Suggest improvements for a specific symbol."""
    if not api_instance.groq_analyzer.is_available():
        raise HTTPException(status_code=503, detail="Groq API not configured.")
    
    symbol = api_instance.vectorstore.get_symbol_by_name(symbol_name, file_path)
    if not symbol:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {symbol_name}")
    
    try:
        result = await api_instance.groq_analyzer.suggest_improvements(symbol, api_instance.vectorstore)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/analyze/file/bugs", response_model=AnalysisResult)
async def find_file_bugs(file_path: str):
    """Analyze a file for potential bugs."""
    if not api_instance.groq_analyzer.is_available():
        raise HTTPException(status_code=503, detail="Groq API not configured.")
    
    # Check if file exists in index
    file_found = False
    for code_file in api_instance.vectorstore.file_metadata.values():
        if code_file.path == file_path:
            file_found = True
            break
    
    if not file_found:
        raise HTTPException(status_code=404, detail=f"File not found in index: {file_path}")
    
    try:
        result = await api_instance.groq_analyzer.find_bugs(file_path, api_instance.vectorstore)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/analyze/symbol/{symbol_name}/document", response_model=AnalysisResult)
async def generate_symbol_documentation(
    symbol_name: str,
    file_path: Optional[str] = Query(None, description="Specific file path to search in")
):
    """Generate documentation for a symbol."""
    if not api_instance.groq_analyzer.is_available():
        raise HTTPException(status_code=503, detail="Groq API not configured.")
    
    symbol = api_instance.vectorstore.get_symbol_by_name(symbol_name, file_path)
    if not symbol:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {symbol_name}")
    
    try:
        result = await api_instance.groq_analyzer.generate_documentation(symbol, api_instance.vectorstore)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/analyze/file/complexity", response_model=AnalysisResult)
async def analyze_file_complexity(file_path: str):
    """Analyze code complexity and suggest refactoring."""
    if not api_instance.groq_analyzer.is_available():
        raise HTTPException(status_code=503, detail="Groq API not configured.")
    
    # Check if file exists in index
    file_found = False
    for code_file in api_instance.vectorstore.file_metadata.values():
        if code_file.path == file_path:
            file_found = True
            break
    
    if not file_found:
        raise HTTPException(status_code=404, detail=f"File not found in index: {file_path}")
    
    try:
        result = await api_instance.groq_analyzer.analyze_complexity(file_path, api_instance.vectorstore)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)