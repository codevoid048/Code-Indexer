from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class SymbolType(str, Enum):
    """Types of code symbols that can be extracted."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    INTERFACE = "interface"
    ENUM = "enum"
    TYPE_ALIAS = "type_alias"
    PROPERTY = "property"


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    UNKNOWN = "unknown"


class Dependency(BaseModel):
    """Represents a code dependency (import/require)."""
    name: str = Field(..., description="Name of the dependency")
    type: str = Field(..., description="Type of dependency (import, require, etc.)")
    source: Optional[str] = Field(None, description="Source module/package")
    is_external: bool = Field(False, description="Whether it's an external package")
    line_number: int = Field(..., description="Line where dependency is declared")


class Symbol(BaseModel):
    """Represents a code symbol (function, class, variable, etc.)."""
    id: str = Field(..., description="Unique identifier for the symbol")
    name: str = Field(..., description="Name of the symbol")
    type: SymbolType = Field(..., description="Type of symbol")
    file_path: str = Field(..., description="Path to the file containing the symbol")
    line_number: int = Field(..., description="Line number where symbol is defined")
    column: int = Field(0, description="Column number where symbol is defined")
    end_line: Optional[int] = Field(None, description="End line of the symbol")
    signature: Optional[str] = Field(None, description="Function/method signature")
    docstring: Optional[str] = Field(None, description="Documentation string")
    parent_symbol: Optional[str] = Field(None, description="Parent symbol ID (for methods in classes)")
    parameters: List[str] = Field(default_factory=list, description="Function/method parameters")
    return_type: Optional[str] = Field(None, description="Return type annotation")
    access_modifier: Optional[str] = Field(None, description="Access modifier (public, private, etc.)")
    is_async: bool = Field(False, description="Whether function/method is async")
    complexity: int = Field(0, description="Cyclomatic complexity")
    calls: List[str] = Field(default_factory=list, description="Other symbols this symbol calls")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CodeFile(BaseModel):
    """Represents a code file with its metadata and symbols."""
    id: str = Field(..., description="Unique identifier for the file")
    path: str = Field(..., description="Relative path to the file")
    absolute_path: str = Field(..., description="Absolute path to the file")
    language: Language = Field(..., description="Programming language")
    size: int = Field(..., description="File size in bytes")
    lines_count: int = Field(..., description="Total number of lines")
    symbols: List[Symbol] = Field(default_factory=list, description="All symbols in the file")
    dependencies: List[Dependency] = Field(default_factory=list, description="File dependencies")
    hash: str = Field(..., description="File content hash for change detection")
    last_modified: datetime = Field(..., description="Last modification timestamp")
    last_indexed: datetime = Field(..., description="Last indexing timestamp")
    encoding: str = Field("utf-8", description="File encoding")
    imports: List[str] = Field(default_factory=list, description="List of imported modules")
    exports: List[str] = Field(default_factory=list, description="List of exported symbols")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional file metadata")


class SearchResult(BaseModel):
    """Represents a search result."""
    id: str = Field(..., description="Unique result identifier")
    type: str = Field(..., description="Type of result (symbol, file, etc.)")
    name: str = Field(..., description="Name or title of the result")
    file_path: str = Field(..., description="Path to the file")
    line_number: Optional[int] = Field(None, description="Line number if applicable")
    score: float = Field(..., description="Search relevance score")
    context: Optional[str] = Field(None, description="Surrounding code context")
    symbol: Optional[Symbol] = Field(None, description="Associated symbol if applicable")
    file: Optional[CodeFile] = Field(None, description="Associated file if applicable")


class IndexStats(BaseModel):
    """Statistics about the current index."""
    total_files: int = Field(0, description="Total number of indexed files")
    total_symbols: int = Field(0, description="Total number of indexed symbols")
    total_dependencies: int = Field(0, description="Total number of dependencies")
    languages: Dict[str, int] = Field(default_factory=dict, description="File count per language")
    symbol_types: Dict[str, int] = Field(default_factory=dict, description="Count per symbol type")
    index_size_mb: float = Field(0.0, description="Index size in megabytes")
    last_updated: datetime = Field(..., description="Last index update timestamp")


class SearchQuery(BaseModel):
    """Search query parameters."""
    query: str = Field(..., description="Search query string")
    type: Optional[str] = Field(None, description="Filter by result type")
    language: Optional[Language] = Field(None, description="Filter by programming language")
    symbol_type: Optional[SymbolType] = Field(None, description="Filter by symbol type")
    file_path: Optional[str] = Field(None, description="Filter by file path pattern")
    limit: int = Field(20, description="Maximum number of results", ge=1, le=100)
    include_context: bool = Field(True, description="Include code context in results")


class IndexRequest(BaseModel):
    """Request to index a repository or directory."""
    path: str = Field(..., description="Path to the directory to index")
    recursive: bool = Field(True, description="Whether to index subdirectories")
    include_patterns: List[str] = Field(
        default_factory=lambda: ["*.py", "*.js", "*.ts", "*.java", "*.go", "*.rs", "*.cpp", "*.c", "*.h"],
        description="File patterns to include"
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: ["node_modules", "__pycache__", ".git", "*.pyc", "*.so", "*.dll"],
        description="Patterns to exclude"
    )
    force_reindex: bool = Field(False, description="Force reindexing even if files haven't changed")


class AnalysisRequest(BaseModel):
    """Request for code analysis using Groq."""
    file_path: Optional[str] = Field(None, description="Specific file to analyze")
    symbol_name: Optional[str] = Field(None, description="Specific symbol to analyze")
    query: str = Field(..., description="Analysis question or task")
    include_context: bool = Field(True, description="Include surrounding code context")
    max_context_lines: int = Field(50, description="Maximum lines of context to include")


class AnalysisResult(BaseModel):
    """Result from code analysis."""
    analysis: str = Field(..., description="Analysis result from Groq")
    context: Optional[str] = Field(None, description="Code context that was analyzed")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    related_symbols: List[str] = Field(default_factory=list, description="Related symbol names")