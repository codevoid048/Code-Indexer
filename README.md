# Code Indexer - Advanced RAG with Code Parsing, Indexing and Searching using FAISS

Code analysis and search system that provides multi-language code parsing, fast text-based search using FAISS, real-time file monitoring, and AI-powered code analysis using Groq.

## ~> Features

### Core Capabilities
- **Multi-language Code Parsing**: Support for Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, and more using tree-sitter
- **Fast Text-based Search**: FAISS-powered search for symbols, files, and dependencies
- **Real-time Monitoring**: Automatic incremental updates when code files change
- **REST API**: Comprehensive API for all indexing and search operations
- **AI Analysis**: Groq integration for intelligent code analysis, documentation generation, and bug detection

### Search Types
- **Symbol Search**: Find functions, classes, methods, variables, and constants
- **File Search**: Locate files by name, path, or content
- **Dependency Search**: Find imports, requires, and package usage
- **Combined Search**: Intelligent multi-type search with relevance ranking
- **Regex Search**: Pattern-based searching across symbols and files

### Analysis Features (Groq AI)
- Code explanation and documentation generation
- Bug detection and security analysis
- Code complexity analysis and refactoring suggestions
- Performance optimization recommendations
- Best practices and code quality improvements

## ~> Local Setup

```bash
# Clone the repository
git clone <repository-url>
cd code-indexer

#create Virtual Environment
python -m venv .venv

## Activate venv

#Linux/Mac
source .venv/bin/activate

#Windows
.venv\Scripts\activate
```

## ~> Installation

### Using UV (Recommended)
```bash
# Install dependencies using UV
uv pip install -r requirements.txt
```

### Using pip

```bash
pip install -r requirements.txt
```

## ~> Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-api-key
CODE_INDEXER_INDEX_DIR=./index_data
CODE_INDEXER_CACHE_SIZE=1000
CODE_INDEXER_API_PORT=8000
DEBUG=false
```

### ~> Start the API server

```bash
# Start server
python main.py server

# Start with custom host/port
python main.py server --host 127.0.0.1 --port 8080 --reload
```

## ~> API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key API Endpoints

#### Index Management
- `POST /index` - Index a repository/directory
- `GET /index/stats` - Get indexing statistics
- `GET /index/status` - Check indexing progress
- `DELETE /index/file` - Remove file from index

#### Search Endpoints
- `POST /search` - General search with filters
- `GET /search/symbols` - Search symbols with filters
- `GET /search/files` - Search files
- `GET /search/dependencies` - Search dependencies
- `GET /search/regex` - Regex-based search

#### Symbol Information
- `GET /symbol/{name}` - Get symbol details
- `GET /symbol/{name}/references` - Find symbol references
- `GET /symbol/{name}/hierarchy` - Get symbol hierarchy
- `GET /file/symbols` - Get all symbols in a file

#### File Watching
- `POST /watch/start` - Start watching directory
- `POST /watch/stop` - Stop watching directory
- `GET /watch/status` - Get watcher status

#### AI Analysis (Groq)
- `POST /analyze` - General code analysis
- `POST /analyze/symbol/{name}/explain` - Explain symbol
- `POST /analyze/symbol/{name}/improve` - Suggest improvements
- `POST /analyze/file/bugs` - Find bugs in file
- `POST /analyze/file/complexity` - Analyze complexity

## ~> Architecture

### System Components

```
code-indexer/
├── models/           # Pydantic data models
├── core/            # Core processing modules
│   ├── parser.py        # Tree-sitter multi-language parser
│   ├── vectorstore.py   # FAISS-based text search
│   ├── search.py        # Advanced search engine
│   ├── incremental.py   # File watching and updates  
│   └── groq_analyzer.py # AI-powered code analysis
├── api/             # REST API endpoints
│   └── rest.py          # FastAPI application
├── data/            # Your code repository
├── index_data/      # FAISS indices and metadata
├── config.py        # Configuration management
└── main.py          # CLI and server entry point
```

### Data Flow

1. **Parsing**: Tree-sitter extracts symbols, dependencies, and metadata
2. **Indexing**: FAISS creates searchable text vectors for fast retrieval
3. **Storage**: JSON-based persistence with LRU caching
4. **Monitoring**: Watchdog monitors file changes for incremental updates
5. **Search**: Multi-strategy search with relevance ranking
6. **Analysis**: Groq AI provides intelligent code insights

### Supported Languages

- **Python**: Functions, classes, methods, variables, imports
- **JavaScript/TypeScript**: Functions, classes, modules, exports
- **Java**: Classes, methods, interfaces, packages
- **Go**: Functions, types, interfaces, packages
- **Rust**: Functions, structs, traits, modules
- **C/C++**: Functions, classes, structs, includes

## ~> Search Strategies

### Symbol Search
- **Exact matching**: Direct name matches
- **Fuzzy matching**: Similar names with typo tolerance
- **Type filtering**: Filter by function, class, variable, etc.
- **Scope awareness**: Understand parent-child relationships

### File Search
- **Path matching**: Find files by path components
- **Content analysis**: Search based on symbols within files
- **Language filtering**: Filter by programming language
- **Size and complexity hints**: Prefer moderately-sized files

### Dependency Search
- **Import tracking**: Find all import/require statements
- **External vs local**: Distinguish between external packages and local modules
- **Usage analysis**: Show where dependencies are used

## AI Analysis Features

### Code Explanation
- **Symbol explanation**: Understand what functions/classes do
- **Documentation generation**: Create comprehensive docstrings
- **Usage examples**: Generate example usage code

### Code Quality
- **Bug detection**: Find potential bugs and edge cases
- **Security analysis**: Identify security vulnerabilities  
- **Performance optimization**: Suggest performance improvements
- **Best practices**: Recommend coding standards and patterns

### Refactoring
- **Complexity analysis**: Measure and reduce code complexity
- **Design patterns**: Suggest appropriate design patterns
- **Code organization**: Recommend better code structure
- **Dependencies**: Optimize import/export structures