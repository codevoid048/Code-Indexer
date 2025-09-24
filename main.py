import os
import sys
import argparse
import asyncio
from typing import List, Optional
import uvicorn
from api.rest import app
from models import AnalysisRequest

from core.parser import MultiLanguageParser
from core.vectorstore import TextBasedVectorStore
from core.search import CodeSearchEngine
from core.incremental import IncrementalIndexer
from core.groq_analyzer import GroqCodeAnalyzer
from models import SearchQuery


class CodeIndexerCLI:
    """Command-line interface for the code indexer."""
    
    def __init__(self):
        self.parser = MultiLanguageParser()
        self.vectorstore = TextBasedVectorStore()
        self.search_engine = CodeSearchEngine(self.vectorstore, self.parser)
        self.incremental_indexer = IncrementalIndexer(self.parser, self.vectorstore)
        self.groq_analyzer = GroqCodeAnalyzer()
        
        print("Code Indexer initialized successfully!")
    
    def index_directory(self, directory: str, recursive: bool = True, force: bool = False) -> bool:
        """Index a directory."""
        try:
            if not os.path.isdir(directory):
                print(f"Error: Directory not found: {directory}")
                return False
            
            print(f"Indexing directory: {directory}")
            print(f"Recursive: {recursive}, Force reindex: {force}")
            
            # Count files first
            file_count = 0
            for root, dirs, files in os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]:
                for file in files:
                    file_path = os.path.join(root, file)
                    if self.incremental_indexer._should_process_file(file_path):
                        file_count += 1
            
            print(f"Found {file_count} files to process")
            
            # Process files
            processed = 0
            errors = 0
            
            for root, dirs, files in os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]:
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    if self.incremental_indexer._should_process_file(file_path):
                        try:
                            # Remove existing if force reindex
                            if force:
                                self.vectorstore.remove_file(file_path)
                            
                            # Parse and index file
                            code_file = self.parser.parse_file(file_path)
                            if code_file:
                                success = self.vectorstore.add_file(code_file)
                                if success:
                                    processed += 1
                                    if processed % 10 == 0:
                                        print(f"Processed {processed}/{file_count} files...")
                                else:
                                    errors += 1
                                    print(f"Failed to index: {file_path}")
                            else:
                                errors += 1
                                print(f"Failed to parse: {file_path}")
                        
                        except Exception as e:
                            errors += 1
                            print(f"Error processing {file_path}: {e}")
            
            print(f"Indexing complete: {processed} files indexed, {errors} errors")
            
            # Save indices
            self.vectorstore.save_indices()
            print("Index saved to disk")
            
            # Start watching directory if requested
            if recursive:
                self.incremental_indexer.start_watching(directory, recursive)
                print(f"Started watching directory: {directory}")
            
            return True
            
        except Exception as e:
            print(f"Error during indexing: {e}")
            return False
    
    def search(self, query: str, search_type: str = "combined", limit: int = 10) -> List:
        """Search the indexed code."""
        try:
            search_query = SearchQuery(
                query=query,
                limit=limit,
                include_context=True
            )
            
            if search_type == "symbol":
                results = self.search_engine.search_symbols(search_query)
            elif search_type == "file":
                results = self.search_engine.search_files(search_query)
            elif search_type == "dependency":
                results = self.search_engine.search_dependencies(search_query)
            else:  # combined
                results = self.search_engine.search_combined(search_query)
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def print_search_results(self, results: List, query: str):
        """Print search results in a formatted way."""
        if not results:
            print(f"No results found for query: '{query}'")
            return
        
        print(f"\nFound {len(results)} results for query: '{query}'\n")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.type.upper()}: {result.name}")
            print(f"   File: {result.file_path}")
            if result.line_number:
                print(f"   Line: {result.line_number}")
            print(f"   Score: {result.score:.3f}")
            
            if result.context:
                print(f"   Context: {result.context}")
            
            print("-" * 40)
    
    def get_stats(self):
        """Print index statistics."""
        try:
            stats = self.vectorstore.get_stats()
            
            print(f"\nIndex Statistics:")
            print(f"Total files: {stats.total_files}")
            print(f"Total symbols: {stats.total_symbols}")
            print(f"Total dependencies: {stats.total_dependencies}")
            print(f"Index size: {stats.index_size_mb:.2f} MB")
            print(f"Last updated: {stats.last_updated}")
            
            print(f"\nLanguages:")
            for lang, count in stats.languages.items():
                print(f"  {lang}: {count} files")
            
            print(f"\nSymbol types:")
            for sym_type, count in stats.symbol_types.items():
                print(f"  {sym_type}: {count} symbols")
            
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    async def analyze_with_groq(self, query: str, symbol_name: str = None, file_path: str = None):
        """Analyze code using Groq AI."""
        if not self.groq_analyzer.is_available():
            print("Groq API not configured. Please set GROQ_API_KEY environment variable.")
            return
        
        try:            
            request = AnalysisRequest(
                query=query,
                symbol_name=symbol_name,
                file_path=file_path,
                include_context=True
            )
            
            result = await self.groq_analyzer.analyze_code(request, self.vectorstore)
            
            print(f"\nGroq Analysis Result:")
            print("-" * 50)
            print(result.analysis)
            
            if result.suggestions:
                print(f"\nSuggestions:")
                for i, suggestion in enumerate(result.suggestions, 1):
                    print(f"{i}. {suggestion}")
            
            if result.related_symbols:
                print(f"\nRelated symbols: {', '.join(result.related_symbols)}")
            
        except Exception as e:
            print(f"Error during Groq analysis: {e}")


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Code Indexer - Advanced RAG with Code Parsing, Indexing and Searching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Index current directory
            python main.py index .
            
            # Index with force reindexing
            python main.py index . --force
            
            # Search for symbols
            python main.py search "function_name"
            
            # Search only symbols
            python main.py search "MyClass" --type symbol
            
            # Start API server
            python main.py server
            
            # Show index statistics
            python main.py stats
            
            # Analyze code with AI
            python main.py analyze "explain this function" --symbol my_function
                    """
                )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index a directory')
    index_parser.add_argument('directory', help='Directory to index')
    index_parser.add_argument('--no-recursive', action='store_true', help='Don\'t index subdirectories')
    index_parser.add_argument('--force', action='store_true', help='Force reindexing of all files')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search indexed code')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--type', choices=['symbol', 'file', 'dependency', 'combined'], 
                              default='combined', help='Type of search')
    search_parser.add_argument('--limit', type=int, default=10, help='Maximum results')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='127.0.0.1', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    server_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Stats command
    subparsers.add_parser('stats', help='Show index statistics')
    
    # Analyze command (Groq integration)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze code with AI')
    analyze_parser.add_argument('query', help='Analysis query')
    analyze_parser.add_argument('--symbol', help='Specific symbol to analyze')
    analyze_parser.add_argument('--file', help='Specific file to analyze')
    
    return parser


def run_server(host: str = '127.0.0.1', port: int = 8000, reload: bool = False):
    """Run the API server synchronously."""
    try:
        print(f"Starting Code Indexer API server on {host}:{port}")
        print(f"API documentation available at: http://{host}:{port}/docs")

        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload
        )
    except ImportError:
        print("Error: uvicorn not installed. Please install with: uv pip install uvicorn")
        sys.exit(1)


async def main():
    """Main application entry point."""
    parser = create_arg_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    cli = CodeIndexerCLI()

    if args.command == 'index':
        success = cli.index_directory(
            args.directory,
            recursive=not args.no_recursive,
            force=args.force
        )
        sys.exit(0 if success else 1)

    elif args.command == 'search':
        results = cli.search(args.query, args.type, args.limit)
        cli.print_search_results(results, args.query)

    elif args.command == 'stats':
        cli.get_stats()

    elif args.command == 'analyze':
        await cli.analyze_with_groq(
            args.query,
            symbol_name=args.symbol,
            file_path=args.file
        )

    else:
        parser.print_help()
if __name__ == "__main__":
    try:
        parser = create_arg_parser()
        args = parser.parse_args()

        # Handle server command synchronously before async context
        if args.command == 'server':
            run_server(args.host, args.port, args.reload)
            sys.exit(0)

        # For all other commands, use async main
        try:
            loop = asyncio.get_running_loop()
            print("Warning: Event loop already running, using existing loop")
            asyncio.create_task(main())
        except RuntimeError:
            asyncio.run(main())
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
