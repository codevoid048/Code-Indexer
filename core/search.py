import re
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

from core.vectorstore import TextBasedVectorStore
from core.parser import MultiLanguageParser
from models import SearchResult, Symbol, CodeFile, SearchQuery, SymbolType, Language


class SearchType(str, Enum):
    """Types of search operations."""
    SYMBOL = "symbol"
    FILE = "file"
    COMBINED = "combined"
    DEPENDENCY = "dependency"
    CONTENT = "content"


class SearchStrategy(str, Enum):
    """Search strategy types."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    REGEX = "regex"
    WILDCARD = "wildcard"


class CodeSearchEngine:
    """Advanced search engine for code with multiple search strategies."""
    
    def __init__(self, vectorstore: TextBasedVectorStore, parser: MultiLanguageParser):
        self.vectorstore = vectorstore
        self.parser = parser
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Main search method that routes to appropriate search strategy."""
        if not query.query.strip():
            return []
        
        # Determine search type if not specified
        search_type = self._determine_search_type(query)
        
        # Execute search based on type
        if search_type == SearchType.SYMBOL:
            return self.search_symbols(query)
        elif search_type == SearchType.FILE:
            return self.search_files(query)
        elif search_type == SearchType.DEPENDENCY:
            return self.search_dependencies(query)
        elif search_type == SearchType.CONTENT:
            return self.search_content(query)
        else:  # COMBINED
            return self.search_combined(query)
    
    def search_symbols(self, query: SearchQuery) -> List[SearchResult]:
        """Search for symbols with advanced filtering and ranking."""
        # Use vectorstore for base search
        results = self.vectorstore.search_symbols(
            query.query,
            limit=query.limit * 2,  # Get more to allow for filtering
            symbol_type=query.symbol_type.value if query.symbol_type else None
        )
        
        # Apply additional filters
        filtered_results = []
        for result in results:
            if self._matches_filters(result, query):
                # Enhance result with context if requested
                if query.include_context and result.symbol:
                    result.context = self._get_symbol_context(result.symbol)
                
                filtered_results.append(result)
        
        # Advanced ranking
        ranked_results = self._rank_symbol_results(filtered_results, query)
        
        return ranked_results[:query.limit]
    
    def search_files(self, query: SearchQuery) -> List[SearchResult]:
        """Search for files with path and content filtering."""
        # Use vectorstore for base search
        results = self.vectorstore.search_files(
            query.query,
            limit=query.limit * 2,
            language=query.language.value if query.language else None
        )
        
        # Apply additional filters
        filtered_results = []
        for result in results:
            if self._matches_filters(result, query):
                # Add file summary as context
                if query.include_context and result.file:
                    result.context = self._get_file_summary(result.file)
                
                filtered_results.append(result)
        
        # Advanced ranking for files
        ranked_results = self._rank_file_results(filtered_results, query)
        
        return ranked_results[:query.limit]
    
    def search_dependencies(self, query: SearchQuery) -> List[SearchResult]:
        """Search for dependencies and their usage."""
        results = []
        
        # Search through all files for dependency matches
        for code_file in self.vectorstore.file_metadata.values():
            for dep in code_file.dependencies:
                if self._matches_dependency_query(dep.name, query.query):
                    # Create search result for dependency
                    result = SearchResult(
                        id=f"dep_{code_file.id}_{dep.line_number}",
                        type="dependency",
                        name=dep.name,
                        file_path=code_file.path,
                        line_number=dep.line_number,
                        score=self._calculate_dependency_score(dep.name, query.query),
                        context=f"{dep.type}: {dep.name}" + (f" from {dep.source}" if dep.source else "")
                    )
                    results.append(result)
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:query.limit]
    
    def search_content(self, query: SearchQuery) -> List[SearchResult]:
        """Search within file content (not just metadata)."""
        results = []
        search_terms = query.query.lower().split()
        
        # This would ideally index file content, but for now we'll search through symbols
        symbol_results = self.search_symbols(query)
        file_results = self.search_files(query)
        
        # Combine and deduplicate
        all_results = symbol_results + file_results
        seen_ids = set()
        unique_results = []
        
        for result in all_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        return unique_results[:query.limit]
    
    def search_combined(self, query: SearchQuery) -> List[SearchResult]:
        """Combined search across all types with intelligent weighting."""
        # Get results from all search types
        symbol_results = self.search_symbols(query)
        file_results = self.search_files(query)
        dep_results = self.search_dependencies(query)
        
        # Apply different weights based on result type
        for result in symbol_results:
            result.score *= 1.2  # Symbols are usually more relevant
        
        for result in file_results:
            result.score *= 1.0  # Neutral weight
        
        for result in dep_results:
            result.score *= 0.8  # Dependencies are less likely to be what user wants
        
        # Combine all results
        all_results = symbol_results + file_results + dep_results
        
        # Remove duplicates and sort
        seen_ids = set()
        unique_results = []
        
        for result in all_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:query.limit]
    
    def find_symbol_references(self, symbol_name: str, file_path: Optional[str] = None) -> List[SearchResult]:
        """Find all references to a specific symbol."""
        results = []
        
        # Search through all symbols for references
        for symbol in self.vectorstore.symbol_metadata.values():
            # Check if this symbol calls the target symbol
            if symbol_name in symbol.calls:
                result = SearchResult(
                    id=f"ref_{symbol.id}",
                    type="reference",
                    name=f"Reference in {symbol.name}",
                    file_path=symbol.file_path,
                    line_number=symbol.line_number,
                    score=1.0,
                    symbol=symbol,
                    context=f"{symbol.name} calls {symbol_name}"
                )
                results.append(result)
        
        return results
    
    def get_symbol_hierarchy(self, symbol_name: str, file_path: Optional[str] = None) -> Dict[str, List[Symbol]]:
        """Get the hierarchy of a symbol (parent class, child methods, etc.)."""
        hierarchy = {
            'parents': [],
            'children': [],
            'siblings': []
        }
        
        # Find the target symbol
        target_symbol = self.vectorstore.get_symbol_by_name(symbol_name, file_path)
        if not target_symbol:
            return hierarchy
        
        # Find parent symbols
        if target_symbol.parent_symbol:
            for symbol in self.vectorstore.symbol_metadata.values():
                if symbol.id == target_symbol.parent_symbol:
                    hierarchy['parents'].append(symbol)
        
        # Find child symbols
        for symbol in self.vectorstore.symbol_metadata.values():
            if symbol.parent_symbol == target_symbol.id:
                hierarchy['children'].append(symbol)
        
        # Find sibling symbols (same parent)
        if target_symbol.parent_symbol:
            for symbol in self.vectorstore.symbol_metadata.values():
                if (symbol.parent_symbol == target_symbol.parent_symbol and 
                    symbol.id != target_symbol.id):
                    hierarchy['siblings'].append(symbol)
        
        return hierarchy
    
    def suggest_similar_symbols(self, symbol_name: str, limit: int = 10) -> List[SearchResult]:
        """Suggest symbols similar to the given name."""
        query = SearchQuery(
            query=symbol_name,
            limit=limit,
            include_context=False
        )
        
        results = self.search_symbols(query)
        
        # Filter out exact matches and enhance with similarity score
        similar_results = []
        for result in results:
            if result.symbol and result.symbol.name != symbol_name:
                # Calculate similarity score
                similarity = self._calculate_string_similarity(symbol_name, result.symbol.name)
                result.score = similarity
                similar_results.append(result)
        
        # Sort by similarity
        similar_results.sort(key=lambda x: x.score, reverse=True)
        return similar_results[:limit]
    
    def search_by_regex(self, pattern: str, search_type: SearchType = SearchType.SYMBOL) -> List[SearchResult]:
        """Search using regular expressions."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            results = []
            
            if search_type in [SearchType.SYMBOL, SearchType.COMBINED]:
                for symbol in self.vectorstore.symbol_metadata.values():
                    if regex.search(symbol.name) or (symbol.signature and regex.search(symbol.signature)):
                        result = SearchResult(
                            id=symbol.id,
                            type="symbol",
                            name=symbol.name,
                            file_path=symbol.file_path,
                            line_number=symbol.line_number,
                            score=1.0,
                            symbol=symbol
                        )
                        results.append(result)
            
            if search_type in [SearchType.FILE, SearchType.COMBINED]:
                for code_file in self.vectorstore.file_metadata.values():
                    if regex.search(code_file.path) or regex.search(code_file.path.split('/')[-1]):
                        result = SearchResult(
                            id=code_file.id,
                            type="file",
                            name=code_file.path.split('/')[-1],
                            file_path=code_file.path,
                            score=1.0,
                            file=code_file
                        )
                        results.append(result)
            
            return results
            
        except re.error as e:
            print(f"Invalid regex pattern: {e}")
            return []
    
    def _determine_search_type(self, query: SearchQuery) -> SearchType:
        """Determine the appropriate search type based on query."""
        query_text = query.query.lower()
        
        # Check for file-specific indicators
        if any(ext in query_text for ext in ['.py', '.js', '.ts', '.java', '.go', '.rs']):
            return SearchType.FILE
        
        if any(keyword in query_text for keyword in ['file', 'module', 'package']):
            return SearchType.FILE
        
        # Check for dependency-specific indicators
        if any(keyword in query_text for keyword in ['import', 'require', 'from', 'dependency']):
            return SearchType.DEPENDENCY
        
        # Check for symbol-specific indicators
        if any(keyword in query_text for keyword in ['function', 'class', 'method', 'variable']):
            return SearchType.SYMBOL
        
        # Default to combined search
        return SearchType.COMBINED
    
    def _matches_filters(self, result: SearchResult, query: SearchQuery) -> bool:
        """Check if a search result matches the query filters."""
        # Language filter
        if query.language:
            if result.symbol:
                # Get file for this symbol to check language
                code_file = None
                for cf in self.vectorstore.file_metadata.values():
                    if cf.path == result.file_path:
                        code_file = cf
                        break
                if not code_file or code_file.language != query.language:
                    return False
            elif result.file:
                if result.file.language != query.language:
                    return False
        
        # File path filter
        if query.file_path:
            if query.file_path not in result.file_path:
                return False
        
        return True
    
    def _matches_dependency_query(self, dep_name: str, query: str) -> bool:
        """Check if a dependency matches the search query."""
        query_lower = query.lower()
        dep_lower = dep_name.lower()
        
        # Exact match
        if query_lower == dep_lower:
            return True
        
        # Substring match
        if query_lower in dep_lower:
            return True
        
        # Fuzzy match (simple implementation)
        if self._calculate_string_similarity(query_lower, dep_lower) > 0.7:
            return True
        
        return False
    
    def _calculate_dependency_score(self, dep_name: str, query: str) -> float:
        """Calculate relevance score for a dependency."""
        if query.lower() == dep_name.lower():
            return 1.0
        
        if query.lower() in dep_name.lower():
            return 0.8
        
        return self._calculate_string_similarity(query.lower(), dep_name.lower())
    
    def _get_symbol_context(self, symbol: Symbol) -> Optional[str]:
        """Get context around a symbol (surrounding code)."""
        # This is a simplified version - in practice you'd read the actual file
        context_parts = []
        
        if symbol.signature:
            context_parts.append(f"Signature: {symbol.signature}")
        
        if symbol.docstring:
            context_parts.append(f"Docstring: {symbol.docstring[:100]}...")
        
        if symbol.parameters:
            context_parts.append(f"Parameters: {', '.join(symbol.parameters)}")
        
        if symbol.return_type:
            context_parts.append(f"Returns: {symbol.return_type}")
        
        return " | ".join(context_parts) if context_parts else None
    
    def _get_file_summary(self, code_file: CodeFile) -> str:
        """Get a summary of a code file."""
        summary_parts = [
            f"Language: {code_file.language.value}",
            f"Lines: {code_file.lines_count}",
            f"Symbols: {len(code_file.symbols)}"
        ]
        
        if code_file.imports:
            summary_parts.append(f"Imports: {len(code_file.imports)}")
        
        if code_file.dependencies:
            summary_parts.append(f"Dependencies: {len(code_file.dependencies)}")
        
        return " | ".join(summary_parts)
    
    def _rank_symbol_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Advanced ranking for symbol search results."""
        query_lower = query.query.lower()
        
        for result in results:
            if not result.symbol:
                continue
            
            # Base score from vectorstore
            base_score = result.score
            
            # Name similarity bonus
            name_similarity = self._calculate_string_similarity(query_lower, result.symbol.name.lower())
            
            # Exact match bonus
            if query_lower == result.symbol.name.lower():
                base_score *= 2.0
            elif query_lower in result.symbol.name.lower():
                base_score *= 1.5
            
            # Symbol type preference (functions/classes are usually more relevant)
            if result.symbol.type in [SymbolType.FUNCTION, SymbolType.CLASS]:
                base_score *= 1.2
            elif result.symbol.type == SymbolType.METHOD:
                base_score *= 1.1
            
            # Documentation bonus (well-documented code is usually more important)
            if result.symbol.docstring:
                base_score *= 1.1
            
            # Update final score
            result.score = base_score * name_similarity
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _rank_file_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Advanced ranking for file search results."""
        query_lower = query.query.lower()
        
        for result in results:
            if not result.file:
                continue
            
            base_score = result.score
            
            # Filename match bonus
            filename = result.file.path.split('/')[-1].lower()
            if query_lower in filename:
                base_score *= 1.5
            
            # Path depth penalty (prefer files in root or shallow directories)
            depth = len(result.file.path.split('/'))
            base_score *= max(0.5, 1.0 - (depth - 1) * 0.1)
            
            # Size bonus (moderate-sized files are usually more relevant)
            if 100 < result.file.lines_count < 1000:
                base_score *= 1.1
            
            result.score = base_score
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _calculate_string_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings using simple ratio."""
        if not s1 or not s2:
            return 0.0
        
        # Simple Levenshtein-based similarity
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        
        distance = levenshtein_distance(s1.lower(), s2.lower())
        return 1.0 - (distance / max_len)