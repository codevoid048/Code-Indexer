import os
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import tree_sitter_languages
from tree_sitter import Parser

from models import CodeFile, Symbol, Dependency, SymbolType, Language as Lang


class MultiLanguageParser:
    """Multi-language AST parser using tree-sitter."""
    
    def __init__(self):
        self.parsers: Dict[str, Parser] = {}
        self.languages: Dict[str, Lang] = {}
        self._setup_languages()
    
    def _setup_languages(self):
        """Initialize tree-sitter languages."""
        try:
            # Use tree-sitter-languages for easy language setup
            language_configs = {
                'python': 'python',
                'javascript': 'javascript', 
                'typescript': 'typescript',
                'java': 'java',
                'go': 'go',
                'rust': 'rust',
                'cpp': 'cpp',
            }
            
            for lang_name, ts_lang_name in language_configs.items():
                try:
                    # Get parser directly from tree-sitter-languages
                    parser = tree_sitter_languages.get_parser(ts_lang_name)
                    self.parsers[lang_name] = parser
                    print(f"✓ Loaded {lang_name} parser")
                except Exception as e:
                    print(f"Warning: Could not load {lang_name} parser: {e}")
                    # Create a fallback parser for regex-based parsing
                    self.parsers[lang_name] = None
                    
        except Exception as e:
            print(f"Warning: Tree-sitter setup incomplete: {e}")
            # Fallback: all parsers will be None, using regex-based parsing
    
    def detect_language(self, file_path: str) -> Lang:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': Lang.PYTHON,
            '.js': Lang.JAVASCRIPT,
            '.mjs': Lang.JAVASCRIPT,
            '.ts': Lang.TYPESCRIPT,
            '.tsx': Lang.TYPESCRIPT,
            '.java': Lang.JAVA,
            '.go': Lang.GO,
            '.rs': Lang.RUST,
            '.cpp': Lang.CPP,
            '.cxx': Lang.CPP,
            '.cc': Lang.CPP,
            '.c': Lang.C,
            '.h': Lang.C,
            '.hpp': Lang.CPP,
        }
        
        return language_map.get(ext, Lang.UNKNOWN)
    
    def parse_file(self, file_path: str) -> Optional[CodeFile]:
        """Parse a code file and extract metadata and symbols."""
        try:
            if not os.path.exists(file_path):
                return None
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Get file metadata
            stat = os.stat(file_path)
            file_hash = hashlib.md5(content.encode()).hexdigest()
            language = self.detect_language(file_path)
            
            # Create CodeFile object
            code_file = CodeFile(
                id=self._generate_id(file_path),
                path=os.path.relpath(file_path),
                absolute_path=os.path.abspath(file_path),
                language=language,
                size=stat.st_size,
                lines_count=len(content.split('\n')),
                hash=file_hash,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                last_indexed=datetime.now().isoformat(),
            )
            
            # Parse symbols based on language
            if language == Lang.PYTHON:
                self._parse_python(content, code_file)
            elif language in [Lang.JAVASCRIPT, Lang.TYPESCRIPT]:
                self._parse_javascript(content, code_file)
            elif language == Lang.JAVA:
                self._parse_java(content, code_file)
            elif language == Lang.CPP:
                self._parse_cpp(content, code_file)
            else:
                # Fallback to regex-based parsing
                self._parse_generic(content, code_file)
            
            return code_file
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return None
    
    def _parse_python(self, content: str, code_file: CodeFile):
        """Parse Python-specific symbols using regex patterns."""
        lines = content.split('\n')
        
        # Parse imports
        for i, line in enumerate(lines, 1):
            # Standard imports: import module, from module import item
            import_match = re.match(r'^\s*(from\s+(\S+)\s+)?import\s+(.+)$', line.strip())
            if import_match:
                from_module = import_match.group(2)
                imports = [item.strip() for item in import_match.group(3).split(',')]
                
                for imp in imports:
                    # Clean up 'as' aliases
                    clean_imp = imp.split(' as ')[0].strip()
                    code_file.dependencies.append(Dependency(
                        name=clean_imp,
                        type='import',
                        source=from_module,
                        is_external=not self._is_local_import(clean_imp, from_module),
                        line_number=i
                    ))
                    code_file.imports.append(clean_imp)
        
        # Parse functions
        for i, line in enumerate(lines, 1):
            func_match = re.match(r'^\s*(async\s+)?def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.+?))?\s*:', line)
            if func_match:
                is_async = bool(func_match.group(1))
                func_name = func_match.group(2)
                params = func_match.group(3)
                return_type = func_match.group(4)
                
                # Find docstring
                docstring = self._extract_docstring(lines, i)
                
                symbol = Symbol(
                    id=self._generate_symbol_id(code_file.path, func_name, i),
                    name=func_name,
                    type=SymbolType.FUNCTION,
                    file_path=code_file.path,
                    line_number=i,
                    signature=line.strip(),
                    docstring=docstring,
                    parameters=self._parse_parameters(params),
                    return_type=return_type.strip() if return_type else None,
                    is_async=is_async
                )
                code_file.symbols.append(symbol)
        
        # Parse classes
        for i, line in enumerate(lines, 1):
            class_match = re.match(r'^\s*class\s+(\w+)(?:\(([^)]*)\))?\s*:', line)
            if class_match:
                class_name = class_match.group(1)
                inheritance = class_match.group(2)
                
                # Find docstring
                docstring = self._extract_docstring(lines, i)
                
                symbol = Symbol(
                    id=self._generate_symbol_id(code_file.path, class_name, i),
                    name=class_name,
                    type=SymbolType.CLASS,
                    file_path=code_file.path,
                    line_number=i,
                    signature=line.strip(),
                    docstring=docstring,
                    metadata={'inheritance': inheritance.strip() if inheritance else None}
                )
                code_file.symbols.append(symbol)
                
                # Parse class methods
                self._parse_class_methods(lines, i, class_name, code_file)
        
        # Parse variables and constants
        for i, line in enumerate(lines, 1):
            # Constants (ALL_CAPS)
            const_match = re.match(r'^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.+)$', line.strip())
            if const_match:
                const_name = const_match.group(1)
                symbol = Symbol(
                    id=self._generate_symbol_id(code_file.path, const_name, i),
                    name=const_name,
                    type=SymbolType.CONSTANT,
                    file_path=code_file.path,
                    line_number=i,
                    signature=line.strip()
                )
                code_file.symbols.append(symbol)
            
            # Regular variables (simple assignment)
            elif re.match(r'^\s*[a-z_][a-z0-9_]*\s*=\s*', line.strip()) and not line.strip().startswith('#'):
                var_match = re.match(r'^\s*([a-z_][a-z0-9_]*)\s*=', line.strip())
                if var_match:
                    var_name = var_match.group(1)
                    symbol = Symbol(
                        id=self._generate_symbol_id(code_file.path, var_name, i),
                        name=var_name,
                        type=SymbolType.VARIABLE,
                        file_path=code_file.path,
                        line_number=i,
                        signature=line.strip()
                    )
                    code_file.symbols.append(symbol)
    
    def _parse_javascript(self, content: str, code_file: CodeFile):
        """Parse JavaScript/TypeScript symbols using regex patterns."""
        lines = content.split('\n')
        
        # Parse imports
        for i, line in enumerate(lines, 1):
            # ES6 imports
            import_match = re.match(r'^\s*import\s+(.+?)\s+from\s+[\'"](.+?)[\'"]', line.strip())
            if import_match:
                imports = import_match.group(1)
                module = import_match.group(2)
                code_file.dependencies.append(Dependency(
                    name=imports.strip(),
                    type='import',
                    source=module,
                    is_external=not module.startswith('.'),
                    line_number=i
                ))
            
            # Require statements
            require_match = re.match(r'^\s*(?:const|let|var)\s+(.+?)\s*=\s*require\s*\(\s*[\'"](.+?)[\'"]\s*\)', line.strip())
            if require_match:
                var_name = require_match.group(1)
                module = require_match.group(2)
                code_file.dependencies.append(Dependency(
                    name=var_name.strip(),
                    type='require',
                    source=module,
                    is_external=not module.startswith('.'),
                    line_number=i
                ))
        
        # Parse functions
        for i, line in enumerate(lines, 1):
            # Function declarations
            func_match = re.match(r'^\s*(async\s+)?function\s+(\w+)\s*\((.*?)\)', line.strip())
            if func_match:
                is_async = bool(func_match.group(1))
                func_name = func_match.group(2)
                params = func_match.group(3)
                
                symbol = Symbol(
                    id=self._generate_symbol_id(code_file.path, func_name, i),
                    name=func_name,
                    type=SymbolType.FUNCTION,
                    file_path=code_file.path,
                    line_number=i,
                    signature=line.strip(),
                    parameters=self._parse_parameters(params),
                    is_async=is_async
                )
                code_file.symbols.append(symbol)
            
            # Arrow functions
            arrow_match = re.match(r'^\s*(?:const|let|var)\s+(\w+)\s*=\s*(async\s+)?\(([^)]*)\)\s*=>', line.strip())
            if arrow_match:
                func_name = arrow_match.group(1)
                is_async = bool(arrow_match.group(2))
                params = arrow_match.group(3)
                
                symbol = Symbol(
                    id=self._generate_symbol_id(code_file.path, func_name, i),
                    name=func_name,
                    type=SymbolType.FUNCTION,
                    file_path=code_file.path,
                    line_number=i,
                    signature=line.strip(),
                    parameters=self._parse_parameters(params),
                    is_async=is_async
                )
                code_file.symbols.append(symbol)
        
        # Parse classes
        for i, line in enumerate(lines, 1):
            class_match = re.match(r'^\s*(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{?', line.strip())
            if class_match:
                class_name = class_match.group(1)
                extends = class_match.group(2)
                
                symbol = Symbol(
                    id=self._generate_symbol_id(code_file.path, class_name, i),
                    name=class_name,
                    type=SymbolType.CLASS,
                    file_path=code_file.path,
                    line_number=i,
                    signature=line.strip(),
                    metadata={'extends': extends if extends else None}
                )
                code_file.symbols.append(symbol)
    
    def _parse_java(self, content: str, code_file: CodeFile):
        """Parse Java symbols using regex patterns."""
        lines = content.split('\n')
        
        # Parse imports
        for i, line in enumerate(lines, 1):
            import_match = re.match(r'^\s*import\s+(?:static\s+)?(.+?)\s*;', line.strip())
            if import_match:
                import_name = import_match.group(1)
                code_file.dependencies.append(Dependency(
                    name=import_name.split('.')[-1],
                    type='import',
                    source=import_name,
                    is_external=not import_name.startswith('com.yourcompany'),
                    line_number=i
                ))
        
        # Parse classes and interfaces
        for i, line in enumerate(lines, 1):
            class_match = re.match(r'^\s*(?:public|private|protected)?\s*(class|interface|enum)\s+(\w+)', line.strip())
            if class_match:
                type_kind = class_match.group(1)
                name = class_match.group(2)
                
                symbol_type = SymbolType.CLASS
                if type_kind == 'interface':
                    symbol_type = SymbolType.INTERFACE
                elif type_kind == 'enum':
                    symbol_type = SymbolType.ENUM
                
                symbol = Symbol(
                    id=self._generate_symbol_id(code_file.path, name, i),
                    name=name,
                    type=symbol_type,
                    file_path=code_file.path,
                    line_number=i,
                    signature=line.strip()
                )
                code_file.symbols.append(symbol)
        
        # Parse methods
        for i, line in enumerate(lines, 1):
            method_match = re.match(r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(\w+)\s+(\w+)\s*\((.*?)\)', line.strip())
            if method_match and not line.strip().startswith('//'):
                return_type = method_match.group(1)
                method_name = method_match.group(2)
                params = method_match.group(3)
                
                symbol = Symbol(
                    id=self._generate_symbol_id(code_file.path, method_name, i),
                    name=method_name,
                    type=SymbolType.METHOD,
                    file_path=code_file.path,
                    line_number=i,
                    signature=line.strip(),
                    parameters=self._parse_parameters(params),
                    return_type=return_type
                )
                code_file.symbols.append(symbol)
    
    def _parse_cpp(self, content: str, code_file: CodeFile):
        """Parse C++ symbols using tree-sitter."""
        try:
            parser = self.parsers.get('cpp')
            if not parser:
                print("C++ parser not available, falling back to generic parsing")
                self._parse_generic(content, code_file)
                return
            
            tree = parser.parse(content.encode('utf-8'))
            root = tree.root_node
            
            def traverse_tree(node, parent_context=None):
                """Recursively traverse the AST to find symbols."""
                context = parent_context or {}
                
                # Only process function definitions at the top level or in namespaces/classes
                if node.type == 'function_definition':
                    # Make sure this is not inside a control structure
                    if not self._is_inside_control_structure(node):
                        # Extract function name and signature
                        declarator = None
                        return_type = None
                        
                        for child in node.children:
                            if child.type in ['primitive_type', 'type_identifier', 'template_type']:
                                return_type = content[child.start_byte:child.end_byte].strip()
                            elif child.type == 'function_declarator':
                                declarator = child
                                break
                        
                        if declarator:
                            # Find function name
                            func_name = None
                            parameters = []
                            
                            for child in declarator.children:
                                if child.type == 'identifier':
                                    func_name = content[child.start_byte:child.end_byte]
                                elif child.type == 'parameter_list':
                                    # Extract parameters
                                    for param in child.children:
                                        if param.type == 'parameter_declaration':
                                            param_text = content[param.start_byte:param.end_byte].strip()
                                            if param_text and not param_text.startswith(',') and param_text != '(' and param_text != ')':
                                                parameters.append(param_text)
                            
                            if func_name and func_name not in ['if', 'for', 'while', 'switch', 'case', 'default']:
                                # Get line number
                                start_line = content[:node.start_byte].count('\n') + 1
                                
                                symbol = Symbol(
                                    id=self._generate_symbol_id(code_file.path, func_name, start_line),
                                    name=func_name,
                                    type=SymbolType.FUNCTION,
                                    file_path=code_file.path,
                                    line_number=start_line,
                                    signature=content[node.start_byte:node.end_byte].strip(),
                                    parameters=parameters,
                                    return_type=return_type
                                )
                                code_file.symbols.append(symbol)
                
                elif node.type in ['class_specifier', 'struct_specifier']:
                    # Extract class/struct name
                    class_name = None
                    for child in node.children:
                        if child.type == 'type_identifier':
                            class_name = content[child.start_byte:child.end_byte]
                            break
                    
                    if class_name:
                        start_line = content[:node.start_byte].count('\n') + 1
                        symbol_type = SymbolType.CLASS if node.type == 'class_specifier' else SymbolType.STRUCT
                        
                        symbol = Symbol(
                            id=self._generate_symbol_id(code_file.path, class_name, start_line),
                            name=class_name,
                            type=symbol_type,
                            file_path=code_file.path,
                            line_number=start_line,
                            signature=content[node.start_byte:node.end_byte].strip()
                        )
                        code_file.symbols.append(symbol)
                
                # Recursively traverse children
                for child in node.children:
                    traverse_tree(child, context)
            
            def _is_inside_control_structure(self, node):
                """Check if a node is inside a control structure."""
                current = node.parent
                while current:
                    if current.type in ['if_statement', 'for_statement', 'while_statement', 
                                      'switch_statement', 'compound_statement']:
                        return True
                    current = current.parent
                return False
            
            traverse_tree(root)
            
        except Exception as e:
            print(f"Error parsing C++ with tree-sitter: {e}, falling back to generic parsing")
            self._parse_generic(content, code_file)
    
    def _is_inside_control_structure(self, node):
        """Check if a node is inside a control structure."""
        current = node.parent
        while current:
            if current.type in ['if_statement', 'for_statement', 'while_statement', 
                              'switch_statement', 'compound_statement']:
                return True
            current = current.parent
        return False
    
    def _parse_generic(self, content: str, code_file: CodeFile):
        """Generic parsing for unsupported languages using simple patterns."""
        lines = content.split('\n')
        
        # Look for function-like patterns
        for i, line in enumerate(lines, 1):
            # Generic function pattern: word followed by parentheses
            func_match = re.match(r'^\s*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{?', line.strip())
            if func_match and not line.strip().startswith('//') and not line.strip().startswith('#'):
                func_name = func_match.group(1)
                
                symbol = Symbol(
                    id=self._generate_symbol_id(code_file.path, func_name, i),
                    name=func_name,
                    type=SymbolType.FUNCTION,
                    file_path=code_file.path,
                    line_number=i,
                    signature=line.strip()
                )
                code_file.symbols.append(symbol)
    
    def _parse_class_methods(self, lines: List[str], class_start: int, class_name: str, code_file: CodeFile):
        """Parse methods within a Python class."""
        indent_level = len(lines[class_start - 1]) - len(lines[class_start - 1].lstrip())
        
        for i in range(class_start, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and line.strip():
                break  # End of class
            
            method_match = re.match(r'^\s*(async\s+)?def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*(.+?))?\s*:', line)
            if method_match:
                is_async = bool(method_match.group(1))
                method_name = method_match.group(2)
                params = method_match.group(3)
                return_type = method_match.group(4)
                
                # Find docstring
                docstring = self._extract_docstring(lines, i + 1)
                
                symbol = Symbol(
                    id=self._generate_symbol_id(code_file.path, f"{class_name}.{method_name}", i + 1),
                    name=method_name,
                    type=SymbolType.METHOD,
                    file_path=code_file.path,
                    line_number=i + 1,
                    signature=line.strip(),
                    docstring=docstring,
                    parent_symbol=self._generate_symbol_id(code_file.path, class_name, class_start),
                    parameters=self._parse_parameters(params),
                    return_type=return_type.strip() if return_type else None,
                    is_async=is_async
                )
                code_file.symbols.append(symbol)
    
    def _extract_docstring(self, lines: List[str], start_line: int) -> Optional[str]:
        """Extract docstring starting from the given line."""
        if start_line >= len(lines):
            return None
        
        line = lines[start_line].strip()
        if line.startswith('"""') or line.startswith("'''"):
            quote_type = line[:3]
            if line.endswith(quote_type) and len(line) > 6:
                # Single-line docstring
                return line[3:-3].strip()
            else:
                # Multi-line docstring
                docstring = [line[3:]]
                for i in range(start_line + 1, len(lines)):
                    line = lines[i].strip()
                    if line.endswith(quote_type):
                        docstring.append(line[:-3])
                        break
                    docstring.append(line)
                return '\n'.join(docstring).strip()
        
        return None
    
    def _parse_parameters(self, params_str: str) -> List[str]:
        """Parse function parameters from string."""
        if not params_str.strip():
            return []
        
        # Simple comma split (doesn't handle complex cases)
        params = [p.strip() for p in params_str.split(',')]
        return [p for p in params if p and p != 'self']
    
    def _is_local_import(self, import_name: str, from_module: Optional[str]) -> bool:
        """Check if an import is local to the project."""
        if from_module and from_module.startswith('.'):
            return True
        
        # Basic heuristic: if it's not a common standard library or well-known package
        standard_libs = {
            'os', 'sys', 'json', 'datetime', 'time', 'random', 're', 'math',
            'collections', 'itertools', 'functools', 'pathlib', 'typing'
        }
        
        common_packages = {
            'numpy', 'pandas', 'requests', 'flask', 'django', 'fastapi',
            'pydantic', 'sqlalchemy', 'pytest', 'click', 'jinja2'
        }
        
        return import_name not in standard_libs and import_name not in common_packages
    
    def _generate_id(self, file_path: str) -> str:
        """Generate unique ID for a file."""
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def _generate_symbol_id(self, file_path: str, symbol_name: str, line_number: int) -> str:
        """Generate unique ID for a symbol."""
        identifier = f"{file_path}:{symbol_name}:{line_number}"
        return hashlib.md5(identifier.encode()).hexdigest()