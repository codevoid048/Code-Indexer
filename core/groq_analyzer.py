import os
from typing import List, Dict, Optional, Any
from groq import Groq
from config import Config
from models import AnalysisRequest, AnalysisResult, CodeFile, Symbol
from core.vectorstore import TextBasedVectorStore

config = Config()

class GroqCodeAnalyzer:
    """Code analysis using Groq API."""
    
    def __init__(self, api_key: str = None, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key or config.groq_api_key
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            print("Warning: GROQ_API_KEY not found. Code analysis features will be disabled.")
    
    def is_available(self) -> bool:
        """Check if Groq API is available."""
        return self.client is not None
    
    async def analyze_code(self, request: AnalysisRequest, vectorstore: TextBasedVectorStore) -> AnalysisResult:
        """Analyze code using Groq API."""
        if not self.is_available():
            return AnalysisResult(
                analysis="Groq API not available. Please set GROQ_API_KEY environment variable.",
                suggestions=["Configure Groq API key to enable code analysis"]
            )
        
        try:
            context = self._build_context(request, vectorstore)
            prompt = self._build_prompt(request, context)
            
            # Make API call to Groq
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=5000,
                temperature=0.1  # Lower temperature for more focused analysis
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse response to extract analysis and suggestions
            analysis, suggestions = self._parse_response(analysis_text)
            
            # Find related symbols
            related_symbols = self._find_related_symbols(request, vectorstore, analysis_text)
            
            return AnalysisResult(
                analysis=analysis,
                context=context,
                suggestions=suggestions,
                related_symbols=related_symbols
            )
            
        except Exception as e:
            return AnalysisResult(
                analysis=f"Error during code analysis: {str(e)}",
                suggestions=["Check your Groq API key and network connection"]
            )
    
    async def explain_symbol(self, symbol: Symbol, vectorstore: TextBasedVectorStore) -> AnalysisResult:
        """Explain what a specific symbol does."""
        request = AnalysisRequest(
            symbol_name=symbol.name,
            query=f"Explain what the {symbol.type.value} '{symbol.name}' does",
            include_context=True
        )
        
        return await self.analyze_code(request, vectorstore)
    
    async def suggest_improvements(self, symbol: Symbol, vectorstore: TextBasedVectorStore) -> AnalysisResult:
        """Suggest improvements for a symbol."""
        request = AnalysisRequest(
            symbol_name=symbol.name,
            query=f"Suggest improvements for the {symbol.type.value} '{symbol.name}'",
            include_context=True
        )
        
        return await self.analyze_code(request, vectorstore)
    
    async def find_bugs(self, file_path: str, vectorstore: TextBasedVectorStore) -> AnalysisResult:
        """Analyze code for potential bugs."""
        request = AnalysisRequest(
            file_path=file_path,
            query="Analyze this code for potential bugs, security issues, or logical errors",
            include_context=True,
            max_context_lines=100
        )
        
        return await self.analyze_code(request, vectorstore)
    
    async def generate_documentation(self, symbol: Symbol, vectorstore: TextBasedVectorStore) -> AnalysisResult:
        """Generate documentation for a symbol."""
        request = AnalysisRequest(
            symbol_name=symbol.name,
            query=f"Generate comprehensive documentation for the {symbol.type.value} '{symbol.name}'",
            include_context=True
        )
        
        return await self.analyze_code(request, vectorstore)
    
    async def analyze_complexity(self, file_path: str, vectorstore: TextBasedVectorStore) -> AnalysisResult:
        """Analyze code complexity and suggest refactoring."""
        request = AnalysisRequest(
            file_path=file_path,
            query="Analyze the complexity of this code and suggest refactoring opportunities",
            include_context=True
        )
        
        return await self.analyze_code(request, vectorstore)
    
    def _build_context(self, request: AnalysisRequest, vectorstore: TextBasedVectorStore) -> Optional[str]:
        """Build context for the analysis request."""
        if not request.include_context:
            return None
        
        context_parts = []
        
        # Get specific symbol context
        if request.symbol_name:
            symbol = vectorstore.get_symbol_by_name(request.symbol_name, request.file_path)
            if symbol:
                context_parts.append(f"Symbol: {symbol.name}")
                context_parts.append(f"Type: {symbol.type.value}")
                context_parts.append(f"File: {symbol.file_path}")
                context_parts.append(f"Line: {symbol.line_number}")
                
                if symbol.signature:
                    context_parts.append(f"Signature: {symbol.signature}")
                
                if symbol.docstring:
                    context_parts.append(f"Current docstring: {symbol.docstring}")
                
                if symbol.parameters:
                    context_parts.append(f"Parameters: {', '.join(symbol.parameters)}")
                
                if symbol.return_type:
                    context_parts.append(f"Return type: {symbol.return_type}")
        
        # Get file context
        if request.file_path:
            file_symbols = vectorstore.get_file_symbols(request.file_path)
            if file_symbols:
                context_parts.append(f"\nFile symbols ({len(file_symbols)}):")
                for sym in file_symbols[:10]:  # Limit to first 10 symbols
                    context_parts.append(f"  - {sym.type.value}: {sym.name} (line {sym.line_number})")
                
                if len(file_symbols) > 10:
                    context_parts.append(f"  ... and {len(file_symbols) - 10} more symbols")
        
        return "\n".join(context_parts) if context_parts else None
    
    def _build_prompt(self, request: AnalysisRequest, context: Optional[str]) -> str:
        """Build the prompt for Groq API."""
        prompt_parts = []
        
        # Add the main query
        prompt_parts.append(f"Query: {request.query}")
        
        # Add context if available
        if context:
            prompt_parts.append(f"\nContext:\n{context}")
        
        # Add specific instructions based on query type
        if "explain" in request.query.lower():
            prompt_parts.append("\nPlease provide a clear, detailed explanation suitable for developers.")
        
        elif "improve" in request.query.lower() or "suggest" in request.query.lower():
            prompt_parts.append("\nPlease provide specific, actionable improvement suggestions.")
        
        elif "bug" in request.query.lower() or "error" in request.query.lower():
            prompt_parts.append("\nFocus on potential bugs, edge cases, and security vulnerabilities.")
        
        elif "document" in request.query.lower():
            prompt_parts.append("\nGenerate proper documentation including purpose, parameters, return values, and examples.")
        
        elif "complex" in request.query.lower() or "refactor" in request.query.lower():
            prompt_parts.append("\nAnalyze complexity metrics and suggest specific refactoring strategies.")
        
        prompt_parts.append("\nPlease structure your response clearly and provide actionable insights.")
        
        return "\n".join(prompt_parts)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for code analysis."""
        return """You are an expert software engineer and code analyst. Your role is to:

1. Analyze code for quality, performance, security, and maintainability
2. Provide clear, actionable suggestions for improvements
3. Explain complex code concepts in understandable terms
4. Identify potential bugs, edge cases, and security issues
5. Suggest refactoring opportunities and best practices
6. Generate comprehensive documentation

Guidelines:
- Be specific and provide concrete examples when possible
- Consider multiple programming languages and their idioms
- Focus on practical, implementable suggestions
- Explain the reasoning behind your recommendations
- Consider both immediate fixes and long-term architectural improvements
- Be concise but thorough in your analysis

Format your responses clearly with:
- Main analysis/explanation
- Specific improvement suggestions (if applicable)
- Related concepts or patterns (if relevant)"""
    
    def _parse_response(self, response_text: str) -> tuple[str, List[str]]:
        """Parse Groq response to extract analysis and suggestions."""
        lines = response_text.split('\n')
        
        analysis_lines = []
        suggestions = []
        
        current_section = "analysis"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for suggestion indicators
            if any(indicator in line.lower() for indicator in [
                "suggestion:", "recommend:", "improve:", "consider:", "try:", "you could:", "you might:"
            ]):
                current_section = "suggestions"
                # Extract the actual suggestion
                for indicator in ["suggestion:", "recommend:", "improve:", "consider:", "try:"]:
                    if indicator in line.lower():
                        suggestion = line[line.lower().find(indicator) + len(indicator):].strip()
                        if suggestion:
                            suggestions.append(suggestion)
                        break
                continue
            
            # Look for bullet points or numbered lists that might be suggestions
            if line.startswith(('- ', '* ', '1. ', '2. ', '3. ')) and current_section == "suggestions":
                suggestion = line[2:].strip() if line.startswith(('- ', '* ')) else line[3:].strip()
                suggestions.append(suggestion)
                continue
            
            # Everything else goes to analysis
            if current_section == "analysis":
                analysis_lines.append(line)
        
        analysis = '\n'.join(analysis_lines).strip()
        
        # If no explicit suggestions found, extract them from analysis
        if not suggestions:
            suggestions = self._extract_suggestions_from_text(analysis)
        
        return analysis, suggestions[:5]  # Limit to 5 suggestions
    
    def _extract_suggestions_from_text(self, text: str) -> List[str]:
        """Extract suggestions from analysis text using keywords."""
        suggestions = []
        sentences = text.split('.')
        
        suggestion_keywords = [
            "should", "could", "might", "consider", "recommend", "suggest",
            "improve", "refactor", "optimize", "use", "try", "implement"
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Ignore very short sentences
                for keyword in suggestion_keywords:
                    if keyword in sentence.lower():
                        suggestions.append(sentence)
                        break
        
        return suggestions[:3]  # Limit to 3 auto-extracted suggestions
    
    def _find_related_symbols(self, request: AnalysisRequest, vectorstore: TextBasedVectorStore, analysis: str) -> List[str]:
        """Find symbols related to the analysis."""
        related_symbols = []
        
        # If analyzing a specific symbol, find its references
        if request.symbol_name:
            symbol = vectorstore.get_symbol_by_name(request.symbol_name, request.file_path)
            if symbol:
                # Add symbols this symbol calls
                related_symbols.extend(symbol.calls)
                
                # Find symbols that call this symbol
                for other_symbol in vectorstore.symbol_metadata.values():
                    if request.symbol_name in other_symbol.calls:
                        related_symbols.append(other_symbol.name)
        
        # Look for symbol names mentioned in the analysis
        words = analysis.lower().split()
        for symbol in vectorstore.symbol_metadata.values():
            if symbol.name.lower() in words and symbol.name not in related_symbols:
                related_symbols.append(symbol.name)
        
        return list(set(related_symbols))[:10]  # Limit and deduplicate