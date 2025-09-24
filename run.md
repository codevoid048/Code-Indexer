## CLI 

### 1. Set up environment (Optional - for AI features)

```bash
# Set your Groq API key for AI analysis features
export GROQ_API_KEY="your-groq-api-key"
```

### 2. Index your code

```bash
# Index current directory
python main.py index .

# Index with force reindexing
python main.py index /path/to/your/code --force

# Index without subdirectories
python main.py index . --no-recursive
```

### 3. Search your code

```bash
# General search
python main.py search "MyClass"

# Search only symbols
python main.py search "function_name" --type symbol

# Search files
python main.py search "utils.py" --type file

# Limit results
python main.py search "MyClass" --limit 5
```

### 4. AI-powered analysis

```bash
# Analyze code with AI
python main.py analyze "explain this function" --symbol my_function

# Analyze entire file
python main.py analyze "find potential bugs" --file src/utils.py
```