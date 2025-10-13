# PYTHON_HOME Environment Variable

The Redline SpaCy integration supports the `PYTHON_HOME` environment variable to use Python virtual environments at runtime.

## Overview

When you set `PYTHON_HOME`, the SpaCy analyzer will:

1. **Auto-detect Python version** - Reads `sys.version_info` to get major.minor version
2. **Configure sys.path** - Prepends `{PYTHON_HOME}/lib/python{major}.{minor}/site-packages` to Python's module search path
3. **Load venv packages** - SpaCy and other packages from your virtual environment are used instead of system packages

This works with **any Python version** (3.7+) as the version is detected automatically.

## Usage

### Quick Start

```bash
# Set PYTHON_HOME to your virtual environment
export PYTHON_HOME=$(pwd)/.venv

# Run your application
cargo run --example spacy_syntactic_demo --features spacy
```

### One-liner

```bash
PYTHON_HOME=$(pwd)/.venv cargo run --example spacy_syntactic_demo --features spacy
```

### In Your Shell Profile

Add to `~/.bashrc`, `~/.zshrc`, or similar:

```bash
# For a specific project
export PYTHON_HOME=/path/to/project/.venv

# Or dynamically based on current directory
export PYTHON_HOME=$(pwd)/.venv
```

### With direnv (.envrc)

```bash
# In your .envrc file
export PYTHON_HOME=$(pwd)/.venv
```

Then run:
```bash
direnv allow
```

## How It Works

### Implementation Details

The `SpacySyntacticAnalyzer::configure_python_path()` function:

```rust
fn configure_python_path(py: Python, python_home: &str) -> Result<(), String> {
    // 1. Get sys module
    let sys = py.import_bound("sys")?;
    
    // 2. Detect Python version
    let version_info = sys.getattr("version_info")?;
    let major = version_info.getattr("major")?.extract::<i32>()?;
    let minor = version_info.getattr("minor")?.extract::<i32>()?;
    
    // 3. Construct site-packages path
    let site_packages = format!("{}/lib/python{}.{}/site-packages", python_home, major, minor);
    
    // 4. Prepend to sys.path
    let path = sys.getattr("path")?;
    path.call_method1("insert", (0, &site_packages))?;
    
    Ok(())
}
```

This is called automatically in `get_or_load_nlp()` when `PYTHON_HOME` is set.

### Python Version Detection

The implementation uses Python's own `sys.version_info` to detect the version, so it works correctly with:

- Python 3.7
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12+

## Examples

### Example 1: Project-Local Virtual Environment

```bash
# Create venv in project
cd /path/to/redline
python3 -m venv .venv
source .venv/bin/activate
pip install spacy
python -m spacy download en_core_web_sm

# Use it
export PYTHON_HOME=$(pwd)/.venv
cargo run --example spacy_syntactic_demo --features spacy
```

### Example 2: System-Wide Virtual Environment

```bash
# Create venv elsewhere
python3 -m venv ~/my-python-env
source ~/my-python-env/bin/activate
pip install spacy
python -m spacy download en_core_web_sm

# Use it from any project
export PYTHON_HOME=~/my-python-env
cd /path/to/redline
cargo run --example spacy_syntactic_demo --features spacy
```

### Example 3: Multiple Python Versions

```bash
# Python 3.10 venv
python3.10 -m venv .venv-310
export PYTHON_HOME=$(pwd)/.venv-310
cargo run --features spacy  # Uses Python 3.10 packages

# Python 3.12 venv
python3.12 -m venv .venv-312
export PYTHON_HOME=$(pwd)/.venv-312
cargo run --features spacy  # Uses Python 3.12 packages
```

## Troubleshooting

### Package Not Found

If you get "Failed to import spacy" even with `PYTHON_HOME` set:

1. **Verify SpaCy is installed in the venv:**
   ```bash
   $PYTHON_HOME/bin/python -c "import spacy; print(spacy.__version__)"
   ```

2. **Check the constructed path:**
   The analyzer looks for packages at:
   ```
   {PYTHON_HOME}/lib/python{major}.{minor}/site-packages
   ```

3. **Verify the path exists:**
   ```bash
   ls $PYTHON_HOME/lib/python*/site-packages/spacy
   ```

### Wrong Python Version Detected

If the wrong Python version is being used:

1. **Check which Python PyO3 is linked against:**
   ```bash
   cargo clean
   cargo build --features spacy -vv | grep python
   ```

2. **Set PYO3_PYTHON for build time:**
   ```bash
   export PYO3_PYTHON=$PYTHON_HOME/bin/python
   cargo build --features spacy
   ```

### Virtual Environment Not Activated

You **do NOT need** to activate the virtual environment. Just set `PYTHON_HOME`:

```bash
# ❌ Not necessary
source .venv/bin/activate

# ✅ This is enough
export PYTHON_HOME=$(pwd)/.venv
```

## Build vs Runtime

### Build Time (PyO3 Linking)

Set `PYO3_PYTHON` to specify which Python to link against during compilation:

```bash
export PYO3_PYTHON=$(pwd)/.venv/bin/python
cargo build --features spacy
```

### Runtime (Package Loading)

Set `PYTHON_HOME` to specify which packages to use when the program runs:

```bash
export PYTHON_HOME=$(pwd)/.venv
cargo run --features spacy
```

### Both Together

For complete virtual environment isolation:

```bash
export PYO3_PYTHON=$(pwd)/.venv/bin/python
export PYTHON_HOME=$(pwd)/.venv
cargo build --features spacy
cargo run --features spacy
```

## Platform Notes

### Linux

```bash
# Standard location
export PYTHON_HOME=$(pwd)/.venv
# Expects: .venv/lib/python3.x/site-packages
```

### macOS

```bash
# Standard location (same as Linux)
export PYTHON_HOME=$(pwd)/.venv
# Expects: .venv/lib/python3.x/site-packages
```

### Windows

```bash
# PowerShell
$env:PYTHON_HOME = "$PWD\.venv"
# Expects: .venv\Lib\site-packages

# Note: Path construction may need adjustment for Windows
```

## See Also

- [SPACY_SETUP.md](SPACY_SETUP.md) - Complete SpaCy integration guide
- [PyO3 Documentation](https://pyo3.rs/) - Python-Rust bindings
- [Python Virtual Environments](https://docs.python.org/3/library/venv.html)
