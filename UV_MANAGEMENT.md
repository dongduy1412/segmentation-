# UV Package Manager - Management Guide

## Installation Status

✅ **uv is now properly installed via pipx**

This is the recommended installation method according to [uv documentation](https://docs.astral.sh/uv/getting-started/installation/#pypi).

## Installation Details

- **Method**: pipx (isolated Python environment)
- **Version**: 0.9.5
- **Location**: `C:\Users\Victus\.local\bin\`
- **Binaries**: `uv.exe`, `uvx.exe`, `uvw.exe`

## Why pipx?

1. **Isolated Environment** - uv runs in its own virtual environment
2. **No Conflicts** - Won't interfere with other Python packages
3. **Easy Updates** - Simple upgrade command
4. **System-wide Access** - Available globally in PATH

## Updating uv

Since uv was installed via pipx, use pipx to update:

```bash
# Update uv to latest version
py -m pipx upgrade uv

# Check current version
uv --version
```

**Note**: `uv self update` is only available for standalone installer installations. With pipx, use `pipx upgrade uv` instead.

## Using uv in This Project

Now you can use `uv` directly without `py -m uv`:

### Before (with pip-installed uv):
```bash
py -m uv run python main.py test.jpg
py -m uv add some-package
py -m uv sync
```

### After (with pipx-installed uv):
```bash
# Direct command (if PATH is updated in current session)
uv run python main.py test.jpg
uv add some-package
uv sync

# Or refresh PATH first
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
uv run python main.py test.jpg
```

**Important**: After installation, you may need to:
1. Open a new terminal window, OR
2. Refresh PATH in current session (see command above)

## Verifying Installation

```bash
# Check uv is accessible
uv --version

# Check installation location
pipx list

# Test with project
cd D:\segmentation
uv run python main.py test.jpg --no-display
```

## Managing uv with pipx

### List installed pipx packages
```bash
py -m pipx list
```

### Reinstall uv
```bash
py -m pipx reinstall uv
```

### Uninstall uv
```bash
py -m pipx uninstall uv
```

## Alternative Installation Methods

If you want to use a different method in the future:

### 1. Standalone Installer (Windows)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
- Supports `uv self update`
- Installs to `~/.local/bin/`

### 2. WinGet
```bash
winget install --id=astral-sh.uv -e
```
- Update with: `winget upgrade astral-sh.uv`

### 3. Scoop
```bash
scoop install main/uv
```
- Update with: `scoop update uv`

## Troubleshooting

### "uv: command not found"

**Solution 1**: Open a new terminal window

**Solution 2**: Refresh PATH in current session:
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
```

**Solution 3**: Use full path:
```bash
C:\Users\Victus\.local\bin\uv.exe --version
```

### "Self-update not available"

This is expected with pipx installation. Use instead:
```bash
py -m pipx upgrade uv
```

### Update pipx itself

```bash
py -m pip install --upgrade pipx
```

## Project Commands (Updated)

All commands can now use `uv` directly:

```bash
# Run the application
uv run python main.py test.jpg

# Add a new package
uv add package-name

# Remove a package
uv remove package-name

# Update all dependencies
uv sync

# Lock dependencies
uv lock

# Show project info
uv pip list
```

## Benefits of This Setup

✅ **Proper Installation** - Following uv documentation recommendations
✅ **Isolated** - No conflicts with other Python packages
✅ **Easy Updates** - `pipx upgrade uv`
✅ **System-wide** - Available in all projects
✅ **Clean** - Can be fully removed with `pipx uninstall uv`

## References

- [uv Documentation - Installation](https://docs.astral.sh/uv/getting-started/installation/)
- [pipx Documentation](https://pipx.pypa.io/)
- [uv GitHub Repository](https://github.com/astral-sh/uv)

---

**Status**: ✅ Properly configured and ready to use!

**Last Updated**: 2025-10-28
