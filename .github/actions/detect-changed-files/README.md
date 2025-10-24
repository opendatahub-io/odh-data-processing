# Detect Changed Files Action

A GitHub Action that detects changed files using native Git commands with support for pull requests and push events.

## Features

- 🔍 **Smart Detection**: Automatically detects changed files based on Git history
- 📋 **Event Support**: Works with pull requests, push events, and manual triggers
- 🎯 **Pattern Matching**: Supports glob patterns for file filtering
- 📊 **Rich Outputs**: Provides file lists, change counts, and boolean flags

## Usage

### Basic Usage

```yaml
- name: Get changed files
  id: changed-files
  uses: ./.github/actions/detect-changed-files
  with:
    files: 'notebooks/**/*.ipynb'
```

### Advanced Usage

```yaml
- name: Get changed files
  id: changed-files
  uses: ./.github/actions/detect-changed-files
  with:
    files: 'src/**/*.py'
    base-sha: ${{ github.event.pull_request.base.sha }}
    head-sha: ${{ github.event.pull_request.head.sha }}
```

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `files` | File pattern to match (e.g., `notebooks/**/*.ipynb`) | ✅ | - |
| `token` | GitHub token for API access | ❌ | `${{ github.token }}` |
| `base-sha` | Base SHA for comparison | ❌ | Auto-detected |
| `head-sha` | Head SHA for comparison | ❌ | Auto-detected |

## Outputs

| Output | Description |
|--------|-------------|
| `all_changed_files` | JSON array of all changed files matching the pattern |
| `has_changes` | Boolean indicating if any files changed |
| `files_count` | Number of changed files |

## Examples

### Check for notebook changes

```yaml
- name: Check for notebook changes
  id: notebooks
  uses: ./.github/actions/detect-changed-files
  with:
    files: 'notebooks/**/*.ipynb'

- name: Run tests if notebooks changed
  if: steps.notebooks.outputs.has_changes == 'true'
  run: |
    echo "Found changed notebooks:"
    echo "${{ steps.notebooks.outputs.all_changed_files }}"
```

### Conditional workflow steps

```yaml
- name: Get changed Python files
  id: python-files
  uses: ./.github/actions/detect-changed-files
  with:
    files: 'src/**/*.py'

- name: Run linting
  if: steps.python-files.outputs.has_changes == 'true'
  run: |
    echo "Linting ${{ steps.python-files.outputs.files_count }} Python files"
```

## Supported Events

- **Pull Requests**: Compares PR head with base branch
- **Push Events**: Compares current commit with previous commit
- **Manual Triggers**: Finds all files matching the pattern

## Pattern Examples

- `notebooks/**/*.ipynb` - All Jupyter notebooks in notebooks directory
- `src/**/*.py` - All Python files in src directory
- `*.md` - All Markdown files in root directory
- `docs/**/*` - All files in docs directory

## Author

ODH Data Processing Team