# Detect Changed Files Action

A simple GitHub Action that detects changed files using native Git commands.

## Usage

```yaml
- name: Detect changed files
  id: changed-files
  uses: ./.github/actions/detect-changed-files
  with:
    files: 'notebooks/**/*.ipynb'

- name: Use results
  if: steps.changed-files.outputs.has_changes == 'true'
  run: |
    echo "Changed files: ${{ steps.changed-files.outputs.all_changed_files }}"
    echo "File count: ${{ steps.changed-files.outputs.files_count }}"
```

## Inputs

| Input | Description | Required |
|-------|-------------|----------|
| `files` | File pattern to match (e.g., `notebooks/**/*.ipynb`) | Yes |

## Outputs

| Output | Description | Example |
|--------|-------------|---------|
| `all_changed_files` | JSON array of changed files | `["notebooks/example.ipynb"]` |
| `has_changes` | True if any files changed | `"true"` or `"false"` |
| `files_count` | Number of changed files | `"2"` |

## How it Works

- **Pull Requests**: Compares PR base vs head
- **Push Events**: Compares current vs previous commit  
- **Manual/Scheduled**: Returns all matching files

## Examples

```yaml
# Notebooks only
files: 'notebooks/**/*.ipynb'

# Python files
files: 'src/**/*.py'

# Multiple types
files: '**/*.{py,yml,yaml}'
```

Built with native Git commands for security and reliability.