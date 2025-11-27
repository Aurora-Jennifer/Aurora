#!/usr/bin/env python3
"""Script to add GNU AGPL v3 license headers to all files in ALPACA_trading."""

import os
from pathlib import Path

LICENSE_HEADER_PYTHON = """# ALPACA Paper Trading Service
#
# Copyright (C) 2024  <name of author>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

LICENSE_HEADER_YAML = """# ALPACA Paper Trading Service
#
# Copyright (C) 2024  <name of author>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

LICENSE_HEADER_JSON = """{
  "_license": "GNU Affero General Public License v3 or later",
  "_copyright": "Copyright (C) 2024 <name of author>",
  "_notice": "This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details. You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.",
"""

LICENSE_HEADER_MD = """<!--
ALPACA Paper Trading Service

Copyright (C) 2024  <name of author>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

"""

def has_license_header(content: str, file_type: str) -> bool:
    """Check if file already has a license header."""
    if file_type == "python":
        return "GNU Affero General Public License" in content[:500]
    elif file_type == "yaml":
        return "GNU Affero General Public License" in content[:500]
    elif file_type == "json":
        return "_license" in content[:200]
    elif file_type == "md":
        return "GNU Affero General Public License" in content[:500]
    return False

def add_header_to_python(file_path: Path):
    """Add license header to Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if has_license_header(content, "python"):
        print(f"  ✓ {file_path} already has license header")
        return
    
    # Handle shebang
    if content.startswith('#!/'):
        lines = content.split('\n', 1)
        new_content = lines[0] + '\n' + LICENSE_HEADER_PYTHON + lines[1] if len(lines) > 1 else lines[0] + '\n' + LICENSE_HEADER_PYTHON
    else:
        new_content = LICENSE_HEADER_PYTHON + content
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"  ✓ Added header to {file_path}")

def add_header_to_yaml(file_path: Path):
    """Add license header to YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if has_license_header(content, "yaml"):
        print(f"  ✓ {file_path} already has license header")
        return
    
    new_content = LICENSE_HEADER_YAML + content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"  ✓ Added header to {file_path}")

def add_header_to_json(file_path: Path):
    """Add license header to JSON file."""
    import json
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "_license" in data:
            print(f"  ✓ {file_path} already has license header")
            return
        
        # Add license fields at the beginning
        new_data = {
            "_license": "GNU Affero General Public License v3 or later",
            "_copyright": "Copyright (C) 2024 <name of author>",
            "_notice": "This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details. You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.",
            **data
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Added header to {file_path}")
    except json.JSONDecodeError:
        print(f"  ⚠ {file_path} is not valid JSON, skipping")

def add_header_to_md(file_path: Path):
    """Add license header to Markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if has_license_header(content, "md"):
        print(f"  ✓ {file_path} already has license header")
        return
    
    new_content = LICENSE_HEADER_MD + content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"  ✓ Added header to {file_path}")

def process_directory(root_dir: Path):
    """Process all files in directory tree."""
    python_files = list(root_dir.rglob("*.py"))
    yaml_files = list(root_dir.rglob("*.yaml")) + list(root_dir.rglob("*.yml"))
    json_files = list(root_dir.rglob("*.json"))
    md_files = list(root_dir.rglob("*.md"))
    
    print(f"Processing {len(python_files)} Python files...")
    for f in python_files:
        if f.name == "add_license_headers.py":
            continue
        add_header_to_python(f)
    
    print(f"\nProcessing {len(yaml_files)} YAML files...")
    for f in yaml_files:
        add_header_to_yaml(f)
    
    print(f"\nProcessing {len(json_files)} JSON files...")
    for f in json_files:
        add_header_to_json(f)
    
    print(f"\nProcessing {len(md_files)} Markdown files...")
    for f in md_files:
        add_header_to_md(f)

if __name__ == "__main__":
    root = Path(__file__).parent
    print(f"Adding license headers to files in {root}")
    process_directory(root)
    print("\n✓ Done!")



