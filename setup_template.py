#!/usr/bin/env python3
"""
Template setup script to help customize the Python package template.

This script helps users replace template placeholders with their actual values.
Run this after creating a new repository from the template.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def get_user_input(prompt: str, default: str = "", required: bool = True) -> str:
    """Get user input with optional default value."""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    while True:
        value = input(full_prompt).strip()
        if value:
            return value
        elif default:
            return default
        elif not required:
            return ""
        else:
            print("This field is required. Please provide a value.")


def collect_template_variables() -> Dict[str, str]:
    """Collect all template variables from user input."""
    print("🚀 Python Package Template Setup")
    print("=" * 50)
    print("This script will help you customize your new Python package.")
    print("Press Ctrl+C at any time to exit.\n")
    
    variables = {}
    
    # Required variables
    print("📝 Required Information:")
    variables["repo_name"] = get_user_input("Repository name (e.g., my-awesome-package)")
    variables["package_name"] = get_user_input(
        "Python package name (e.g., my_awesome_package)", 
        variables["repo_name"].replace("-", "_").replace(" ", "_").lower()
    )
    variables["pypi_package_name"] = get_user_input(
        "PyPI package name (e.g., my-awesome-package)", 
        variables["repo_name"]
    )
    variables["description"] = get_user_input("Short description")
    variables["author_name"] = get_user_input("Your full name")
    variables["author_email"] = get_user_input("Your email address")
    variables["github_username"] = get_user_input("Your GitHub username")
    
    # Derive first and last name
    name_parts = variables["author_name"].split()
    variables["author_first_name"] = name_parts[0] if name_parts else ""
    variables["author_last_name"] = name_parts[-1] if len(name_parts) > 1 else ""
    
    # License selection
    print("\n📜 License Selection:")
    licenses = {
        "1": ("MIT License", "MIT", "MIT License"),
        "2": ("BSD 3-Clause License", "BSD-3-Clause", "BSD License"),
        "3": ("Apache License 2.0", "Apache-2.0", "Apache Software License"),
        "4": ("GNU GPL v3.0", "GPL-3.0", "GNU General Public License v3 (GPLv3)"),
    }
    
    print("Choose a license:")
    for key, (name, _, _) in licenses.items():
        print(f"  {key}. {name}")
    
    license_choice = get_user_input("License choice", "2")
    if license_choice in licenses:
        variables["license_name"], variables["license_id"], variables["license_classifier"] = licenses[license_choice]
    else:
        variables["license_name"] = "BSD 3-Clause License"
        variables["license_id"] = "BSD-3-Clause"
        variables["license_classifier"] = "BSD License"
    
    # Optional variables
    print("\n🔧 Optional Information (press Enter to skip):")
    variables["long_description"] = get_user_input(
        "Detailed description", 
        f"A Python package for {variables['description'].lower()}.", 
        required=False
    )
    variables["main_class"] = get_user_input(
        "Main class name", 
        variables["package_name"].title().replace("_", ""), 
        required=False
    )
    variables["keywords"] = get_user_input(
        "Keywords (comma-separated)", 
        "python,package", 
        required=False
    )
    variables["orcid_id"] = get_user_input("ORCID ID (optional)", "", required=False)
    variables["codecov_token"] = get_user_input("Codecov token (optional)", "", required=False)
    
    # Derived variables
    from datetime import datetime
    variables["year"] = str(datetime.now().year)
    variables["version"] = "0.1.0"
    variables["release_date"] = datetime.now().strftime("%Y-%m-%d")
    variables["citation_key"] = f"{variables['github_username']}{variables['year']}{variables['package_name'].lower()}"
    
    # Format keywords as Python list
    if variables["keywords"]:
        keywords_list = [f'"{k.strip()}"' for k in variables["keywords"].split(",")]
        variables["keywords"] = ", ".join(keywords_list)
    else:
        variables["keywords"] = '"python", "package"'
    
    return variables


def find_template_files(root_dir: Path) -> List[Path]:
    """Find all files that contain template placeholders."""
    template_files = []
    exclude_patterns = {
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        ".venv",
        "venv",
        "htmlcov",
    }
    
    for path in root_dir.rglob("*"):
        if path.is_file() and not any(pattern in str(path) for pattern in exclude_patterns):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "{{" in content and "}}" in content:
                        template_files.append(path)
            except (UnicodeDecodeError, PermissionError):
                # Skip binary files or files we can't read
                continue
    
    return template_files


def replace_placeholders(file_path: Path, variables: Dict[str, str]) -> bool:
    """Replace template placeholders in a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        original_content = content
        
        # Replace all placeholders
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            content = content.replace(placeholder, value)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False
    
    return False


def rename_template_directories(root_dir: Path, package_name: str) -> None:
    """Rename template directories to match the actual package name."""
    # Rename source directory
    template_src = root_dir / "src" / "{{package_name}}"
    if template_src.exists():
        new_src = root_dir / "src" / package_name
        template_src.rename(new_src)
        print(f"✅ Renamed {template_src} to {new_src}")
    
    # Rename test file
    template_test = root_dir / "tests" / "test_{{package_name}}.py"
    if template_test.exists():
        new_test = root_dir / "tests" / f"test_{package_name}.py"
        template_test.rename(new_test)
        print(f"✅ Renamed {template_test} to {new_test}")


def main():
    """Main setup function."""
    try:
        # Get current directory
        root_dir = Path.cwd()
        
        # Collect variables
        variables = collect_template_variables()
        
        print(f"\n🔍 Searching for template files in {root_dir}...")
        template_files = find_template_files(root_dir)
        
        if not template_files:
            print("❌ No template files found. Are you in the right directory?")
            return 1
        
        print(f"📝 Found {len(template_files)} files to process:")
        for file_path in template_files:
            print(f"  - {file_path.relative_to(root_dir)}")
        
        # Confirm before proceeding
        print(f"\n📋 Summary:")
        print(f"  Repository: {variables['repo_name']}")
        print(f"  Package: {variables['package_name']}")
        print(f"  Author: {variables['author_name']} <{variables['author_email']}>")
        print(f"  License: {variables['license_name']}")
        
        confirm = input("\n🤔 Proceed with template customization? [y/N]: ").lower()
        if confirm not in ["y", "yes"]:
            print("❌ Setup cancelled.")
            return 0
        
        # Process files
        print(f"\n🔄 Processing {len(template_files)} files...")
        processed_count = 0
        
        for file_path in template_files:
            if replace_placeholders(file_path, variables):
                processed_count += 1
                print(f"✅ {file_path.relative_to(root_dir)}")
        
        # Rename directories
        print("\n📁 Renaming template directories...")
        rename_template_directories(root_dir, variables["package_name"])
        
        print(f"\n🎉 Setup complete!")
        print(f"📊 Processed {processed_count} files")
        print(f"\n📚 Next steps:")
        print(f"  1. Review the generated files")
        print(f"  2. Run: make dev-setup")
        print(f"  3. Run: make test")
        print(f"  4. Start coding in src/{variables['package_name']}/")
        print(f"  5. Add your dependencies to pyproject.toml")
        print(f"  6. Set up GitHub secrets (CODECOV_TOKEN, etc.)")
        
        # Offer to delete this setup script
        delete_setup = input(f"\n🗑️  Delete this setup script? [y/N]: ").lower()
        if delete_setup in ["y", "yes"]:
            script_path = Path(__file__)
            if script_path.name == "setup_template.py":
                script_path.unlink()
                print("✅ Setup script deleted.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
