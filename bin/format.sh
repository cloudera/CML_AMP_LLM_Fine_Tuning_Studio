#!/bin/bash

# Run autoflake to remove unused imports
autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports --recursive ft/ tests/ pgs/

# Run autopep8 to format the code
autopep8 ft/ tests/ pgs/
