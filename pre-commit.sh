#!/bin/bash
isort $(git ls-files '*.py')
python -m pylint $(git ls-files '*.py')
