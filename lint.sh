#!/bin/bash
python -m pylint $(git ls-files '*.py')
