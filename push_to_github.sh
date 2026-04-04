#!/bin/bash
# Script to initialize and push this repository to GitHub
# Usage: bash push_to_github.sh

set -e

echo "=== Initializing Git Repository ==="
git init
git add -A
git commit -m "Initial commit: Risk-Aware Corridor-Constrained Pathfinding

- 13 coordinated experiments (6,000+ planned paths)
- Core algorithms: ILS, AILS, RILS on risk-annotated grids
- Baselines: JPS, D*Lite, Weighted A*
- Moving AI Lab benchmark validation
- All experimental results (CSV)
- LaTeX manuscript source
- Corridor comparison figure (real data)"

echo ""
echo "=== Adding Remote ==="
git remote add origin https://github.com/Amr-path/Risk-Aware-Corridor.git
git branch -M main

echo ""
echo "=== Pushing to GitHub ==="
git push -u origin main

echo ""
echo "=== Done! ==="
echo "Repository available at: https://github.com/Amr-path/Risk-Aware-Corridor"
