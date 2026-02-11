#!/bin/bash

# Define Input and Output
INPUT="docs/derivation_uhf.md"
PDF_OUT="docs/derivation_uhf.pdf"
HTML_OUT="docs/derivation_uhf.html"

# Check if input exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file $INPUT not found!"
    exit 1
fi

echo "========================================"
echo "Generating Documentation"
echo "========================================"

# 1. Generate PDF (using pdflatex)
# Note: We add -N to number sections automatically in LaTeX
echo "1. Building PDF..."
pandoc "$INPUT" \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -N \
    -o "$PDF_OUT"

if [ $? -eq 0 ]; then
    echo "   [OK] PDF generated at $PDF_OUT"
else
    echo "   [FAIL] PDF generation failed."
fi

# 2. Generate HTML (Fixing the MathJax link)
# We use a specific CDN link that works reliably in modern browsers.
# We also use --self-contained (or -s) to ensure metadata works.
echo "2. Building HTML..."
pandoc "$INPUT" \
    -s \
    --mathjax="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" \
    --metadata title="NOCI UHF Derivation" \
    -o "$HTML_OUT"

if [ $? -eq 0 ]; then
    echo "   [OK] HTML generated at $HTML_OUT"
else
    echo "   [FAIL] HTML generation failed."
fi

echo "========================================"
echo "Done."
