# Set the PDF mode to use XeLaTeX explicitly
$pdf_mode = 5;

# Optional: Ensure the recorder and synctex flags are passed (good for VS Code/editors)
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -recorder %O %S';

# Optional: Add extra file extensions to clean up when running 'latexmk -c'
$clean_ext = "synctex.gz xdv nav snm vrb";