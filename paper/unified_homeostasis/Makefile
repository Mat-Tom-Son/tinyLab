all: main.pdf

main.pdf: main.tex sections/*.tex
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz

.PHONY: all clean
