TEX=pdflatex
LATEXMK=latexmk

MAIN=main

.PHONY: all clean distclean

all:
	$(LATEXMK) -pdf $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex

distclean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f $(MAIN).pdf
