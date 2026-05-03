PAPER = main

.PHONY: all clean distclean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex neurips_2024.sty
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

clean:
	latexmk -c $(PAPER).tex

distclean:
	latexmk -C $(PAPER).tex

view: $(PAPER).pdf
	open $(PAPER).pdf
