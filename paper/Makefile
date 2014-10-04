

MD_FILE = Prices_and_the_Empirical_Study_of_War_Theory.md
PDF_FILE = $(MD_FILE:%.md=%.pdf)

PANDOC = pandoc
PANDOC_OPTS = 

$(PDF_FILE): $(MD_FILE)
	$(PANDOC) -o $@ $<
