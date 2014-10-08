MD_FILE = Prices_and_the_Empirical_Study_of_War_Theory.md
PDF_FILE = $(MD_FILE:%.md=%.pdf)

SLIDES_MD = PSSS2014.md
SLIDES_PDF = $(SLIDES_MD:%.md=%.pdf)

PANDOC = pandoc
PANDOC_OPTS = -f markdown+yaml_metadata_block

all: $(SLIDES_PDF) $(PDF_FILE)

$(SLIDES_PDF): $(SLIDES_MD)
	$(PANDOC) $(PANDOC_OPTS) -t beamer -o $@ $<

$(PDF_FILE): $(MD_FILE)
	$(PANDOC) $(PANDOC_OPTS) -o $@ $<
