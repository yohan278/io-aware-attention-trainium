PDF=main.pdf

.PHONY: all clean assets

all: assets
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

assets:
	python ../scripts/generate_paper_assets.py

clean:
	latexmk -C
