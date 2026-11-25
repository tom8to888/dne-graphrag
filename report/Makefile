# make         -> main.pdf
# make tables  -> regenerate report/tables/*.tex from results/tables (after notebook 05 changes)
# make clean   -> remove build files
main.pdf: main.tex references.bib $(wildcard tables/*.tex)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

tables:
	cd .. && ~/envs/dne_graphrag/bin/python tools/make_report_tables.py

clean:
	latexmk -C

.PHONY: tables clean
