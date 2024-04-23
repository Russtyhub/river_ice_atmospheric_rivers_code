#!/bin/bash

TEXFILENAME=AR_analysis_

all:
	pdflatex ${TEXFILENAME}
	pdflatex ${TEXFILENAME}
	bibtex ${TEXFILENAME}
	bibtex ${TEXFILENAME}
	pdflatex ${TEXFILENAME}
	bibtex ${TEXFILENAME}
	pdflatex ${TEXFILENAME}

clean:
	rm ${TEXFILENAME}.aux ${TEXFILENAME}.lot ${TEXFILENAME}.toc ${TEXFILENAME}.lof ${TEXFILENAME}.bbl ${TEXFILENAME}.blg ${TEXFILENAME}.log ${TEXFILENAME}.spl ${TEXFILENAME}.out
