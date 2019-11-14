""" Obtain and parse metadata PDFs from the UN SDG site """

import os
import sys
import io
import requests
import pdfminer.high_level
import pdfminer.layout
import lxml.html

METADATA_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'metadata')


def get_all_metadata_pdfs():
    r = requests.get("https://unstats.un.org/sdgs/metadata/")
    html = lxml.html.fromstring(r.content)
    ahrefs = html.xpath('//a[@title=\'View in PDF\']/@href')

    for ahref in ahrefs:
        if 'Metadata-' not in ahref:
            continue
        print("Getting file %s" % ahref, file=sys.stderr)
        get_metadata_pdf(ahref)


def get_metadata_pdf(ahref):
    rf = requests.get("https://unstats.un.org/" + ahref, stream=True)
    with open(os.path.join(METADATA_PATH, ahref[ahref.index('Metadata'):]), 'wb') as f:
        for chunk in rf.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def extract_description(pdf_path):
    laparams = pdfminer.layout.LAParams()
    for param in ("all_texts", "detect_vertical", "word_margin", "char_margin", "line_margin", "boxes_flow"):
        paramv = locals().get(param, None)
        if paramv is not None:
            setattr(laparams, param, paramv)

    inputf = open(pdf_path, "rb")
    ff = io.StringIO()
    pdfminer.high_level.extract_text_to_fp(inputf, ff, laparams=laparams)
    inputf.close()

    converted_text = ff.getvalue()

    return converted_text
