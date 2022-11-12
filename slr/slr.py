import pandas as pd
import argparse
from pathlib import Path

MIN_CITATION = 10
OUT_DIR = Path('./out')
FILTER_MHD = '[Ii]mage|COVID|segment*|medical'
FILTER_TAB = '[Ii]mag*|segment*|visual'
FILTER_ANOM = '[Ii]mag*|segment*|visual'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--title_filter', dest="title_filter")

    args, _ = parser.parse_known_args()

    if not args.filename:
        print('Missing args.')
        exit(1)
    
    title_filter = args.title_filter if args.title_filter else FILTER_MHD

    filename = Path(args.filename)

    papers = pd.read_csv(filename)

    papers = papers.loc[:, ~papers.columns.str.contains('^Unnamed')]

    print('Original size: ', len(papers))
    papers = papers[~papers.Title.str.contains(title_filter, regex=True)]
    print('title filter: ', len(papers))

    papers = papers[
                        (papers.Citations > MIN_CITATION) | 
                        ((papers.Year.astype(int) > 2022) &     # ease citation restriction for recent papers
                            (papers.Citations > MIN_CITATION/2))
                    ]
    print('citation filter: ', len(papers))
    print(papers.head)

    print()
    papers.to_csv(OUT_DIR / f"{filename.stem}_out.csv")

    
    # print(papers.columns)

