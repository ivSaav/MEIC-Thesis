import pandas as pd
import argparse
from pathlib import Path

MIN_CITATION = 10
OUT_DIR = Path('./out')
FILTER_MHD = '[Ii]mage|COVID|segment*|medical'
FILTER_TAB = '[Ii]mag*|segment*|visual'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')

    args, _ = parser.parse_known_args()

    if not args.filename:
        print('Missing args.')
        exit(1)

    filename = Path(args.filename)

    papers = pd.read_csv(filename)

    papers = papers.loc[:, ~papers.columns.str.contains('^Unnamed')]

    print('Original size: ', len(papers))
    papers = papers[~papers.Title.str.contains(FILTER_TAB, regex=True)]
    print('title filter: ', len(papers))

    papers = papers[papers.Citations > MIN_CITATION]
    print('citation filter: ', len(papers))
    print(papers.head)

    print()
    papers.to_csv(OUT_DIR / f"{filename.stem}_out.csv")

    
    # print(papers.columns)

