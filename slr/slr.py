import pandas as pd
import argparse
from pathlib import Path

MIN_CITATION = 6
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
    
    title_filter = args.title_filter if args.title_filter else FILTER_ANOM

    filename = Path(args.filename)

    papers = pd.read_csv(filename)

    papers = papers.loc[:, ~papers.columns.str.contains('^Unnamed')]
    

    print('Original size: ', len(papers))
    
    
    papers = papers[
                        (papers.Citations > MIN_CITATION) | 
                        ((papers.Year.astype(int) > 2021) &     # ease citation restriction for recent papers
                            (papers.Citations > MIN_CITATION//2))
                    ]
    print('citation filter: ', len(papers))
    print(papers.head)

    papers = papers[papers.Year >= 2014]
    print('Date filter: ', len(papers))

    papers = papers[~papers.Title.str.contains(title_filter, regex=True)]
    print('title filter: ', len(papers))
    
    papers = papers.drop_duplicates(subset=["Title"])
    print("haha", len(papers))
    
    print()
    papers.to_csv(OUT_DIR / f"{filename.stem}_out3.csv")

    
    # print(papers.columns)

