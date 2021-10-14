import sys
import pandas as pd


def writefile(filename, texts):
    with open(filename, 'w') as f:
        for item in texts:
            f.write(item.strip())
            f.write('\n')



def main():
    csvname = sys.argv[1] #.e.g. train.tsv
    outputname = sys.argv[2] #.e.g. train/dev
    column1 = sys.argv[3] #.e.g. English
    column2 = sys.argv[4] #.e.g. Yoruba
    srccode = sys.argv[5] #e.g. en
    tgtcode = sys.argv[6] #e.g. yo


    csvfile = pd.read_csv(csvname, sep='\t')
    srctexts = csvfile[column1].tolist()
    tgttexts = csvfile[column2].tolist()

    srctxt = outputname+'.'+srccode
    tgttxt = outputname+'.'+tgtcode

    #write the texts to a text file
    writefile(srctxt, srctexts)
    writefile(tgttxt, tgttexts)


    


if __name__ == '__main__':
    main()
