import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-fin", "--in_file_path", type=str, help="Path to file to be converted.", default="quora.csv")
parser.add_argument("-fout", "--out_file_path", type=str, help="Path to converted file.", default="quora.tsv")
args = parser.parse_args()

with open(args.in_file_path, 'r') as csvin, open(args.out_file_path, 'w') as tsvout:
    csvin = csv.reader(csvin)
    tsvout = csv.writer(tsvout, delimiter='\t')

    for row in csvin:
        tsvout.writerow(row)
