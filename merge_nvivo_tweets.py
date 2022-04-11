#!/usr/bin/python -W ignore

# merge_nvivo_tweets.py
# Script to parse multiple NVIVO archive (.nvcx) files containing captured
# tweets and write them to a text file that will be used as input to 
# analyses run in R.
# NOTE: This is intended to work with Python 3
# This code has been adapted to work in a Linux/Mac OS environment.
# Commented out lines below are from the original implementatation,
# meant to generate an output file (tweets.txt) in UTF-16 encoding 
# for use with Windows applications, like R/RStudio for Windows.

import os
import sys
import io
#import xml.etree.ElementTree as ET
import glob
import lxml.etree as ET
parser = ET.XMLParser(recover=True)


ns = {'nvivo': 'http://qsr.com.au/NVivoWebExportXMLSchema.xsd'}

print(str(sys.argv))
# The script will process all .nvcx files found in this folder
#input_dir = '../S2/MergeNVTweets/nvivo_files/'
input_dir_files = sys.argv[1]
#nv_dir_files = os.listdir(input_dir)

#outfile = io.open('tweets.txt', 'w', encoding='utf-16', newline="\r\n")
#outfile = open('tweets.txt', 'w') # non-Windows

outfile = None
if len(sys.argv) == 3:
  out_fn = sys.argv[2]
  outfile = open(out_fn, 'w') # non-Windows

row_count = 0
#for entity in nv_dir_files:
for file_path in glob.glob(input_dir_files):
  #file_path = input_dir + entity
  if os.path.isfile(file_path):
    print("Extracting tweets from " + file_path)
    fileExt = file_path.split('.')[-1]
    if (fileExt != 'nvcx'):
      continue

    f_tree = ET.parse(file_path, parser=parser)
    f_root = f_tree.getroot()

    f_dataset = f_tree.find('nvivo:Dataset', ns)
    if f_dataset is None:
      print("ERROR FINDING DATASET IN",file_path,"SKIPPING")
      continue
    f_rows = f_dataset.find('nvivo:Rows', ns)

    for f_row in f_rows.findall('nvivo:Row', ns):
      row_count += 1
      if not outfile:
        continue 
      column_count = 0
      tweet_array = [str(row_count)]
      for f_col in f_row.findall('nvivo:Column', ns):
        column_count += 1
        col_text_iter = f_col.itertext()
        col_text = ""
        for c_text in col_text_iter:
          col_text += c_text.replace("\n", "") + " "
          #col_text += c_text
        tweet_array.append(col_text)

      tweet_text = "\t".join(tweet_array) + "\n"

      #outfile.write(tweet_text) # non-Windows (commented out by MB)
      #outfile.write(unicode(tweet_text))
      outfile.write(tweet_text.encode('utf-8'))
   
if outfile:       
  outfile.close()

print("TOTAL TWEETS:",row_count)
