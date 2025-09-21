LEXICON SET UP
1. copy words from links below
2. create two files ./data/lexicons/pos.txt and ./data/lexicons/neg.txt
3. paste contents into corresponding file
positive lexicon: https://ptrckprry.com/course/ssd/data/positive-words.txt
negative lexicon: https://ptrckprry.com/course/ssd/data/negative-words.txt

DATA SET UP
1. Get the .tar.gz from from https://www.cs.cornell.edu/people/pabo/movie-review-data/
2. Drag into repo 
3. unzip with command `tar -xzvf review_polarity.tar.gz`
4. drag the created folder (txt_sentoken) into ./data/raw

CLEAN AND FORMAT DATA
run command `python ./src/data_managment/clean_polarity2.py`

NOTE
./data also contains polarity2.md with information about the dataset if you'd like to read more.