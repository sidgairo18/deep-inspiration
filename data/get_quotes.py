import os
import sqlite3

conn = sqlite3.connect('./author_quotes.sqlite3')
cursor = conn.execute("SELECT quote from author_quote")

#f = open('quotes_data.txt', 'w')
for i, row in enumerate(cursor):
    s = row[0].strip()
    s = s.encode('utf-8')
    #f.write(s+'\n')
    print(s)
#f.close()

f = open('out')
f2 = open('quotes_data.txt', 'w')
for line in f:
    line = line.strip()
    l = len(line)
    line = line[2:l-1]
    print (line)
    f2.write(line+'\n')

f2.close()
f.close()
