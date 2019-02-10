def read_data(filename = './data/default.txt'):
    
    quotes = []
    f = open(filename, 'r')
    for i, line in enumerate(f):
        line = line.strip().lower()
        print ("Reading", i, line)
        quotes.append(line)
    f.close()

    return quotes

if __name__ == "__main__":

    read_data('../data/quotes_data.txt')
