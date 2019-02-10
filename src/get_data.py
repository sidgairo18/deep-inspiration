def read_data(filename = './data/default.txt'):
    
    quotes = []
    f = open(filename)
    for i, line in enumerate(f):
        line = line.strip().split('    ')
        print (line)
        if i == 100:
            break
        quotes.append(line)

    f.close()

    return quotes

if __name__ == "__main__":

    read_data('../data/quotes_data.txt')
