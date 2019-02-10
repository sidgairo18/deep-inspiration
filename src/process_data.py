from get_data import read_data

def process(filename):

    quotes = read_data(filename)
    
    #quotes appended to a single string.

    all_quotes = ""
    for q in quotes:
        all_quotes += q+" "

    #Map each character to an integer
    chars = tuple(set(all_quotes))

