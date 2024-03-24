file_path = 'maze_runner.txt'

# num words

def load_text(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

text = load_text(file_path)
words = text.split()
num_words = len(words)
print(num_words)