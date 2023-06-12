import regex as re

def file_to_sentence_list(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    # Splitting the text into sentences using delimiters like '.', '?', and '!'
    sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', text) if sentence.strip()]

    return sentences


# Example dataset
file_path = 'E:/gfg post/pizza.txt'
text_data = file_to_sentence_list(file_path)

print(text_data)
