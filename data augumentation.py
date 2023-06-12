import random
from nltk.tokenize import word_tokenize

text_data = [
    "I love to eat pizza",
    "Pizza is my favorite food",
    "I could eat pizza every day",
    "Pizza toppings include cheese, pepperoni, and mushrooms",
    "What's your favorite pizza topping?"
]

# Function to perform data augmentation by swapping words randomly
def augment_data(text):
    tokens = word_tokenize(text)
    augmented_texts = []
    
    for _ in range(3):
        augmented_tokens = tokens.copy()
        for i in range(len(augmented_tokens)):
            if random.random() < 0.2:  # 20% probability of word swap
                j = random.randint(0, len(augmented_tokens)-1)
                augmented_tokens[i], augmented_tokens[j] = augmented_tokens[j], augmented_tokens[i]
        
        augmented_texts.append(" ".join(augmented_tokens))
    
    return augmented_texts

text_data_augmented = []
for text in text_data:
    text_data_augmented.append(text)
    text_data_augmented.extend(augment_data(text))

# Add more entries to the dataset
additional_entries = [
    "I enjoy trying different pizza flavors",
    "Pizza parties are always fun",
    "I can never resist a cheesy pizza",
    "What's your go-to pizza joint?",
    "Pizza night is the best night",
    "I like to experiment with pizza toppings",
    "I prefer thin-crust pizza over deep-dish",
    "Do you like pineapple on pizza?",
    "Pizza delivery is so convenient",
    "I'm always in the mood for pizza"
    # Add more entries of your choice
]

text_data_augmented.extend(additional_entries)

# Print the augmented dataset
for text in text_data_augmented:
    print(text)
