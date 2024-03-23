import numpy as np
import json

def load_embeddings(embeddings_path):
    with open(embeddings_path, 'r') as f:
        embeddings = json.load(f)
    return {word: np.array(embeddings[word]) for word in embeddings}

embeddings = load_embeddings('embeddings.json')

happy_vector = embeddings['happy']
joy_vector = embeddings['joy']
sad_vector = embeddings['sad']
depressed_vector = embeddings['depressed']

happy_dot_joy = np.dot(happy_vector, joy_vector)
sad_dot_joy = np.dot(sad_vector, joy_vector)

happy_dot_depressed = np.dot(happy_vector, depressed_vector)
sad_dot_depressed = np.dot(sad_vector, depressed_vector)

print('happy dot joy:', happy_dot_joy)
print('sad dot joy:', sad_dot_joy)
print('-----------------')
print('happy dot depressed:', happy_dot_depressed)
print('sad dot depressed:', sad_dot_depressed)


sentence = "I was happy when I had fun"
sentence = sentence.split()
sentence_vector = np.mean([embeddings[word] for word in sentence], axis=0)

Q = sentence_vector
K_t = sentence_vector.T

attention = np.dot(Q, K_t)
print(attention)