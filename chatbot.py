import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

import json, pickle, random, argparse, re
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

lemmatizer = WordNetLemmatizer()

def clean_text(t): return re.sub(r"[^\w\s]", "", t).lower()

intents = json.load(open("job_intents.json", encoding="utf-8"))

words, classes, docs = [], [], []
for it in intents["intents"]:
    tag = it["tag"]
    if tag not in classes: classes.append(tag)
    for p in it["patterns"]:
        cleaned = clean_text(p)
        toks = wordpunct_tokenize(cleaned)
        words.extend(toks)
        docs.append((toks, tag))

words = sorted(list(set([lemmatizer.lemmatize(w) for w in words if w.strip()])))
classes = sorted(list(set(classes)))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
out_empty = [0] * len(classes)
for toks, tag in docs:
    pattern = [lemmatizer.lemmatize(w.lower()) for w in toks]
    bag = [1 if w in pattern else 0 for w in words]
    row = list(out_empty)
    row[classes.index(tag)] = 1
    training.append([bag, row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:,0]))
train_y = np.array(list(training[:,1]))

model = Sequential([
    Dense(256, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(train_y[0]), activation='softmax')
])
opt = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
hist = model.fit(train_x, train_y, epochs=args.epochs, batch_size=args.batch_size, verbose=1, callbacks=[es])

model.save("chatbot_model.h5")
pickle.dump(hist.history, open("history.pkl", "wb"))
print("✅ Trained & saved chatbot_model.h5")
