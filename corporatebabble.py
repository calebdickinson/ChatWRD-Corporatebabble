import random
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk
from pygame import mixer
import tensorflow as tf
import numpy as np

class TextGenerator:
    def __init__(self, text_file, seed_text, num_epochs=50, batch_size=64, embedding_dim=64, rnn_units=128):
        with open(text_file, 'r') as file:
            text = file.read()
        self.word_seq, self.word_ids = self.preprocess_text(text)
        self.vocab_size = len(self.word_ids)
        self.seed_text = seed_text
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        
        self.model = self.build_model()
        self.id_words = {i: word for word, i in self.word_ids.items()}
        
    def preprocess_text(self, text):
        text = text.lower()
        words = text.split()
        word_ids = {word: i for i, word in enumerate(sorted(set(words)))}
        word_seq = [word_ids[word] for word in words]
        return word_seq, word_ids
    
    def build_model(self):
        timesteps = len(self.seed_text.split())
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=timesteps),
            tf.keras.layers.LSTM(self.rnn_units),
            tf.keras.layers.Dense(self.rnn_units, activation='relu'),
            tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def generate_text(self, num_words):
        generated_text = self.seed_text.split()
        timesteps = len(generated_text)
        for i in range(num_words):
            seed_seq = [self.word_ids.get(word, 0) for word in self.seed_text.lower().split()]
            seed_seq = np.array([seed_seq])
            predictions = self.model.predict(seed_seq)
            predictions = predictions[0]
            predicted_id = tf.argmax(predictions, axis=-1).numpy()
            generated_text.append(self.id_words.get(predicted_id, ' '))
            self.seed_text = ' '.join(generated_text[-timesteps:])
        words = (' '.join(generated_text))
        words_list = words.split()[timesteps:]
        words_list = [word.capitalize() if word == "i" else word for word in words_list]
        words_list = [word.replace(" .", ".") for word in words_list]
        punctuations = ['.', '?', '!']
        for i, word in enumerate(words_list):
            if i > 0 and words_list[i-1][-1] in punctuations:
                words_list[i] = word.capitalize()
        words_list[0] = words_list[0].capitalize()
        if words_list[-1][-1] == '.':
            return ' '.join(words_list)
        else:
            return ' '.join(words_list) + '.'

        
    def train_model(self):
        X_train = np.array([self.word_seq[i:i+len(self.seed_text.split())] for i in range(len(self.word_seq)-len(self.seed_text.split()))])
        y_train = np.array([self.word_seq[i+len(self.seed_text.split())] for i in range(len(self.word_seq)-len(self.seed_text.split()))])
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.vocab_size)
        
        target_accuracy = 0.90
        max_epochs = 100
        
        for epoch in range(max_epochs):
            self.model.fit(X_train, y_train, epochs=1, batch_size=self.batch_size, verbose=0)
            
            _, accuracy = self.model.evaluate(X_train, y_train, verbose=0)
            print("Epoch:", epoch, "Accuracy:", accuracy)
            
            if accuracy >= target_accuracy:
                print("Target accuracy reached!")
                break

class TypingText:
    def __init__(self, canvas, x, y, text='', delay=100, **kwargs):
        self.canvas = canvas
        self.delay = delay
        self.text = text
        self.index = 0
        self.typing = False
        self.text_id = canvas.create_text(x, y, text='', **kwargs)
        self.canvas.itemconfigure(self.text_id, fill='#ffffff')#should fill be red, or even another color?

    def do_typing(self):
        if self.index < len(self.text):
            self.canvas.itemconfigure(self.text_id, text=self.text[:self.index + 1])
        else:
            self.stop_typing()
            return None
        self.index += 1
        self.canvas.after(self.delay, self.do_typing)

    def start_typing(self):
        self.typing = True
        self.do_typing()

    def stop_typing(self):
        self.typing = False


mixer.init()
mixer.music.load('ChatWRD-Corporatebabble/at the circus (compressed to 20000 sample rate).wav')
mixer.music.play(loops=-1)
window = tk.Tk()
window.title('ChatWRD: CORPORATEBABBLE')
window.geometry('600x600')
canvas = tk.Canvas(window, width=600, height=600)
canvas.pack()
rand_num = random.randint(1, 31)
image_path = 'ChatWRD-Corporatebabble/creepy office space.jpg'
image = Image.open(image_path)
image = image.resize((600, 600))
tk_image = ImageTk.PhotoImage(image)
canvas.create_image(0, 0, anchor="nw", image=tk_image)
frame = tk.Frame(canvas, background='#ffffff')
frame.place(relx=0.5, rely=0.5, anchor="c")
label = tk.Label(frame, text='Enter some text:', font=('Helvetica', 14), foreground='#333333', background='#ffffff')
label.pack(pady=5)
text_box = tk.Entry(frame, font=('Helvetica', 12), borderwidth=0, background='#eeeeee')
text_box.pack(pady=5)
scrollbar = tk.Scrollbar(frame, orient='horizontal', command=text_box.xview)
scrollbar.pack(fill='x')
text_box.config(xscrollcommand=scrollbar.set)

options = [
    1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, #1, 1 looks like a mistake but it's actually a bug fix.
    12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
    55, 60, 65, 70, 80, 90, 100, 150, 200, 300
    ]
var = tk.IntVar(value=options[0])
style = ttk.Style()
style.configure('Custom.TMenubutton', font=('Helvetica', 12), foreground='#333333', background='#bbbbbb', borderwidth=0)
menu = ttk.OptionMenu(frame, var, *options, style='Custom.TMenubutton')
menu.pack(pady=10)

def button_click():
    user_input = text_box.get()
    print(f'User input: {user_input}')
    print(f'Menu option: {var.get()}')
    
    if __name__ == '__main__':
        generator = TextGenerator('ChatWRD-Corporatebabble/corporatebabble.txt', user_input)
        generator.train_model()
        generated_text = generator.generate_text(var.get())
        print(generated_text)
        text = TypingText(canvas, x=550, y=500, text=generated_text, font=('Arial', 15, 'bold'), anchor=tk.SE)
        text.start_typing()

submit_button = tk.Button(frame, text='Submit', command=button_click, font=('', 14), foreground='#ffffff', background='#333333', borderwidth=0)
submit_button.pack(pady=10)

window.mainloop()
