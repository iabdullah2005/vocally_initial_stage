import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sqlite3
import uuid
import nltk
import spacy
import whisper
import language_tool_python
import torch
import librosa
import numpy as np
from datetime import datetime
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from pydub import AudioSegment

# ------------------ MODEL SETUP ------------------
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")
whisper_model = whisper.load_model("tiny")
tool = language_tool_python.LanguageTool('en-US')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
model = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

DB_NAME = "Speech_Recognition.db"

def connect_db():
    return sqlite3.connect(DB_NAME)

def setup_database():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")

    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS audio_files (
        file_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        file_name TEXT,
        upload_date TEXT,
        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS transcriptions (
        transcription_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER,
        text TEXT,
        FOREIGN KEY(file_id) REFERENCES audio_files(file_id) ON DELETE CASCADE
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS grammar_analysis (
        grammar_id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcription_id INTEGER,
        issue TEXT,
        suggestion TEXT,
        FOREIGN KEY(transcription_id) REFERENCES transcriptions(transcription_id) ON DELETE CASCADE
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS pos_tags (
        pos_id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcription_id INTEGER,
        token TEXT,
        pos_tag TEXT,
        full_tag TEXT,
        FOREIGN KEY(transcription_id) REFERENCES transcriptions(transcription_id) ON DELETE CASCADE
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS emotions (
        emotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcription_id INTEGER,
        emotion TEXT,
        fluency_score REAL,
        wpm REAL,
        FOREIGN KEY(transcription_id) REFERENCES transcriptions(transcription_id) ON DELETE CASCADE
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS voice_qualities (
        vq_id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcription_id INTEGER,
        pitch REAL,
        volume REAL,
        clarity REAL,
        score REAL,
        FOREIGN KEY(transcription_id) REFERENCES transcriptions(transcription_id) ON DELETE CASCADE
    )''')

    conn.commit()
    conn.close()

setup_database()

# ------------------ START GUI ------------------
root = tk.Tk()
root.title("Vocalli")
root.geometry("1000x700")

notebook = ttk.Notebook(root)
tab_users = ttk.Frame(notebook)
tab_audio = ttk.Frame(notebook)
tab_transcription = ttk.Frame(notebook)
tab_grammar = ttk.Frame(notebook)
tab_pos = ttk.Frame(notebook)
tab_emotion = ttk.Frame(notebook)
tab_summary = ttk.Frame(notebook)

notebook.add(tab_users, text="Users")
notebook.add(tab_audio, text="Upload Audio")
notebook.add(tab_transcription, text="Transcriptions")
notebook.add(tab_grammar, text="Grammar Analysis")
notebook.add(tab_pos, text="POS Tagging")
notebook.add(tab_emotion, text="Emotion & Fluency")
notebook.add(tab_summary, text="Voice Quality Summary")
notebook.pack(expand=True, fill="both")

# ------------------ USERS TAB ------------------
user_tree = ttk.Treeview(tab_users, columns=("ID", "Name", "Email"), show="headings")
for col in ("ID", "Name", "Email"):
    user_tree.heading(col, text=col)
user_tree.pack(expand=True, fill="both", padx=10, pady=10)

name_var = tk.StringVar()
email_var = tk.StringVar()
form_frame = tk.Frame(tab_users)
form_frame.pack(pady=5)
tk.Label(form_frame, text="Name").grid(row=0, column=0)
tk.Entry(form_frame, textvariable=name_var).grid(row=0, column=1)
tk.Label(form_frame, text="Email").grid(row=1, column=0)
tk.Entry(form_frame, textvariable=email_var).grid(row=1, column=1)

def refresh_users():
    for i in user_tree.get_children(): user_tree.delete(i)
    conn = connect_db()
    rows = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    for row in rows:
        user_tree.insert("", "end", values=row)

def add_user():
    name, email = name_var.get(), email_var.get()
    if name and email:
        conn = connect_db()
        conn.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
        conn.commit()
        conn.close()
        refresh_users()
        name_var.set("")
        email_var.set("")

def update_user():
    selected = user_tree.selection()
    if selected:
        uid = user_tree.item(selected[0])['values'][0]
        new_name = name_var.get()
        new_email = email_var.get()
        if new_name and new_email:
            conn = connect_db()
            conn.execute("UPDATE users SET name = ?, email = ? WHERE user_id = ?", (new_name, new_email, uid))
            conn.commit()
            conn.close()
            refresh_users()


def delete_user():
    selected = user_tree.selection()
    if selected:
        uid = user_tree.item(selected[0])['values'][0]
        conn = connect_db()
        conn.execute("DELETE FROM users WHERE user_id = ?", (uid,))
        conn.commit()
        conn.close()
        refresh_users()

btn_frame = tk.Frame(tab_users)
btn_frame.pack()
tk.Button(btn_frame, text="Add User", command=add_user).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Update Selected", command=update_user).grid(row=1, column=1, padx=9)
tk.Button(btn_frame, text="Delete Selected", command=delete_user).grid(row=0, column=1, padx=5)

refresh_users()

# ------------------ AUDIO UPLOAD TAB ------------------
audio_user_var = tk.StringVar()
audio_users = {}

def refresh_audio_users():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, name FROM users")
    records = cursor.fetchall()
    conn.close()
    audio_users.clear()
    for uid, name in records:
        key = f"{uid} - {name}"
        audio_users[key] = uid
    audio_user_menu['values'] = list(audio_users.keys())

def process_audio():
    filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.mp4 *.wav")])
    if not filepath: return

    user_text = audio_user_var.get()
    if not user_text or user_text not in audio_users:
        messagebox.showerror("Select User", "Please select a user before uploading audio.")
        return

    uid = audio_users[user_text]
    filename = os.path.basename(filepath)
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.mp3', '.mp4']:
        audio = AudioSegment.from_file(filepath)
        wav_path = filepath.replace(ext, f"_{uuid.uuid4().hex}.wav")
        audio.export(wav_path, format="wav")
    else:
        wav_path = filepath

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO audio_files (user_id, file_name, upload_date) VALUES (?, ?, ?)", (uid, filename, upload_date))
    file_id = cursor.lastrowid

    result = whisper_model.transcribe(wav_path)
    text = result['text']
    cursor.execute("INSERT INTO transcriptions (file_id, text) VALUES (?, ?)", (file_id, text))
    transcription_id = cursor.lastrowid

    for match in tool.check(text):
        cursor.execute("INSERT INTO grammar_analysis (transcription_id, issue, suggestion) VALUES (?, ?, ?)",
                       (transcription_id, match.message, ', '.join(match.replacements)))

    doc = nlp(text)
    for token in doc:
        cursor.execute("INSERT INTO pos_tags (transcription_id, token, pos_tag, full_tag) VALUES (?, ?, ?, ?)",
                       (transcription_id, token.text, token.pos_, token.tag_))

    y, sr = librosa.load(wav_path, sr=16000)
    input_values = feature_extractor(y, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    emotion = model.config.id2label[int(logits.argmax())]

    duration = librosa.get_duration(y=y, sr=sr)
    word_count = len(text.split())
    wpm = word_count / (duration / 60) if duration else 0
    fluency = min(1.0, wpm / 150)
    pitch = float(np.mean(librosa.piptrack(y=y, sr=sr)[0][librosa.piptrack(y=y, sr=sr)[0] > 0]))
    volume = float(np.mean(librosa.feature.rms(y=y)))
    clarity = float(np.std(y))

    cursor.execute("INSERT INTO emotions (transcription_id, emotion, fluency_score, wpm) VALUES (?, ?, ?, ?)",
                   (transcription_id, emotion, fluency, wpm))
    cursor.execute("INSERT INTO voice_qualities (transcription_id, pitch, volume, clarity, score) VALUES (?, ?, ?, ?, ?)",
                   (transcription_id, pitch, volume, clarity, fluency))

    conn.commit()
    conn.close()
    messagebox.showinfo("Success", "Audio processed and results saved successfully.")
    refresh_transcriptions()
    refresh_grammar()
    refresh_pos()
    refresh_emotion()
    refresh_summary()

audio_frame = tk.Frame(tab_audio)
audio_frame.pack(pady=20)
tk.Label(audio_frame, text="Select User:").grid(row=0, column=0, padx=5)
audio_user_menu = ttk.Combobox(audio_frame, textvariable=audio_user_var, width=40)
audio_user_menu.grid(row=0, column=1, padx=5)
tk.Button(audio_frame, text="Upload and Analyze Audio", command=process_audio).grid(row=0, column=2, padx=10)
tk.Button(audio_frame, text="Refresh User List", command=refresh_audio_users).grid(row=0, column=3, padx=10)
refresh_audio_users()

refresh_audio_users()

# ------------------ RESULTS TABS ------------------

# Transcriptions Tab
trans_tree = ttk.Treeview(tab_transcription, columns=("File ID", "Transcription"), show="headings")
trans_tree.heading("File ID", text="File ID")
trans_tree.heading("Transcription", text="Transcription")
trans_tree.pack(expand=True, fill="both")

def refresh_transcriptions():
    for row in trans_tree.get_children(): trans_tree.delete(row)
    conn = connect_db()
    for row in conn.execute("SELECT file_id, text FROM transcriptions ORDER BY transcription_id DESC LIMIT 1"):
        trans_tree.insert("", "end", values=row)
    conn.close()

refresh_transcriptions()

# Grammar Tab
grammar_tree = ttk.Treeview(tab_grammar, columns=("Transcription ID", "Issue", "Suggestion"), show="headings")
grammar_tree.heading("Transcription ID", text="Transcription ID")
grammar_tree.heading("Issue", text="Issue")
grammar_tree.heading("Suggestion", text="Suggestion")
grammar_tree.pack(expand=True, fill="both")

def refresh_grammar():
    for row in grammar_tree.get_children(): grammar_tree.delete(row)
    conn = connect_db()
    for row in conn.execute("SELECT transcription_id, issue, suggestion FROM grammar_analysis ORDER BY grammar_id DESC LIMIT 10"):
        grammar_tree.insert("", "end", values=row)
    conn.close()

refresh_grammar()

# POS Tab
pos_tree = ttk.Treeview(tab_pos, columns=("Transcription ID", "Token", "POS", "Tag"), show="headings")
pos_tree.heading("Transcription ID", text="Transcription ID")
pos_tree.heading("Token", text="Token")
pos_tree.heading("POS", text="POS")
pos_tree.heading("Tag", text="Tag")
pos_tree.pack(expand=True, fill="both")

def refresh_pos():
    for row in pos_tree.get_children(): pos_tree.delete(row)
    conn = connect_db()
    for row in conn.execute("SELECT transcription_id, token, pos_tag, full_tag FROM pos_tags ORDER BY pos_id DESC LIMIT 10"):
        pos_tree.insert("", "end", values=row)
    conn.close()

refresh_pos()

# Emotion Tab
emo_tree = ttk.Treeview(tab_emotion, columns=("Transcription ID", "Emotion", "Fluency", "WPM"), show="headings")
emo_tree.heading("Transcription ID", text="Transcription ID")
emo_tree.heading("Emotion", text="Emotion")
emo_tree.heading("Fluency", text="Fluency")
emo_tree.heading("WPM", text="WPM")
emo_tree.pack(expand=True, fill="both")

def refresh_emotion():
    for row in emo_tree.get_children(): emo_tree.delete(row)
    conn = connect_db()
    for row in conn.execute("SELECT transcription_id, emotion, fluency_score, wpm FROM emotions ORDER BY emotion_id DESC LIMIT 1"):
        emo_tree.insert("", "end", values=row)
    conn.close()

refresh_emotion()

# Summary Tab
sum_tree = ttk.Treeview(tab_summary, columns=("Transcription ID", "Pitch", "Volume", "Clarity", "Score"), show="headings")
sum_tree.heading("Transcription ID", text="Transcription ID")
sum_tree.heading("Pitch", text="Pitch")
sum_tree.heading("Volume", text="Volume")
sum_tree.heading("Clarity", text="Clarity")
sum_tree.heading("Score", text="Score")
sum_tree.pack(expand=True, fill="both")

def refresh_summary():
    for row in sum_tree.get_children(): sum_tree.delete(row)
    conn = connect_db()
    for row in conn.execute("SELECT transcription_id, pitch, volume, clarity, score FROM voice_qualities ORDER BY vq_id DESC LIMIT 1"):
        sum_tree.insert("", "end", values=row)
    conn.close()

refresh_summary()

root.mainloop()
