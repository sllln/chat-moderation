"""
Shows what the cleaning functions do:
  1) on a few hand-written examples,
  2) on a few random tweets from our training data.
"""
import pandas as pd

from text_cleaning import clean_full, clean_light, to_ascii

EXAMPLES = [
    "@USER s3n 4ptals1n",
    "a.p.t.a.l herif",
    "çoooook güzel oyundu!!!!",
    "IŞIK ve İSTANBUL",
    "Saat 3'te maç var, skor 2-1 bitti 2024",
    "bak şuna http://t.co/xyz #rezalet",
    "sen çok s.a.l.a.k.sın",
]


def show(text: str) -> None:
    print(f"original : {text}")
    print(f"light    : {clean_light(text)}")
    print(f"full     : {clean_full(text)}")
    print(f"ascii    : {to_ascii(clean_full(text))}")
    print()


print("===== Hand-written examples =====\n")
for example in EXAMPLES:
    show(example)

print("===== Random tweets from data/train.csv =====\n")
train = pd.read_csv("data/train.csv")
for tweet in train["text"].sample(5, random_state=1):
    show(tweet)