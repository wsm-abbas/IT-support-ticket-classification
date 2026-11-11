"""Makes a small fake dataset of IT support tickets.

I can't share real tickets, so this builds them from a few phrases per
category and adds some noise (typos, extra words, mixed up categories) so
the problem isn't too easy. If you have real tickets, skip this file and
just put a CSV in data/tickets.csv with the columns: text, category.
"""
import random
import pandas as pd

random.seed(42)

PHRASES = {
    "network": [
        "cannot connect to the wifi", "vpn keeps dropping", "internet is very slow",
        "no network access on my desk", "cannot reach the shared drive",
        "ethernet port not working", "dns error when opening websites",
    ],
    "account_access": [
        "forgot my password", "account is locked", "cannot log in to my laptop",
        "need password reset", "two factor code not arriving",
        "my login stopped working after the weekend", "access denied to the finance folder",
    ],
    "hardware": [
        "laptop screen is flickering", "keyboard has stopped working",
        "printer is jammed", "mouse not responding", "monitor shows no signal",
        "laptop battery dies in 20 minutes", "headset has no sound",
    ],
    "software": [
        "excel keeps crashing", "need office installed", "outlook will not open",
        "software update failed", "application freezes on start",
        "cannot install the new version of teams", "browser shows a blank page",
    ],
    "email": [
        "not receiving emails", "mailbox is full", "emails stuck in outbox",
        "calendar invites not syncing", "spam is getting through",
        "cannot send attachments", "shared mailbox missing",
    ],
}

# which team gets the ticket
TEAM = {
    "network": "network_team",
    "account_access": "service_desk",
    "hardware": "desktop_support",
    "software": "desktop_support",
    "email": "service_desk",
}

FILLER = ["please help", "urgent", "this is the third time", "since this morning",
          "hi team", "thanks", "not sure what happened", "need this fixed today"]


def add_typo(text):
    words = text.split()
    i = random.randrange(len(words))
    w = words[i]
    if len(w) > 4:
        j = random.randrange(1, len(w) - 1)
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        words[i] = w
    return " ".join(words)


def make_ticket(category):
    text = random.choice(PHRASES[category])
    if random.random() < 0.5:
        text = random.choice(FILLER) + ", " + text
    if random.random() < 0.3:
        text = text + ", " + random.choice(FILLER)
    if random.random() < 0.25:
        text = add_typo(text)
    # sometimes the ticket mentions another problem as well, which is realistic
    if random.random() < 0.15:
        other = random.choice([c for c in PHRASES if c != category])
        text = text + " and also " + random.choice(PHRASES[other])
    return text


def main(n=3000):
    rows = []
    for _ in range(n):
        cat = random.choice(list(PHRASES))
        rows.append({"text": make_ticket(cat), "category": cat, "team": TEAM[cat]})
    df = pd.DataFrame(rows)
    df.to_csv("data/tickets.csv", index=False)
    print("saved", len(df), "tickets to data/tickets.csv")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
