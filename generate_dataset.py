"""
generate_dataset.py
-------------------
Builds data/emails_synthetic.csv: realistic-looking e-mails (subject + body)
produced from templates with randomised fillers. It complements the real
SMS Spam Collection (data/sms_spam.csv) with longer, e-mail style text:
phishing, prizes, pharma, loans, crypto, fake invoices vs. work / personal
/ transactional mail.

    python generate_dataset.py            # 1,200 rows, seed 42
    python generate_dataset.py --n 3000 --seed 7
"""
import argparse
import random
import pandas as pd

R = random.Random()

def pick(xs): return R.choice(xs)
def amount(): return pick(["$500", "$1,000", "$2,500", "$10,000", "$50,000", "£750", "€1,200", "$1,000,000"])
def spam_url(): return pick(["http://claim-now.win/x", "http://bit.ly/3xYz9", "http://secure-login-verify.info/acct", "http://free-gift.top/offer", "www.cheap-meds-online.biz", "http://paypa1-support.co/verify"])
def good_url(): return pick(["https://meet.example.com/room/42", "https://docs.example.com/spec", "https://github.com/acme/app/pull/118", "https://calendar.example.com/event/9"])
def name(): return pick(["Priya", "Arun", "Meera", "John", "Sara", "David", "Kavya", "Rahul", "Emily", "Omar", "Lakshmi", "Tom"])
def day(): return pick(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
def time(): return pick(["9am", "10:30am", "2pm", "3:15pm", "4pm"])
def topic(): return pick(["Q3 roadmap", "onboarding doc", "budget review", "client demo", "sprint retro", "design mockups", "API migration", "hiring plan"])

SPAM = [
    ("You have WON {a}!!!", "Congratulations {n}! You were selected as the lucky winner of {a}. Claim your prize now: {u} . Offer expires in 24 hours!"),
    ("URGENT: verify your account", "Dear customer, we detected unusual activity. Your account will be suspended unless you verify your details immediately at {u}"),
    ("Final notice - payment failed", "Your payment could not be processed. Update your billing information within 24 hours to avoid service termination: {u}"),
    ("Cheap meds, no prescription", "Buy V1agra and C1alis at 90% off. No prescription needed, discreet shipping worldwide. Order today {u}"),
    ("Re: your loan is approved", "Good news! You are pre-approved for a {a} loan with no credit check. Apply now, limited time only: {u}"),
    ("Make {a} working from home", "Earn {a} per week from home with no experience. Thousands already joined. Click here to start: {u}"),
    ("Invest in crypto - 500% returns", "Our AI trading bot guarantees 500% returns in 7 days. Deposit now and double your bitcoin. Act fast, spots are limited! {u}"),
    ("Invoice #{k} attached - open now", "Please find the attached invoice for {a}. Open the attachment or click {u} to review. Failure to pay will lead to legal action."),
    ("Your package is on hold", "We tried to deliver your parcel but need a small fee of $2.99. Pay now at {u} or it will be returned to sender."),
    ("FREE gift card for you", "Dear winner, you have been chosen to receive a FREE {a} gift card. Just complete this short survey: {u}"),
    ("Confidential business proposal", "I am a barrister representing a deceased client who left {a}. I need your help to transfer the funds. Reply with your bank details and you will receive 30%."),
    ("Lose 20 pounds in 2 weeks!", "Miracle weight loss pill, doctors hate it! Buy 2 get 1 FREE. Click {u} before the offer ends tonight!!!"),
    ("Your mailbox is almost full", "Your email storage is 99% full. Upgrade now or lose your messages. Login here to verify: {u}"),
    ("Hot singles in your area", "Meet beautiful singles near you tonight. 100% free to join, no credit card. Sign up {u}"),
    ("Security alert: password expires today", "Your password expires today. Click {u} to keep your current password. Ignore this and your account will be locked."),
]
HAM = [
    ("Meeting moved to {d}", "Hi {n}, can we move the {t} sync to {d} at {tm}? Let me know if that works for you. Thanks!"),
    ("Notes from today's call", "Hi team, attaching the notes from today's {t} discussion. Action items are at the bottom. Please add comments in the doc: {g}"),
    ("Lunch tomorrow?", "Hey {n}, are you free for lunch tomorrow around 1pm? There's a new place near the office I've been wanting to try."),
    ("Re: {t}", "Thanks {n}, I reviewed the {t} and it looks good overall. I left a few small comments, nothing blocking. Happy to chat on {d}."),
    ("Your order has shipped", "Hello {n}, your order #{k} has shipped and should arrive by {d}. You can track it from your account page. Thank you for shopping with us."),
    ("Reminder: dentist appointment", "This is a reminder of your appointment on {d} at {tm}. Please reply to confirm or call the clinic to reschedule."),
    ("Pull request ready for review", "Hi {n}, I opened a PR for the {t}: {g}. Tests pass locally. Could you take a look when you have a minute?"),
    ("Happy birthday!", "Happy birthday {n}! Hope you have a wonderful day. Let's catch up over coffee this weekend."),
    ("Course schedule for next week", "Dear students, the class on {d} will start at {tm} in room 204. Please read chapter 5 before attending and bring your laptops."),
    ("Invoice for {t}", "Hi {n}, attached is the invoice for the {t} as discussed in our contract. Payment terms are net 30. Let me know if you need anything else."),
    ("Weekend plans", "Hey {n}, a few of us are going hiking on Saturday morning. Want to join? We'll leave around 7am and be back by lunch."),
    ("Interview confirmation", "Hi {n}, thank you for applying. We would like to invite you for an interview on {d} at {tm}. Please confirm your availability."),
    ("Your monthly statement is ready", "Your statement for this month is now available in your online banking portal. For your security we never ask for passwords by email."),
    ("Quick question about {t}", "Hi {n}, do you remember where we stored the {t} files? I can't find them in the shared drive. Thanks in advance!"),
    ("Team offsite agenda", "Hi all, here is the draft agenda for the offsite. Please send me dietary requirements by {d} so I can finalise the catering."),
]

def fill(tpl):
    subj, body = tpl
    ctx = dict(a=amount(), n=name(), u=spam_url(), g=good_url(), d=day(), tm=time(), t=topic(), k=R.randint(1000, 99999))
    s, b = subj.format(**ctx), body.format(**ctx)
    return f"Subject: {s}\n{b}"

def noisy(text, spam):
    if spam and R.random() < 0.35: text = text.upper() if R.random() < 0.3 else text + " !!!"
    if R.random() < 0.15: text = text.replace(".", "..", 1)
    if R.random() < 0.2: text += "\n\n" + pick(["Regards,\n" + name(), "Thanks,\n" + name(), "Best,\n" + name(), "-- Sent from my phone"])
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/emails_synthetic.csv")
    a = ap.parse_args()
    R.seed(a.seed)
    rows = []
    for i in range(a.n):
        spam = i % 2 == 0
        rows.append({"message": noisy(fill(pick(SPAM if spam else HAM)), spam),
                     "label": "spam" if spam else "ham", "source": "synthetic_email"})
    df = pd.DataFrame(rows).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows to {a.out}")

if __name__ == "__main__":
    main()
