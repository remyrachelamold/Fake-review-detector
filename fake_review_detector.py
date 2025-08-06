from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Real reviews (written by humans)
real_reviews = [
    "Battery backup is decent, lasts around 6 hours.",
    "The audio quality is crisp and clear, loved it!",
    "Fast delivery and great customer support.",
    "Good product for the price range.",
    "Packaging was nice and safe.",
    "Camera quality is okay for daylight shots.",
    "Build quality is strong, but a bit heavy.",
    "Customer care resolved my issue in 2 days.",
    "Installation was easy, no problems faced.",
    "Using it for a week, so far it's performing well.",
]

# Fake reviews (AI-generated style)
fake_reviews = [
    "Absolutely incredible! This changed my life!",
    "Wow wow wow!!! I cannot believe how perfect this is!",
    "Mind-blowing performance. Beyond expectations!",
    "Best product ever created in human history!",
    "100/10. Would recommend to the universe.",
    "This product is beyond magical!",
    "Insane value. I have no words. Unreal!",
    "I'm crying with happiness. Just buy it now!",
    "I would marry this product if I could!",
    "The most flawless invention in all of existence!",
]

# Combine
all_reviews = real_reviews + fake_reviews
labels = [0]*len(real_reviews) + [1]*len(fake_reviews)  # 0 = real, 1 = fake

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_reviews)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n--- MODEL REPORT ---")
print(classification_report(y_test, y_pred))

# Predict custom review
def detect_review(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    return "Fake Review ❌" if result[0] == 1 else "Real Review ✅"

# Example test
print("\n--- REVIEW CHECK ---")
print(detect_review("The camera quality is decent for the price."))