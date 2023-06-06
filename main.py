import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Przygotowanie danych treningowych
data = pd.read_csv('dane_treningowe.csv')  # Zakładamy, że dane treningowe są w formacie CSV
X = data['tekst']  # Kolumna z tekstami
y = data['gatunek']  # Kolumna z gatunkami filmów

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Wektoryzacja tekstu
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Trenowanie klasyfikatora
classifier = LinearSVC()
classifier.fit(X_train_vectors, y_train)

# Predykcja na danych testowych
predictions = classifier.predict(X_test_vectors)

# Ocena dokładności klasyfikatora
accuracy = accuracy_score(y_test, predictions)
print("Dokładność klasyfikatora: {:.2f}%".format(accuracy * 100))

# Wprowadzenie opisu filmu przez użytkownika
opis = input("Wprowadź opis filmu: ")

# Wektoryzacja opisu wprowadzonego przez użytkownika
opis_vector = vectorizer.transform([opis])

# Predykcja gatunku filmowego
prediction = classifier.predict(opis_vector)

print("Przewidywany gatunek filmu: ", prediction)
