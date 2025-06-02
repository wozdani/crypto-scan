#!/bin/bash

echo "🌀 Przygotowuję kod do pushowania..."

# Dodaj tylko kod, nie dane ani raporty
git add .
git add utils/
git add stages/
git add .gitignore
git add .gitattributes

echo "✍️ Podaj opis commita:"
read COMMIT_MSG

git commit -m "$COMMIT_MSG"
git push

echo "✅ Kod został wysłany na GitHub!"
