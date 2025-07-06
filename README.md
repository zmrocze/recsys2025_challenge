
# Recsys2025

Link do konkursu:

 - https://github.com/Synerise/recsys2025
 - https://recsys.synerise.com/#challenge2025

## Zawartość

Głowne wyniki znajdują się w notebooku `src/recsys2025_wyniki.ipynb`, przeznaczonym do wykonania w google colab.
Notebook łączy się z google drive i oczekuje tam folderu `ubc_data_relevant_splitted` z danymi po processingu, gdy dojdzie do treningu to pyta o token do W&B.

Pozostałe notebooki obliczają eksploracyjne statystyki.

Kod modeli znajduje się w `src/our_lib/gat.py`.

Udostępniamy też finalne `scores` i `embeds`.
