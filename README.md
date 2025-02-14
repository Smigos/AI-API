AI API – Finomhangolt GPT-2 Chatbot

Ez egy egyszerű FastAPI chatbot, amely egy finomhangolt GPT-2 modellt használ. Egy szöveges bemenetet kap, és választ generál.

Beállítás és futtatás
---

AI_pretrained.py ezzel tudod traineltetni az AI-t kb 2 órát vesz időbe
AI_Run.py majd ezzel tudod futatni az AI-t később olvashatod hogyan

Anaconda környezet beállítása
---

Az egyszerűség kedvéért egy előre konfigurált Anaconda környezetet mellékelek. Kövesd az alábbi lépéseket a beállításhoz:

Nyisd meg az Anaconda Promptot

Navigálj ahhoz a mappához, ahol az environment fájl található

Futtasd a következő parancsot a környezet létrehozásához:

conda env create -f environment.yaml

Aktiváld a környezetet:

conda activate AIre

Mappák beállítása
---

Győződj meg róla, hogy a finomhangolt modell mappa elérhető (pl.: D:/AI_Stuff/Important/fine_tuned_gpt2/).

Ellenőrizd, hogy a run_api.py a megfelelő modell elérési utat használja.

Az API futtatása
---

Miután beléptél az Anaconda környezetbe, navigálj ahhoz a mappához, ahol a run_api.py található, és indítsd el a szervert:

uvicorn run_api:app --reload


Teszteld az API-t böngészőben: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Ellenőrizd, hogy az API fut-e: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

Hogyan használd
---

Figyelmeztetés: Jelenleg az AI hajlamos ismétlődő válaszokat adni, de ezen a problémán dolgozom a jobb teljesítmény érdekében.

Küldj egy POST kérést a /generate/ végpontra a következő JSON bemenettel:

Megjegyzés: Egyelőre az AI csak angol nyelven működik.

{
  "prompt": "Hello, how are you?"
}


Az API egy ilyen választ fog visszaadni:

{
  "response": "I'm doing well! How can I assist you?"
}

Ha megszeretnéd állítani akkor a promton belül egy CTRL+C segítségével megtudod tenni.

Hibaelhárítás
---

ModuleNotFoundError

Győződj meg róla, hogy az API-t a megfelelő Anaconda környezetben (AIre) futtatod.

Ha szükséges, frissítsd az Anaconda környezetet a következő paranccsal:

conda env update --file environment.yaml --prune


Model not found

Ellenőrizd, hogy a finomhangolt modell a megfelelő mappában van. Ha szükséges, módosítsd a run_api.py fájlban található elérési utat.

---

Készen is vagy! Jó szórakozást a chatbot használatához!
