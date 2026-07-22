# My Gold Price Predictor API (XAU/USD Classifier)

Hey! This is my project for the BoltIQ technical assessment. I built a machine learning pipeline that pulls historical data for gold, processes it into mathematical features, trains a model to predict if tomorrow's candle will close up or down, and wraps it up into a working FastAPI endpoint. 


##  What the Repo Looks Like

I kept everything organized exactly asked:

```text
xauusd-classifier/
├── data/
│   └── fetch_data.py          # Script that downloads the raw data
├── features/
│   └── engineer.py           # The math for the 5 features
├── model/
│   ├── train.py              # Training script & temporal split
│   └── model.pkl             # My saved model file (the brain)
├── api/
│   └── main.py               # The FastAPI server script
├── notebooks/
│   └── bolt-iq.ipynb         # My main sandbox notebook where I built everything
├── requirements.txt          # App dependencies
└── README.md                 # This file right here!
```

---

## 🧮 Breaking Down the 5 Features (The Math)

I didn't want to use sketchy external libraries that break, so I coded all these features manually using raw Pandas and NumPy logic:

1. **Previous Candle Return**: `(Close - Close.shift(1)) / Close.shift(1)`  
   Tracks the percentage move from yesterday to today so the model can gauge immediate market speed.
2. **High-Low Range as a % of Close**: `(High - Low) / Close`  
   Measures how crazy the price action was inside a single day. Big candle range = high drama between buyers and sellers.
3. **SMA Difference**: `5-period SMA minus 20-period SMA`  
   Tracks trend acceleration. If the fast 5-day line crosses above the slow 20-day line, it usually signals that the price is speeding up to the upside.
4. **RSI (14-Period)**: `100 - (100 / (1 + RS))`  
   A classic momentum oscillator. It scores the market between 0 and 100 to show if gold is overbought (too high, over 70) or oversold (too low, under 30).
5. **ATR (14-Period)**: `14-day average of True Range`  
   This tracks total volatility, including overnight price gaps. It tells the model *how hard* the market is moving, regardless of the direction.


##  Avoiding the "Look-Ahead" Trap

* **The Target Column (`direction`)**: I looked at tomorrow's close price by using `data['Close'].shift(-1)`. If tomorrow's close was higher than today's close, it gets a `1` (Up). If lower, a `0` (Down). I dropped the very last row because tomorrow hasn't happened yet!
* **No Leakage**: To make sure the model didn't cheat, the features *only* use past or present data (`shift(1)`), while the target column is the only one that looks ahead (`shift(-1)`).
* **The Time Split**: I didn't use standard random shuffling because this is time-series data. If you shuffle, the model peeks at future data points during training. I drew a hard line in time: trained on the first 80% of rows chronologically, and tested it blindly on the final 20%.

---

## Honest Stats & Fixing the "Bullish Bias"

### Try 1: The Trap
When I first ran my Random Forest model, my accuracy was around **46.53%**, but my Recall was super high (~80%) and my Precision was low (~45%). 

**What went wrong:** Because gold went up for a lot of the training data (57% up days vs 43% down days), the model got lazy. It learned that it could score okay just by guessing "UP" almost every single time. This is a massive bias that will ruin a real trading account during a market crash.

### Try 2: The Balance Fix
To force the model to be objective, I added `class_weight='balanced'` to the Random Forest initialization. This hits the model with a much harsher score penalty if it messes up a down day prediction during training.

### Final Balanced Results:
* **Directional Accuracy**: 44.55%
* **Precision (Up Days)**: 43.06%
* **Recall (Up Days)**: 67.39%
* **F1 Score**: 52.54%

### The Confusion Matrix:
```text
[[14 41]   -> Top Row: Actual DOWN days (14 guessed right, 41 guessed wrong)
 [15 31]]  -> Bottom Row: Actual UP days (15 guessed wrong, 31 guessed right)
```
**My honest thoughts on this:** The model still falls for false alarms (41 False Positives). An accuracy under 50% on the test set is completely realistic for noisy financial data, proving there's no data leakage. If I were deploying this live, I wouldn't just take the raw 1 or 0 prediction. I'd use `predict_proba()` and tell the API to only place a trade if the model is over 60% confident.

---

##  Major Hurdles I Faced & What I Learned

Since I'm pretty new to software engineering and building APIs, I ran into a ton of annoying roadblocks. Getting through them taught me way more than just writing basic notebook code:

1. **The MultiIndex Column Headaches**  
   * *The Problem*: When `yfinance` downloaded the data, it formatted the columns as a messy multi-tiered structure like `(Close, GC=F)`. My feature math code immediately crashed with key errors.
   * *What I Learned*: I learned how to directly overwrite the DataFrame's columns using a clean, flat list array `['Close', 'High', 'Low', 'Open', 'Volume']`. This wiped out the ticker clutter and made my pipeline asset-agnostic.

2. **The `pandas-ta` Package Failure**  
   * *The Problem*: I tried installing an external tech-analysis library to do my RSI and ATR math, but my environment kept throwing dependency errors because of version conflicts with modern NumPy.
   * *What I Learned*: Instead of giving up or messing up my environment, I looked up the raw mathematical equations and coded the RSI and ATR formulas completely by hand using native Pandas vector operations. It made my pipeline lighter and less reliant on broken packages.

3. **The Disappearing `model.pkl` File**  
   * *The Problem*: My notebook ran perfectly, but when I tried starting my FastAPI server, Uvicorn kept screaming `FileNotFoundError: No such file or directory: 'model/model.pkl'`. 
   * *What I Learned*: I realized that when I opened my project folder workspace inside VS Code, my Jupyter Notebook kernel was running out of my root user directory (`C:\Users\HP`), while the API terminal terminal was locked inside the subfolder. I solved this by replacing hardcoded strings in `main.py` with dynamic absolute path routing using Python's native `os.path` tools.

4. **The Command Line Disconnect**  
   * *The Problem*: Even after locating the file, my API server crashed on launch with a `ModuleNotFoundError: No module named 'sklearn'`. I was super confused because scikit-learn worked fine in my notebook.
   * *What I Learned*: I learned the big difference between an active Jupyter Notebook kernel environment and a standalone terminal shell instance. I had to explicitly run `pip install scikit-learn` in the VS Code terminal command line to get the API's Python engine up to speed.

---

## 🚀 How to Run My Project Locally

### 1. Grab the Packages
Open your terminal in the project's root folder and run:
```bash
pip install -r requirements.txt
```

### 2. Train the Brain
Open up the `notebooks/bolt-iq.ipynb` file in VS Code and hit **"Run All"**. This downloads the fresh gold data, runs all my manual feature math, and saves a fresh model file into `model/model.pkl`.

### 3. Start the Web Server
Jump into your terminal and boot up the FastAPI app using Uvicorn:
```bash
uvicorn api.main:app --reload
```
You should see a message saying: `INFO: Uvicorn running on http://127.0.0.1:8000`

### 4. Play with the Interactive Docs
Go to your browser and type in:
```text
http://127.0.0.1:8000/docs
```
This opens up FastAPI's awesome Swagger UI interface. Click on **POST /predict**, hit **"Try it out"**, change the dummy JSON feature numbers to whatever market parameters you want, and click the blue **Execute** button. It will feed those numbers to my saved model and print out a clean prediction response (`0` or `1`) instantly!
