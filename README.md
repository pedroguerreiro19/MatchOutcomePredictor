#  Portuguese league Match Outcome Predictor

This repository holds a machine learning project that predicts the outcome of Portuguese League football matches, trained with data since 2010.  

The model predicts the probability of **Home Win, Draw, or Away Win** for any selected match.



## Demo

<img width="653" height="936" alt="image" src="https://github.com/user-attachments/assets/377f618c-9a3c-4624-a83a-4ef387b01c7a" />


## Features

- **Machine Learning Model (XGBoost + Elo + Rolling Stats)**
- Features include:
    - Team form (last 10 games: goals, wins, losses, points, etc.)
    - Game stats (shots, fouls, corners, yellow/red cards etc.)
    - League rank differences
    - Elo ratings (dynamic rating system updated after every match)
    - Head-to-head history

- **Elo System Explanation:**
  - Each team starts with a baseline rating.
  - Beating a stronger opponent means a large Elo boost.
  - Beating a weaker opponent means a smaller Elo boost.
  - Losing to a weaker opponent means a large Elo drop.
  - Draws benefits weaker team, slightly penalizes stronger team.
  - Home wins, since the home team has advantage, means small Elo bonus.

- **Web App (React, JavaScript + CSS)**
  - Choose two teams and predict the outcome instantly.
  - Visual explanation of probabilities.
  - Factor breakdown: shows which features influenced the prediction.


## Tech Stack

- **Backend (Java + Springboot):** Provides the 25/26 portuguese league teams logos and names to the frontend, while also receiving from the ml service the data that was requested from the frontend, using REST API.
- **Frontend(React, JavaScript + CSS)**: Simple UI made for the user to choose the teams and visualize the prediction.
- **ML service(Python, Flask, scikit-learn, pandas, joblib):** Cleans and merges all datasets into one. Then prepares the data for the model to be trained. 

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/pedroguerreiro19/MatchOutcomePredictor.git
cd MatchOutcomePredictor
```

### 2. Install dependencies

Make sure you have **Python** installed.  
Then, create a virtual environment and install the required libraries:

```
pip install -r requirements.txt
```
## 3. Train the model

The training script builds features, computes Elo ratings, and fits an XGBoost classifier.

```
cd ml
python scripts/train_xgb.py
```
This will:

Train the model on historical data (data/clean/matches_P1_1011_2526.csv)

Evaluate accuracy, F1-score, and log loss

Save the trained model and metadata to models/model_xgb.pkl

## 4. Run the prediction service
Start the Flask backend that exposes the prediction API:

```
cd ml/scripts
python serve.py
```
The API will be available at http://localhost:8000.

Endpoints:

POST /predict (requires JSON) 
{ 
  "homeTeam": "Benfica", 
  "awayTeam": "Porto" 
} 
e. g.

If you wish so, this is enough to run the model. The frontend and backend aren´t necessesary if you wish to test it on Postman.
## 5. Run the backend 

In a separate terminal

```
cd backend
mvnw spring-boot:run     
````

The backend will run at http://localhost:8080

Endpoints:

GET /teams → returns the list of 2025/26 Portuguese League teams (with names + logos).

POST /predict → forwards prediction requests to the ML service and returns results to the frontend.

## 6. Run the frontend

In a separate terminal:

```
cd frontend
npm install
npm run dev
```
This launches the React web app at http://localhost:5173.

You can now:

- Pick two teams from dropdowns

- Get predicted probabilities.

- See factor breakdown with explanations.

## Project Structure
```php
MatchOutcomePredictor/
│
├── data/                   # Cleaned historical data (since 2010)
│   └── clean/
│
├── ml/                     # Machine learning service (Python)
│   ├── scripts/
│   │   ├── train_xgb.py    # Training script
│   │   ├── serve.py        # API service (Flask)
│   ├── clean_datasets.py   # Dataset cleaning
│   ├── unite_datasets.py   # Create main dataset
│   ├── feature_pipeline.py  # Feature engineering (Elo, rolling stats)
│   └── models/             # Saved models (.pkl via joblib)
│
├── backend/                # Backend service (Java + Spring Boot)
│   ├── src/
│   │   ├── main/java/...   # Controllers, Services, Config
│   │   └── main/resources/ # application.properties, configs
│
├── frontend/               # React app
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── utils/          # Factor templates, helpers
│   │   └── App.js          # Main UI logic
│   └── public/
│
└── README.md               # Project documentation
```
## Notes
Some teams (like Estrela, Vizela, etc.) have fewer seasons in the Primeira Liga, so their Elo ratings and stats are based on limited history. Predictions for them may be less reliable.

Elo is dynamic: beating a strong team (e.g., Porto) boosts your rating more than beating a newly promoted side.
Conversely, losing to a weak team penalizes you harder.

The model is trained up to Matchday 4 of the 2025/2026 season.




