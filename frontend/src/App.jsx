import { useState, useEffect } from "react";
import api from "./api";
import TeamBox from "./components/TeamBox";
import "./App.css";
import { factorTemplates } from "./utils/factorTemplates";

export default function App() {
  const [teams, setTeams] = useState([]);
  const [teamA, setTeamA] = useState(null);
  const [teamB, setTeamB] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    api.get("/teams").then((res) => setTeams(res.data));
  }, []);

  const handlePredict = async () => {
    if (!teamA || !teamB) {
      alert("Please select both teams.");
      return;
    }
    
    setIsLoading(true);
    try {
      const res = await api.post("/predict", {
        homeTeam: teamA.name,
        awayTeam: teamB.name,
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setResult({ 
        prediction: "Error", 
        probabilities: {}, 
        keyFactors: [] 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getPredictionText = (prediction) => {
    switch (prediction) {
      case "HomeWin":
        return `${teamA.name} Victory`;
      case "AwayWin":
        return `${teamB.name} Victory`;
      case "Draw":
        return "Draw";
      default:
        return "Unknown Result";
    }
  };

  const getProbabilityLabel = (outcome) => {
    switch (outcome) {
      case "HomeWin":
        return `${teamA.name} Win`;
      case "AwayWin":
        return `${teamB.name} Win`;
      case "Draw":
        return "Draw";
      default:
        return outcome;
    }
  };

  return (
    <div className="app">
      <div className="container">
        <div className="header">
          <h1 className="title">Portuguese League Match Predictor</h1>
        </div>
        
        <div className="info-text">
          <p>
            This project uses a machine learning model trained on Portuguese
            League data from 2010 to 2025. The model considers team form (last 10
            games), gameplay stats (fouls, cards, corners, shots), league ranking,
            ELO ratings, and head-to-head results.
          </p>

          <p>
            <strong>How ELO works:</strong> Each team starts with a baseline
            rating. Beating a stronger team gives a large ELO boost, while beating
            a weaker team gives a smaller boost. Losing to a weaker team causes a
            big ELO drop, and draws slightly benefit the weaker side. Playing at
            home also provides a small ELO bonus.
          </p>

          <p>
            These features are combined and fed into an ensemble ML model that
            predicts the probability of home win, draw, or away win.
          </p>
        </div>

        <div className="teams-boxes">
          <div className="team-container">
            <div className="team-box">
              <div className="team-content">
                {teamA ? (
                  <>
                    <div className="team-logo-container">
                      <img 
                        src={`http://localhost:8080${teamA.logo}`} 
                        alt={`${teamA.name} logo`}
                        className="team-logo" 
                      />
                    </div>
                    <div className="team-name">{teamA.name}</div>
                  </>
                ) : (
                  <>
                    <div className="team-logo-container">
                      <div className="placeholder">🏠</div>
                    </div>
                    <div className="team-name">Select Home Team</div>
                  </>
                )}
              </div>
            </div>
            <select
              onChange={(e) => {
                setTeamA(teams.find((t) => t.name === e.target.value) || null);
                setResult(null);
              }}
              value={teamA?.name || ""}
            >
              <option value="">Choose home team</option>
              {teams
                .filter((t) => t.name !== teamB?.name)
                .map((team) => (
                  <option key={team.name} value={team.name}>
                    {team.name}
                  </option>
                ))}
            </select>
          </div>

          <div className="vs">VS</div>

          <div className="team-container">
            <div className="team-box">
              <div className="team-content">
                {teamB ? (
                  <>
                    <div className="team-logo-container">
                      <img 
                        src={`http://localhost:8080${teamB.logo}`} 
                        alt={`${teamB.name} logo`}
                        className="team-logo" 
                      />
                    </div>
                    <div className="team-name">{teamB.name}</div>
                  </>
                ) : (
                  <>
                    <div className="team-logo-container">
                      <div className="placeholder">✈️</div>
                    </div>
                    <div className="team-name">Select Away Team</div>
                  </>
                )}
              </div>
            </div>
            <select
              onChange={(e) => {
                setTeamB(teams.find((t) => t.name === e.target.value) || null);
                setResult(null);
              }}
              value={teamB?.name || ""}
            >
              <option value="">Choose away team</option>
              {teams
                .filter((t) => t.name !== teamA?.name)
                .map((team) => (
                  <option key={team.name} value={team.name}>
                    {team.name}
                  </option>
                ))}
            </select>
          </div>
        </div>

        <button 
          className="predict-btn" 
          onClick={handlePredict}
          disabled={!teamA || !teamB || isLoading}
        >
          {isLoading ? "Predicting..." : "Predict Match"}
        </button>

        {result && (
          <div className="result">
            <h2>{getPredictionText(result.prediction)}</h2>
            
            <h3>Probabilities</h3>
            <ul>
              {result.probabilities &&
                Object.entries(result.probabilities).map(([outcome, prob]) => (
                  <li key={outcome}>
                    <span>{getProbabilityLabel(outcome)}</span>
                    <span className="probability-value">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </li>
                ))}
            </ul>

            <h3>Key Factors</h3>
            <ul>
              {result.keyFactors && result.keyFactors.length > 0 ? (
                result.keyFactors.map((factor, idx) => {
                  const template = factorTemplates[factor.feature];
                  if (!template) {
                    return (
                      <li key={idx} className="factor-item">
                        <span className="factor-emoji">ℹ️</span>
                        <span>{factor.feature}: No explanation available</span>
                      </li>
                    );
                  }

                  let text;
                  if (factor.impact > 0.01)
                    text = template.positive(teamA.name, teamB.name);
                  else if (factor.impact < -0.01)
                    text = template.negative(teamA.name, teamB.name);
                  else text = template.neutral(teamA.name, teamB.name);

                  return (
                    <li key={idx} className="factor-item">
                      <span className="factor-emoji">{template.emoji}</span>
                      <span>{text}</span>
                    </li>
                  );
                })
              ) : (
                <li className="factor-item">
                  <span className="factor-emoji">❔</span>
                  <span>No key factors available</span>
                </li>
              )}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}