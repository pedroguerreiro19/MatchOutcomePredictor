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

  useEffect(() => {
    api.get("/teams").then((res) => setTeams(res.data));
  }, []);

  const handlePredict = async () => {
    if (!teamA || !teamB) {
      alert("Choose both teams.");
      return;
    }
    try {
      const res = await api.post("/predict", {
        homeTeam: teamA.name,
        awayTeam: teamB.name,
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setResult({ prediction: "Error", probabilities: {}, keyFactors: [] });
    }
  };

  return (
    <div className="app">
      <h1 className="title">Portuguese League Match Outcome Predictor</h1>
      <p className="info-text">
        <p>
          This project uses a machine learning model trained on Portuguese
          League data, from 2010 till 2025. The model considers: team form (last 10
          games), gameplay stats (fouls, cards, corners, shots), league ranking,
          Elo ratings, and head-to-head results.
        </p>

        <p>
          <strong>How Elo works:</strong> Each team starts with a baseline
          rating. Beating a stronger team gives a large Elo boost, while beating
          a weaker team gives a smaller boost. Losing to a weaker team causes a
          big Elo drop, and draws slightly benefit the weaker side. Playing at
          home also provides a small Elo bonus.
        </p>

        <p>
          These features are combined and fed into an ensemble ML model that
          predicts the probability of home win, draw, or away win.
        </p>
      </p>

      <div className="teams-boxes">
        <div className="team-container">
          <TeamBox team={teamA} />
          <select
            onChange={(e) => {
              setTeamA(teams.find((t) => t.name === e.target.value));
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
          <TeamBox team={teamB} />
          <select
            onChange={(e) => {
              setTeamB(teams.find((t) => t.name === e.target.value));
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

      <button className="predict-btn" onClick={handlePredict}>
        Predict match
      </button>

      {result && (
        <div className="result">
          <h2>
            Predicted outcome:{" "}
            {result.prediction === "HomeWin"
              ? teamA.name + " victory 🏆"
              : result.prediction === "AwayWin"
              ? teamB.name + " victory 🏆"
              : "Draw 🤝"}
          </h2>
          <h3>Probabilities:</h3>
          <ul>
            {result.probabilities &&
              Object.entries(result.probabilities).map(([outcome, prob]) => {
                let label = outcome;
                if (outcome === "HomeWin")
                  label = `${teamA.name} win probability`;
                else if (outcome === "AwayWin")
                  label = `${teamB.name} win probability`;
                else label = "Draw";

                return (
                  <li key={outcome}>
                    {label}: {(prob * 100).toFixed(2)}%
                  </li>
                );
              })}
          </ul>

          <h3>Factors that influenced the prediction:</h3>
          <ul>
            {result.keyFactors && result.keyFactors.length > 0 ? (
              result.keyFactors.map((factor, idx) => {
                const template = factorTemplates[factor.feature];
                if (!template) {
                  return (
                    <li key={idx} className="factor-item">
                      ℹ️ {factor.feature}: no explanation available
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
                    {template.emoji} {text}
                  </li>
                );
              })
            ) : (
              <li>No key factors available</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
