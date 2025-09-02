import { useState, useEffect } from "react";
import api from "./api";
import TeamBox from "./components/TeamBox";
import "./App.css";

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
        This project uses machine learning to predict the outcome of matches in
        the 2025/26 Portuguese League season. The model was trained using
        historical match data since 2010. Some teams have fewer games in the
        first portuguese divison, which may affect prediction accuracy.
      </p>

      <div className="teams-boxes">
        <div className="team-container">
          <TeamBox team={teamA} />
          <select
            onChange={(e) => {
              setTeamA(teams.find((t) => t.name === e.target.value))
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
              setTeamB(teams.find((t) => t.name === e.target.value))
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
            Predicted outcome:{ " "}
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
                if (outcome === "HomeWin") label = `${teamA.name} win probability`;
                else if (outcome === "AwayWin") label = `${teamB.name} win probability`;
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
      let text = "";
      let emoji = "";
      let cssClass = "factor-neutral";

      if (factor.feature.includes("League Rank")) {
        emoji = "📊";
        if (factor.impact > 0) {
          text = `The league position favors ${teamA.name}.`;
          cssClass = "factor-home";
        } else if (factor.impact < 0) {
          text = `The league position favors ${teamB.name}.`;
          cssClass = "factor-away";
        } else {
          text = "The league position is evenly matched.";
        }
      } else if (factor.feature.includes("Elo Rating")) {
        emoji = "📈";
        if (factor.impact > 0) {
          text = `The Elo rating suggests an advantage for ${teamA.name}.`;
          cssClass = "factor-home";
        } else if (factor.impact < 0) {
          text = `The Elo rating suggests an advantage for ${teamB.name}.`;
          cssClass = "factor-away";
        } else {
          text = "The Elo rating suggests the teams are evenly matched.";
        }
      } else if (factor.feature.includes("Avg Goals")) {
        emoji = "⚽";
        if (factor.impact < 0) {
          text = `${teamA.name} has scored more goals on average in the last 10 matches.`;
          cssClass = "factor-home";
        } else if (factor.impact > 0) {
          text = `${teamB.name} has scored more goals on average in the last 10 matches.`;
          cssClass = "factor-away";
        } else {
          text = "Both teams scored about the same in their last 10 matches.";
        }
      }

      return (
        <li key={idx} className={`factor-item ${cssClass}`}>
          {emoji} {text}
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
