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
    api.get("/teams").then(res => setTeams(res.data));
  }, []);

  const handlePredict = async () => {
    if (!teamA || !teamB) {
      alert("Choose both teams.");
      return;
    }
    try {
      const res = await axios.post("/predict", {
        teamA: teamA.name,
        teamB: teamB.name,
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
        historical match data since 2017/18. Some teams have fewer games in the
        first portuguese divison, which may affect prediction accuracy.
      </p>

      <div className="teams-boxes">
        <div className="team-container">
          <TeamBox team={teamA} />
          <select
            onChange={(e) =>
              setTeamA(teams.find((t) => t.name === e.target.value))
            }
            value={teamA?.name || ""}
          >
            <option value="">Choose home team</option>
            {teams
              .filter((c) => t.name !== teamB?.name)
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
            onChange={(e) =>
              setTeamB(teams.find((c) => t.name === e.target.value))
            }
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
          <h2>Predicted Winner: {result.prediction}</h2>
          <h3>Probabilities:</h3>
            <ul>
              {Object.entries(result.probabilities).map(([team, prob]) => (
                <li key={team}>
                  {team}: {prob * 100}.toFixed(2)%
                </li>
              ))}
            </ul>
            <h3>Key factors: </h3>
            <ul>
              {result.keyFactors.map((factor, idx) => (
                <li key={idx}>{factor}</li>
              ))}
            </ul>
            </div>
      )}
    </div>
  );
}
