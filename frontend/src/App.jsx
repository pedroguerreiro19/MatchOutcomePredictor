import { useState, useEffect } from "react";
import axios from "axios";
import api from "./api";
import { clubs } from "./clubs";
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
      setResult(res.data.prediction);
    } catch (err) {
      console.error(err);
      setResult("Error predicting");
    }
  };

  return (
    <div className="app">
      {/* Título e Info */}
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
              setTeamA(clubs.find((c) => c.name === e.target.value))
            }
            value={teamA?.name || ""}
          >
            <option value="">Choose home team</option>
            {clubs
              .filter((c) => c.name !== teamB?.name)
              .map((club) => (
                <option key={club.name} value={club.name}>
                  {club.name}
                </option>
              ))}
          </select>
        </div>

        <div className="vs">VS</div>

        <div className="team-container">
          <TeamBox team={teamB} />
          <select
            onChange={(e) =>
              setTeamB(clubs.find((c) => c.name === e.target.value))
            }
            value={teamB?.name || ""}
          >
            <option value="">Choose away team</option>
            {clubs
              .filter((c) => c.name !== teamA?.name)
              .map((club) => (
                <option key={club.name} value={club.name}>
                  {club.name}
                </option>
              ))}
          </select>
        </div>
      </div>

      <button className="predict-btn" onClick={handlePredict}>
        Predict match
      </button>

      {/* Resultado */}
      {result && <div className="result">{result}</div>}
    </div>
  );
}
