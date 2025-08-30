import "../App.css";

export default function TeamBox({ team }) {
  return (
    <div className="team-box">
      {team ? (
        <div className="team-content">
          <img src={team.logo} alt={team.name} className="team-logo" />
          <p>{team.name}</p>
        </div>
      ) : (
        <p className="placeholder">No team selected</p>
      )}
    </div>
  );
}