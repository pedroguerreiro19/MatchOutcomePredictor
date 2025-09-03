export const factorTemplates = {
  "Elo Rating Difference": {
    emoji: "📈",
    positive: (teamA, teamB) => `The Elo rating suggests an advantage for ${teamA}.`,
    negative: (teamA, teamB) => `The Elo rating suggests an advantage for ${teamB}.`,
    neutral: () => "The Elo ratings suggest the teams are evenly matched."
  },
  "League Rank Difference": {
    emoji: "📊",
    positive: (teamA, teamB) => `The league position favors ${teamA}.`,
    negative: (teamA, teamB) => `The league position favors ${teamB}.`,
    neutral: () => "Both teams are similarly ranked in the league."
  },
  "Avg Goals Scored (last 10)": {
    emoji: "⚽",
    positive: (teamA, teamB) => `${teamA} has scored more goals on average in recent matches.`,
    negative: (teamA, teamB) => `${teamB} has scored more goals on average in recent matches.`,
    neutral: () => "Both teams have scored about the same recently."
  },
  "Avg Goals Conceded (last 10)": {
    emoji: "🛡️",
    positive: (teamA, teamB) => `${teamA} has conceded fewer goals on average in recent matches.`,
    negative: (teamA, teamB) => `${teamB} has conceded fewer goals on average in recent matches.`,
    neutral: () => "Both teams concede about the same number of goals."
  },
  "Avg Wins (last 10)": {
    emoji: "🏆",
    positive: (teamA, teamB) => `${teamA} has won more matches recently.`,
    negative: (teamA, teamB) => `${teamB} has won more matches recently.`,
    neutral: () => "Both teams have similar win rates."
  },
  "Avg Draws (last 10)": {
    emoji: "🤝",
    positive: (teamA, teamB) => `${teamA} tends to draw more often.`,
    negative: (teamA, teamB) => `${teamB} tends to draw more often.`,
    neutral: () => "Both teams have similar draw frequency."
  },
  "Avg Losses (last 10)": {
    emoji: "❌",
    positive: (teamA, teamB) => `${teamA} has lost fewer matches recently.`,
    negative: (teamA, teamB) => `${teamB} has lost fewer matches recently.`,
    neutral: () => "Both teams have similar loss records."
  },
  "Avg Points (last 10)": {
    emoji: "📈",
    positive: (teamA, teamB) => `${teamA} has earned more points on average in recent matches.`,
    negative: (teamA, teamB) => `${teamB} has earned more points on average in recent matches.`,
    neutral: () => "Both teams average about the same number of points."
  },
  "Avg Shots (last 10)": {
    emoji: "🥅",
    positive: (teamA, teamB) => `${teamA} creates more shooting opportunities.`,
    negative: (teamA, teamB) => `${teamB} creates more shooting opportunities.`,
    neutral: () => "Both teams take about the same number of shots."
  },
  "Avg Shots on Target (last 10)": {
    emoji: "🎯",
    positive: (teamA, teamB) => `${teamA} hits the target more often with their shots.`,
    negative: (teamA, teamB) => `${teamB} hits the target more often with their shots.`,
    neutral: () => "Both teams are equally accurate with their shots."
  },
  "Avg Fouls (last 10)": {
    emoji: "🛑",
    positive: (teamA, teamB) => `${teamA} commits fewer fouls on average.`,
    negative: (teamA, teamB) => `${teamB} commits fewer fouls on average.`,
    neutral: () => "Both teams commit a similar number of fouls."
  },
  "Avg Yellow Cards (last 10)": {
    emoji: "🟨",
    positive: (teamA, teamB) => `${teamA} receives fewer yellow cards on average.`,
    negative: (teamA, teamB) => `${teamB} receives fewer yellow cards on average.`,
    neutral: () => "Both teams receive about the same number of yellow cards."
  },
  "Avg Red Cards (last 10)": {
    emoji: "🟥",
    positive: (teamA, teamB) => `${teamA} has been more disciplined with fewer red cards.`,
    negative: (teamA, teamB) => `${teamB} has been more disciplined with fewer red cards.`,
    neutral: () => "Both teams have similar red card records."
  },
  "Avg Corners (last 10)": {
    emoji: "🚩",
    positive: (teamA, teamB) => `${teamA} wins more corners on average.`,
    negative: (teamA, teamB) => `${teamB} wins more corners on average.`,
    neutral: () => "Both teams win about the same number of corners."
  },
  "Head-to-Head Home Wins": {
    emoji: "🏠",
    positive: (teamA, teamB) => `${teamA} has dominated matches between the two teams at home.`,
    negative: (teamA, teamB) => `${teamB} has dominated matches between the two teams away.`,
    neutral: () => "Head-to-head results are balanced."
  },
  "Head-to-Head Away Wins": {
    emoji: "✈️",
    positive: (teamA, teamB) => `${teamA} tends to win more often when playing away in head-to-head.`,
    negative: (teamA, teamB) => `${teamB} tends to win more often when playing away in head-to-head.`,
    neutral: () => "Neither team has a clear away advantage in head-to-head."
  },
};