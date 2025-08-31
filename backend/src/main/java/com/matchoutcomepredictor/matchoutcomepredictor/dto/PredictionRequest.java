package com.matchoutcomepredictor.dto;

public class PredictionRequest {
    private string homeTeam;
    private string awayTeam;

    public String getHomeTeam() { return homeTeam; }
    public void setHomeTeam(String homeTeam) { this.homeTeam = homeTeam; }

    public String getAwayTeam() { return awayTeam; }
    public void setawayTeam(String awayTeam) { this.awayTeam = awayTeam; }
}