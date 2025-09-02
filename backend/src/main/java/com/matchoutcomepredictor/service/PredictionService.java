package com.matchoutcomepredictor.service;

import com.matchoutcomepredictor.dto.PredictionRequest;
import com.matchoutcomepredictor.dto.PredictionResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Service
public class PredictionService {

    private final RestTemplate restTemplate;

    public PredictionService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public PredictionResponse getPrediction(PredictionRequest request) {
        try {
            String url = "http://localhost:8000/predict";

            Map<String, String> payload = Map.of(
                "home_team", request.getHomeTeam(),
                "away_team", request.getAwayTeam()
            );

            Map response = restTemplate.postForObject(url, payload, Map.class);

            if (response == null) {
                throw new RuntimeException("Empty response from ML service.");
            }

            String winner = (String) response.get("winner");
            Map<String, Double> probabilities = (Map<String, Double>) response.get("probabilities");
            List<String> keyFactors = (List<String>) response.get("keyFactors");

            return new PredictionResponse(winner, probabilities, keyFactors);
        } catch (Exception e) {
            e.printStackTrace();

            Map<String, Double> fallbackProbs = Map.of(
                "HomeWin", 0.0,
                "AwayWin", 0.0,
                "Draw", 0.0
            );
            List<String> fallbackFactors = List.of("Fallback: ML service unavailable");

            return new PredictionResponse("Unknown", fallbackProbs, fallbackFactors);
        }
    }
}