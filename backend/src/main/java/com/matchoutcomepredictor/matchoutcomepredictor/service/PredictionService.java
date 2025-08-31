package com.matchoutcomepredictor.service;

import com.matchoutcomepredictor.dto.predictionRequest;
import com.matchoutcomepredictor.dto.PredictionResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Service
public class PredictionService {

    public PredictionResponse getPrediction(PredictionRequest request) {
        try {
            String url = "http://localhost:5000/predict";

            Map<String, String> payload = Map.of("home_team", request.getTeamA(), "away_team", request.getTeamB());

            Map response = restTemplate.postForObject(url, payload, Map.class);

            if (response == null) {
                throw new RuntimeException("Empty response from ML service.");
            }

            String winner = (String) reponse.get("winner");
            Map<String, Double> probabilities = (Map<String, Double>) reponse.get("probabilities");
            List<String> keyFactors = (List<String>) reponse.get("keyFactors");


            return new PredictionResponse(winner, probabilities, keyFactors);
        } catch (Exception e) {
            e.PrintStackTrace();

            List<String> factors = List.of("Fallback: ML service unavailable");
            return new PredictionResponse("Unknown", probs, factors);
        }
    }

}