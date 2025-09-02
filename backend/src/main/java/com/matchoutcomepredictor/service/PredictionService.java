package com.matchoutcomepredictor.service;

import com.matchoutcomepredictor.dto.PredictionRequest;
import com.matchoutcomepredictor.dto.PredictionResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Service
public class PredictionService {

    private final RestTemplate restTemplate;

    private static final Map<String, String> TEAM_NAME_MAP = Map.ofEntries(
        Map.entry("SL Benfica", "Benfica"),
        Map.entry("Sporting CP", "Sp Lisbon"),
        Map.entry("FC Porto", "Porto"),
        Map.entry("SC Braga", "Sp Braga"),
        Map.entry("Vitória SC", "Guimaraes"),
        Map.entry("Estoril Praia", "Estoril"),
        Map.entry("CF Estrela da Amadora", "Estrela"),
        Map.entry("Casa Pia AC", "Casa Pia"),
        Map.entry("FC Famalicão", "Famalicao"),
        Map.entry("Gil Vicente FC", "Gil Vicente"),
        Map.entry("Rio Ave FC", "Rio Ave"),
        Map.entry("Santa Clara", "Santa Clara"),
        Map.entry("CD Tondela", "Tondela"),
        Map.entry("Moreinense FC", "Moreirense"),
        Map.entry("CD Nacional", "Nacional"),
        Map.entry("FC Arouca", "Arouca"),
        Map.entry("FC Alverca", "Alverca"),
        Map.entry("AFS", "Aves")
    );

    public PredictionService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public PredictionResponse getPrediction(PredictionRequest request) {
        try {
            String url = "http://localhost:8000/predict";

            System.out.println("➡️ Home recebido: " + request.getHomeTeam());
            System.out.println("➡️ Away recebido: " + request.getAwayTeam());

            String home = TEAM_NAME_MAP.getOrDefault(request.getHomeTeam(), request.getHomeTeam());
            String away = TEAM_NAME_MAP.getOrDefault(request.getAwayTeam(), request.getAwayTeam());
            Map<String, String> payload = Map.of(
                "home_team", home,
                "away_team", away
            );
            System.out.println("Sending to Python: home=" + home + ", away=" + away);

            Map response = restTemplate.postForObject(url, payload, Map.class);

            if (response == null) {
                throw new RuntimeException("Empty response from ML service.");
            }

            String winner = (String) response.get("winner");
            Map<String, Double> probabilities = (Map<String, Double>) response.get("probabilities");
            List<Map<String, Object>> keyFactors = (List<Map<String, Object>>) response.get("keyFactors");

            return new PredictionResponse(winner, probabilities, keyFactors);
        } catch (Exception e) {
            e.printStackTrace();

            Map<String, Double> fallbackProbs = Map.of(
                "HomeWin", 0.0,
                "AwayWin", 0.0,
                "Draw", 0.0
            );
            List<Map<String, Object>> fallbackFactors = List.of(
                Map.of("feature", "Fallback", "impact", 0.0)
            );

            return new PredictionResponse("Unknown", fallbackProbs, fallbackFactors);
        }
    }
}