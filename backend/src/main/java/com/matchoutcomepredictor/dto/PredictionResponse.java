package com.matchoutcomepredictor.dto;

import java.util.Map;
import java.util.List;

public class PredictionResponse {
    private String prediction;
    private Map<String, Double> probabilities;
    private List<Map<String, Object>> keyFactors;

    public PredictionResponse(String prediction, Map<String, Double> probabilities, List<Map<String, Object>> keyFactors ) {
        this.prediction = prediction;
        this.probabilities = probabilities;
        this.keyFactors = keyFactors;
    }

    public String getPrediction() { return prediction; }
    public Map<String, Double> getProbabilities() { return probabilities; }
    public List<Map<String, Object>> getKeyFactors() { return keyFactors; }
}