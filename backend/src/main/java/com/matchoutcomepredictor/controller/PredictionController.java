package com.matchoutcomepredictor.controller;

import com.matchoutcomepredictor.dto.PredictionRequest;
import com.matchoutcomepredictor.dto.PredictionResponse;
import com.matchoutcomepredictor.service.PredictionService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("api")
@CrossOrigin(origins = "http://localhost:5173")
public class PredictionController {

    private final PredictionService predictionService;

    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @PostMapping("/predict")
    public PredictionResponse predict(@RequestBody PredictionRequest request) {
        return predictionService.getPrediction(request);
    }
}
