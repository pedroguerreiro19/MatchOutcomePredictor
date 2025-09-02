package com.matchoutcomepredictor.controller;

import com.matchoutcomepredictor.dto.TeamDto;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/teams")
@CrossOrigin(origins = "http:localhost:5173")
public class TeamController {
    
    @GetMapping
    public List<TeamDto> getTeams() {
        return List.of(
            new TeamDto("FC Alverca", "/logos/alverca.png" ),
            new TeamDto("FC Arouca", "/logos/arouca.png" ),
            new TeamDto("AFS", "/logos/aves.png" ),
            new TeamDto("SL Benfica", "/logos/benfica.png" ),
            new TeamDto("SC Braga", "/logos/braga.png" ),
            new TeamDto("Casa Pia AC", "/logos/casa_pia.png" ),
            new TeamDto("Estoril Praia", "/logos/estoril.png" ),
            new TeamDto("Estrela Amadora", "/logos/estrela_amadora.png" ),
            new TeamDto("FC Famalicão", "/logos/famalicao.png" ),
            new TeamDto("Gil Vicente FC", "/logos/gil_vicente.png" ),
            new TeamDto("Vitória SC", "/logos/guimaraes.png" ),
            new TeamDto("Moreinense FC", "/logos/moreinense.png" ),
            new TeamDto("CD Nacional", "/logos/nacional.png" ),
            new TeamDto("FC Porto", "/logos/porto.png" ),
            new TeamDto("Rio Ave FC", "/logos/rio_ave.png" ),
            new TeamDto("Santa Clara", "/logos/santa_clara.png" ),
            new TeamDto("Sporting CP", "/logos/sporting.png" ),
            new TeamDto("CD Tondela", "/logos/tondela.png" )
            );
    }
}