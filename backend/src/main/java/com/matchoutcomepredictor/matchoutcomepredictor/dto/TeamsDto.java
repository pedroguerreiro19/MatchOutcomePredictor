package com.matchoutcomepredictor.dto;

public class TeamDto {
    private String name;
    private String logo;

    public TeamDto(String name, String logo) {
        this.name = name;
        this.logo = logo;
    }

    public String getName() { return name; }
    public String getLogo() { return logo; }
}