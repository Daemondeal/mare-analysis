use analysis::DailyEnergyBalancing;

use crate::analysis::{output_balancing, HourlyData, Project, WeatherData};
use crate::genetic::{genetic_optimize, Individual};
use crate::parameters::*;
use std::fs;
use std::str::FromStr;

mod analysis;
mod genetic;
mod parameters;

fn read_file(data: &str) -> Vec<HourlyData> {
    let mut res = vec![];

    for row in data.lines().skip(1) {
        let entry: Vec<&str> = row.split(',').collect();
        if entry.len() != 5 {
            continue;
        }

        let solar_res = f32::from_str(entry[1]);
        let wind_res = f32::from_str(entry[2]);
        let wec_res = f32::from_str(entry[3]);
        let consumed_res = f32::from_str(entry[4]);

        if let (Ok(solar), Ok(wind), Ok(wec), Ok(consumed)) =
            (solar_res, wind_res, wec_res, consumed_res)
        {
            res.push(HourlyData {
                wind_cf: wind,
                solar_cf: solar,
                wec_cf: wec,
                energy_used: consumed,
            });
        }
    }

    res
}

fn main() {
    let data = read_file(WALLIS_DATA);
    let diesel_cost = WALLIS_DIESEL_COST;

    let yearly_wind_cf: f32 = data.iter().map(|x| x.wind_cf).sum::<f32>();
    let yearly_wec_cf: f32 = data.iter().map(|x| x.wec_cf).sum::<f32>();
    let yearly_solar_cf: f32 = data.iter().map(|x| x.solar_cf).sum::<f32>();

    let energy_used_yearly: f32 = data.iter().map(|x| x.energy_used).sum();

    let n: f32 = data.len() as f32;

    let wind_cf_avg: f32 = yearly_wind_cf / n;
    let wec_cf_avg: f32 = yearly_wec_cf / n;
    let solar_cf_avg: f32 = yearly_solar_cf / n;

    println!("Energy used: {:.02} GWh", energy_used_yearly / 1e9);
    println!(
        "Average wind cf: {:.02} %\nAverage WEC cf: {:.02} %\nAverage solar cf: {:.02} %\n",
        wind_cf_avg * 100.,
        wec_cf_avg * 100.,
        solar_cf_avg * 100.
    );

    let weather = WeatherData {
        data,
        yearly_wind_cf,
        yearly_solar_cf,
        yearly_wec_cf,
        energy_used_yearly,

        non_renewable_cost_per_watt: diesel_cost,
    };

    let resulting_project: Project =
        genetic_optimize(weather, POPULATION_SIZE, NUM_GENERATIONS, NUM_PARENTS);

    println!("{}", resulting_project);
}

// Code from here on is used to get resulting data
#[allow(dead_code)]
fn get_dates(data: &str) -> Vec<String> {
    data.lines()
        .skip(1)
        .filter_map(|row| row.split(',').next())
        .map(|date| date.into())
        .collect()
}

#[allow(dead_code)]
fn write_balancing(filename: &str, balancing: Vec<DailyEnergyBalancing>, dates: Vec<String>) {
    assert_eq!(balancing.len(), dates.len());
    let mut res = String::new();

    res.push_str("date,renewable_used,diesel_used,charge,energy_used,accumulator_used,solar_used,wind_used,wec_used,wasted_energy\n");
    for (
        DailyEnergyBalancing {
            renewable_produced,
            diesel_used,
            current_accumulator_charge,
            energy_used,
            accumulator_energy_used,
            solar_energy_produced,
            wind_energy_produced,
            wec_energy_produced,
            wasted_energy,
        },
        date,
    ) in balancing.iter().zip(dates.iter())
    {
        res.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{}\n",
            date,
            renewable_produced,
            diesel_used,
            current_accumulator_charge,
            energy_used,
            accumulator_energy_used,
            solar_energy_produced,
            wind_energy_produced,
            wec_energy_produced,
            wasted_energy
        ));
    }

    fs::write(filename, res).unwrap();
}

#[allow(dead_code)]
fn generate_balancing(environment: &WeatherData) {
    print!("Lipari ");
    let _lipari_balancing_project = Project::new(
        5_000_000.,
        12_000_000.,
        1_000_000.,
        45_080_000.,
        environment,
    );

    print!("Wallis ");
    let wallis_balancing_project =
        Project::new(2_000_000., 9_185_000., 0_000_000., 26_700_000., environment);

    let balancing_project = wallis_balancing_project;

    println!("{}", balancing_project);

    let balancing = output_balancing(
        &environment.data,
        balancing_project.wind_installed,
        balancing_project.solar_installed,
        balancing_project.wec_installed,
        balancing_project.accumulator_installed,
    );

    write_balancing("./results/balancing.csv", balancing, get_dates(WALLIS_DATA));
}
