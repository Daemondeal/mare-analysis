use crate::parameters::*;
use crate::Individual;
use format_num::format_num;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use std::fmt::{Debug, Display, Formatter};

pub struct HourlyData {
    pub wind_cf: f32,
    pub solar_cf: f32,
    pub wec_cf: f32,
    pub energy_used: f32,
}

pub struct WeatherData {
    pub data: Vec<HourlyData>,
    pub non_renewable_cost_per_watt: f32,
    pub yearly_wind_cf: f32,
    pub yearly_solar_cf: f32,
    pub yearly_wec_cf: f32,
    pub energy_used_yearly: f32,
}

pub struct Project {
    pub wind_installed: f32,
    pub solar_installed: f32,
    pub wec_installed: f32,
    pub accumulator_installed: f32,
    pub diesel_used: f32,
    pub wasted_energy: f32,
    pub diesel_cost: f32,
    pub renewables_cost: f32,
    pub cost_first_year: f32,
    pub energy_consumed: f32,
}

impl Project {
    pub fn new(
        wind_installed: f32,
        solar_installed: f32,
        wec_installed: f32,
        accumulator_installed: f32,
        environment: &WeatherData,
    ) -> Self {
        let (diesel_used, max_pow, wasted_energy) = calculate_diesel_maxpow_waste(
            &environment.data,
            wind_installed,
            solar_installed,
            wec_installed,
            accumulator_installed,
        );

        let energy_produced = wec_installed * environment.yearly_wec_cf
            + solar_installed * environment.yearly_solar_cf
            + wind_installed * environment.yearly_wind_cf;

        let renewables_cost = calculate_lcoe(
            wec_installed,
            wind_installed,
            solar_installed,
            accumulator_installed,
            max_pow,
            environment.energy_used_yearly,
        ) * energy_produced;

        let diesel_cost = environment.non_renewable_cost_per_watt * diesel_used;
        let cost_first_year = renewables_cost + diesel_cost;

        Self {
            wind_installed,
            solar_installed,
            wec_installed,
            accumulator_installed,
            diesel_used,
            wasted_energy,
            renewables_cost,
            diesel_cost,
            cost_first_year,
            energy_consumed: environment.energy_used_yearly,
        }
    }
}

fn calculate_lcoe(
    wec_installed: f32,
    wind_installed: f32,
    solar_installed: f32,
    accumulator_installed: f32,
    accumulator_power: f32,
    energy_consumed: f32,
) -> f32 {
    let mut total_cost = 0.;
    let mut total_energy = 0.;

    for t in 1..=EXPECTED_LIFE_LCM {
        let mut cost_step = 0.;

        if t % WIND_EXPECTED_LIFE == 0 {
            cost_step += wind_installed * WIND_CAPEX;
        }
        if t % SOLAR_EXPECTED_LIFE == 0 {
            cost_step += solar_installed * SOLAR_CAPEX;
        }
        if t % WEC_EXPECTED_LIFE == 0 {
            cost_step += wec_installed * WEC_CAPEX;
        }
        if t % ACCUMULATORS_EXPECTED_LIFE == 0 {
            cost_step += ACCUMULATOR_POWER_COST * accumulator_power
                + ACCUMULATOR_CAPACITY_COST * accumulator_installed;
        }

        // OPEX
        cost_step += wind_installed * WIND_OPEX
            + solar_installed * SOLAR_OPEX
            + wec_installed * WEC_OPEX
            + ACCUMULATOR_OPEX_PERCENTAGE_ON_CAPEX
                * (ACCUMULATOR_POWER_COST * accumulator_power
                    + ACCUMULATOR_CAPACITY_COST * accumulator_installed);

        let discount = (1. + RENEWABLE_DISCOUNT_RATE).powi(t);
        total_cost += cost_step / discount;
        total_energy += energy_consumed / discount;
    }

    total_cost / total_energy
}

impl Debug for Project {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Wind: {:.02} MW, Solar: {:.02} MWh, WEC: {:.02} MWh, Accumulators: {:.02} MWh \
            Diesel: {:.02} MWh",
            self.wind_installed / 1e6,
            self.solar_installed / 1e6,
            self.wec_installed / 1e6,
            self.accumulator_installed / 1e6,
            self.diesel_used / 1e6,
        ))
    }
}

impl Display for Project {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let renewable_percentage = 1. - self.diesel_used / self.energy_consumed;
        let renewable_energy_produced =
            self.energy_consumed - self.diesel_used + self.wasted_energy;

        f.write_fmt(format_args!(
            "Installed: Wind: {}W, Solar: {}W, WEC: {}W, Accumulators: {}Wh\n\
            Energy Produced: Renewables: {}Wh, Diesel: {}Wh, Wasted from renewables: {}Wh\n\
            Yearly Cost: Diesel {} eur, Renewables: {} eur, Total: {} eur\n\
            Percentage of renewables: {}",
            format_num!(".4s", self.wind_installed),
            format_num!(".4s", self.solar_installed),
            format_num!(".4s", self.wec_installed),
            format_num!(".4s", self.accumulator_installed),
            format_num!(".4s", renewable_energy_produced),
            format_num!(".4s", self.diesel_used),
            format_num!(".4s", self.wasted_energy),
            format_num!(",.2f", self.diesel_cost),
            format_num!(",.2f", self.renewables_cost),
            format_num!(",.2f", self.cost_first_year),
            format_num!(".2%", renewable_percentage)
        ))
    }
}

impl Individual<WeatherData> for Project {
    fn from_parents(
        parent1: &Self,
        parent2: &Self,
        environment: &WeatherData,
        mut rng: &mut ThreadRng,
    ) -> Self {
        let mutation: Uniform<f32> = Uniform::new(-100_000., 100_000.);

        let wind_installed =
            (parent1.wind_installed + mutation.sample(&mut rng)).clamp(0., MAX_WIND_INSTALLED);
        let solar_installed =
            (parent1.solar_installed + mutation.sample(&mut rng)).clamp(0., MAX_SOLAR_INSTALLED);
        let wec_installed =
            (parent2.wec_installed + mutation.sample(&mut rng)).clamp(0., MAX_WEC_INSTALLED);
        let accumulator_installed = (parent2.accumulator_installed + mutation.sample(&mut rng))
            .clamp(0., MAX_ACCUMULATOR_INSTALLED);

        Self::new(
            wind_installed,
            solar_installed,
            wec_installed,
            accumulator_installed,
            environment,
        )
    }

    fn random(environment: &WeatherData, mut rng: &mut ThreadRng) -> Self {
        let uniform: Uniform<f32> = Uniform::new(EPSILON, 5_000_000.);
        let uniform_solar: Uniform<f32> = Uniform::new(EPSILON, 5_000_000.);

        let wind_installed = uniform.sample(&mut rng);
        let solar_installed = uniform_solar.sample(&mut rng);
        let wec_installed = uniform.sample(&mut rng);
        let accumulator_installed = uniform.sample(&mut rng);

        Self::new(
            wind_installed,
            solar_installed,
            wec_installed,
            accumulator_installed,
            environment,
        )
    }

    // Since the GA maximizes data, we pass the negative cost so that it gets minimized
    fn get_fitness(&self) -> f32 {
        -self.cost_first_year
    }
}

pub fn calculate_diesel_maxpow_waste(
    frame: &[HourlyData],
    wind_installed: f32,
    solar_installed: f32,
    wec_installed: f32,
    accumulator_installed: f32,
) -> (f32, f32, f32) {
    let mut state_of_charge: f32 = 0.0;
    let mut diesel_used = 0.0;
    let mut wasted_energy = 0.0;
    let mut max_pow = 0.0;

    for HourlyData {
        wind_cf,
        solar_cf,
        wec_cf,
        energy_used,
    } in frame
    {
        let wind = wind_installed * wind_cf;
        let solar = solar_installed * solar_cf;
        let wec = wec_installed * wec_cf;

        let generated = wind + solar + wec;
        let mut delta = *energy_used - generated;

        let prev_state = state_of_charge;

        if delta < 0.0 {
            if state_of_charge < accumulator_installed {
                state_of_charge += -delta * CHARGE_EFFICIENCY;
                if state_of_charge > accumulator_installed {
                    wasted_energy += state_of_charge - accumulator_installed;
                    state_of_charge = accumulator_installed;
                }
            } else {
                wasted_energy += -delta;
            }
        } else {
            if delta > state_of_charge * DISCHARGE_EFFICIENCY {
                delta -= state_of_charge * DISCHARGE_EFFICIENCY;
                state_of_charge = 0.0;
            } else {
                state_of_charge -= delta * (1. / DISCHARGE_EFFICIENCY);
                delta = 0.0;
            }

            diesel_used += delta;
        }

        let charge_difference = (state_of_charge - prev_state).abs();

        if charge_difference > max_pow {
            max_pow = charge_difference
        }
    }

    (diesel_used, max_pow, wasted_energy)
}

// Code from here on is used to make graphs
#[allow(dead_code)]
pub struct DailyEnergyBalancing {
    pub renewable_produced: f32,
    pub diesel_used: f32,
    pub current_accumulator_charge: f32,
    pub energy_used: f32,
    pub accumulator_energy_used: f32,
    pub solar_energy_produced: f32,
    pub wind_energy_produced: f32,
    pub wec_energy_produced: f32,
    pub wasted_energy: f32,
}

#[allow(dead_code)]
pub fn output_balancing(
    frame: &[HourlyData],
    wind_installed: f32,
    solar_installed: f32,
    wec_installed: f32,
    accumulator_installed: f32,
) -> Vec<DailyEnergyBalancing> {
    let mut state_of_charge: f32 = 0.0;
    let mut max_pow = 0.0;

    let mut balancing = vec![];

    for HourlyData {
        wind_cf,
        solar_cf,
        wec_cf,
        energy_used,
    } in frame
    {
        let wind = wind_installed * wind_cf;
        let solar = solar_installed * solar_cf;
        let wec = wec_installed * wec_cf;

        let generated = wind + solar + wec;
        let mut delta = *energy_used - generated;

        let prev_state = state_of_charge;

        let diesel_point;
        let wasted_energy;

        let accumulator_energy_used;

        if delta < 0.0 {
            wasted_energy = -delta;
            if state_of_charge < accumulator_installed {
                state_of_charge += -delta * CHARGE_EFFICIENCY;
                if state_of_charge > accumulator_installed {
                    state_of_charge = accumulator_installed;
                }
            }
            accumulator_energy_used = 0.;
            diesel_point = 0.;
        } else {
            wasted_energy = 0.;
            if delta > state_of_charge * DISCHARGE_EFFICIENCY {
                accumulator_energy_used = state_of_charge * DISCHARGE_EFFICIENCY;

                delta -= state_of_charge * DISCHARGE_EFFICIENCY;
                state_of_charge = 0.0;
            } else {
                accumulator_energy_used = delta;
                state_of_charge -= delta * (1. / DISCHARGE_EFFICIENCY);

                delta = 0.0;
            }

            diesel_point = delta;
        }

        let charge_difference = (state_of_charge - prev_state).abs();

        if charge_difference > max_pow {
            max_pow = charge_difference
        }

        balancing.push(DailyEnergyBalancing {
            renewable_produced: generated,
            diesel_used: diesel_point,
            current_accumulator_charge: state_of_charge,
            energy_used: *energy_used,
            accumulator_energy_used,
            solar_energy_produced: solar,
            wind_energy_produced: wind,
            wec_energy_produced: wec,
            wasted_energy,
        })
    }

    balancing
}
