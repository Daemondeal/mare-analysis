#![allow(dead_code)]

// Data files
pub const WALLIS_DATA: &str = include_str!("./data/data_wallis.csv");
pub const LIPARI_DATA: &str = include_str!("./data/data_lipari.csv");

// Diesel cost is stored in euro/wh
pub const WALLIS_DIESEL_COST: f32 = 507.6 / 1e6; // Source: CRE
pub const LIPARI_DIESEL_COST: f32 = 390.0 / 1e6;

// Formula for accumulat
pub const ACCUMULATOR_CAPACITY_COST: f32 = 0.160;
pub const ACCUMULATOR_POWER_COST: f32 = 0.525;

// Capex is stored in euro/w
pub const WEC_CAPEX: f32 = 5.750;
pub const WIND_CAPEX: f32 = 2.993;
// 1.355 Onshore
pub const SOLAR_CAPEX: f32 = 0.83;

// Opex is stored in euro/(w * year)
pub const SOLAR_OPEX: f32 = 0.013;
pub const WIND_OPEX: f32 = 0.094; // Offshore
pub const WIND_OPEX_ONSHORE: f32 = 0.0423;
pub const WEC_OPEX: f32 = WEC_CAPEX * 0.03;

// Percentage on installed
pub const ACCUMULATOR_OPEX_PERCENTAGE_ON_CAPEX: f32 = 0.01;

// Values for accumulators
pub const CHARGE_EFFICIENCY: f32 = 0.97;
pub const DISCHARGE_EFFICIENCY: f32 = 0.98;

// Life is stored in years
pub const SOLAR_EXPECTED_LIFE: i32 = 25;
pub const WIND_EXPECTED_LIFE: i32 = 20;
pub const WEC_EXPECTED_LIFE: i32 = 15;
pub const ACCUMULATORS_EXPECTED_LIFE: i32 = 15;
pub const EXPECTED_LIFE_LCM: i32 = 300;

pub const RENEWABLE_DISCOUNT_RATE: f32 = 0.03;

// Max of how much all energy can be installed. Use a big number for no max
pub const MAX_WIND_INSTALLED: f32 = 1e12;
pub const MAX_SOLAR_INSTALLED: f32 = 1.2e7;
pub const MAX_WEC_INSTALLED: f32 = 1e12;
pub const MAX_ACCUMULATOR_INSTALLED: f32 = 1e12;

// Implementation details

// A number too small to be significant
pub const EPSILON: f32 = 0.01;

// Parameters for the genetic algorithm
pub const POPULATION_SIZE: usize = 1_000;
pub const NUM_PARENTS: usize = 500;
pub const NUM_GENERATIONS: usize = 5_000;
pub const GENERATION_FITNESS_TOLERANCE: f32 = 0.1;

// Proportion of individuals that enter the mating pool ignoring natural selection
pub const ELITISM_PROPORTION: f64 = 0.05;

// Proportion of individuals that get killed even though they're fit enough to survive
pub const UNLUCKY_PROPORTION: f64 = 0.05;
