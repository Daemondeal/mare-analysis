use crate::parameters::*;
use rand::prelude::IteratorRandom;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::fmt::Debug;
use std::time::Instant;

pub trait Individual<E> {
    fn from_parents(parent1: &Self, parent2: &Self, environment: &E, rng: &mut ThreadRng) -> Self;

    fn random(environment: &E, rng: &mut ThreadRng) -> Self;

    fn get_fitness(&self) -> f32;
}

fn filter_mating_pool<I, E>(population: &mut Vec<I>, num_parents: usize, rng: &mut ThreadRng)
where
    I: Individual<E>,
{
    population.sort_by(|x, y| x.get_fitness().partial_cmp(&y.get_fitness()).unwrap());

    let cutoff = population[num_parents - 1].get_fitness();

    population.retain(|x| {
        if x.get_fitness() >= cutoff {
            rng.gen_bool(1. - UNLUCKY_PROPORTION)
        } else {
            rng.gen_bool(ELITISM_PROPORTION)
        }
    });
}

fn crossover<I, E>(
    population: &[I],
    environment: &E,
    mut children: Vec<I>,
    population_size: usize,
    mut rng: &mut ThreadRng,
) -> Vec<I>
where
    I: Individual<E>,
{
    let child_count = population_size - population.len();

    for _ in 0..child_count {
        let parents = population.iter().choose_multiple(&mut rng, 2);

        children.push(Individual::from_parents(
            parents[0],
            parents[1],
            environment,
            rng,
        ));
    }

    children
}

fn get_fittest<I, E>(population: &[I]) -> &I
where
    I: Individual<E>,
{
    population
        .iter()
        .max_by(|x, y| x.get_fitness().partial_cmp(&y.get_fitness()).unwrap())
        .unwrap()
}

pub fn genetic_optimize<I, E>(
    environment: E,
    population_size: usize,
    num_generations: usize,
    num_parents: usize,
) -> I
where
    I: Individual<E> + Debug,
{
    let mut rng = rand::thread_rng();

    let mut population: Vec<I> = vec![];

    println!("Creating population...");
    for _ in 0..population_size {
        population.push(Individual::random(&environment, &mut rng));
    }

    println!("Starting genetic algorithm...");
    let start = Instant::now();
    let mut children = Vec::with_capacity(population_size - num_parents);
    let mut last_fitness: Option<f32> = None;

    for generation in 0..num_generations {
        filter_mating_pool(&mut population, num_parents, &mut rng);
        children = crossover(
            &population,
            &environment,
            children,
            population_size,
            &mut rng,
        );
        population.append(&mut children);

        if generation % 10 == 0 {
            let max = get_fittest::<I, E>(&population);
            let max_fitness = max.get_fitness();
            println!("{}: {:?} (fitness: {:10.3e})", generation, max, max_fitness);

            if let Some(last_best) = last_fitness {
                if (last_best - max_fitness).abs() < GENERATION_FITNESS_TOLERANCE {
                    break;
                }
            }

            last_fitness = Some(max_fitness);
        }
    }
    println!("Done! It took {} s", start.elapsed().as_secs_f64());

    population
        .into_iter()
        .max_by(|x, y| x.get_fitness().partial_cmp(&y.get_fitness()).unwrap())
        .unwrap()
}
