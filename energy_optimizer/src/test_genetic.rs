#![allow(dead_code)]

use rand::distributions::{Distribution, Uniform};
use rand::distributions::uniform::UniformInt;
use rand::prelude::ThreadRng;
use rand::Rng;
use crate::{genetic_optimize, Individual};

#[derive(Debug)]
pub struct Evaluation {
    pub inputs: [f32; 10],
    pub output: f32,
}

pub struct Constants {
    pub a: f32,
}

pub fn benchmark_genetics() {
    let population_size = 10_000;
    let num_parents = 500;
    let num_generations = 1_000;

    let environment = Constants { a: 10. };

    let res: Evaluation = genetic_optimize(
        environment,
        population_size,
        num_generations,
        num_parents
    );

    println!("Final result: {:?}", res);
}

fn eval(constants: &Constants, inputs: &[f32; 10]) -> f32 {
    const N: usize = 10;

    let mut res = constants.a * (N as f32);

    for i in 0..N {
        res += inputs[i] * inputs[i] - constants.a * (2. * 3.14 * inputs[i]).cos();
    }

    res
}

impl Individual<Constants> for Evaluation {
    fn from_parents(parent1: &Self, parent2: &Self, environment: &Constants, mut rng: &mut ThreadRng) -> Self {
        let uniform: Uniform<f32> = Uniform::new(-0.1, 0.1);
        let mut inputs = [0f32;10];

        for i in 0..10 {
            if rng.gen_bool(0.5) {
                inputs[i] = parent1.inputs[i];
            } else {
                inputs[i] = parent2.inputs[i];
            }

            inputs[i] += uniform.sample(&mut rng);
        }

        Self {
            inputs, output: eval(environment, &inputs)
        }
    }

    fn random(environment: &Constants, mut rng: &mut ThreadRng) -> Self {
        let uniform: Uniform<f32> = Uniform::new(-2., -1.);
        let mut inputs = [0f32;10];

        for i in 0..10 {
            inputs[i] = uniform.sample(&mut rng);
        }

        Self {
            inputs, output: eval(environment, &inputs)
        }
    }

    fn get_fitness(&self) -> f32 {
        -self.output
    }
}