pub struct OptimizerAdam {
    pub learning_rate: f64,
}

impl OptimizerAdam {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    pub fn update_params() {
        println!("Updating params");
    }
}
