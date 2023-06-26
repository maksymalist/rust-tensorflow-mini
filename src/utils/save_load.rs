use std::{path::Path, io::Write, io::Read};
use serde::Deserialize;

use crate::Error;
use std::fs::File;


pub fn save_model_params(path: String, params: String) -> Result<(), Error> {

    let path = Path::new(&path);
    let bytes = params.as_bytes();

    let mut file = File::create(path)?;
    file.write_all(bytes)?;

    Ok(())
}

pub fn load_model_params<T>(path: String) -> Result<T, Error>
where
    T: Deserialize<'static>,
{
    let mut file = File::open(path)?;
    let mut buffer = String::new();
    file.read_to_string(&mut buffer)?;

    let deserialized: T = serde_json::from_str(Box::leak(buffer.into_boxed_str())).unwrap();
    Ok(deserialized)
}