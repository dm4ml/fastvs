use std::env;

fn main() {
    match env::var("PYTHON_SYS_EXECUTABLE") {
        Ok(val) => println!("PYTHON_SYS_EXECUTABLE: {}", val),
        Err(e) => println!("couldn't interpret PYTHON_SYS_EXECUTABLE: {}", e),
    }

    // ... rest of your code ...
}
