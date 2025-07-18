se rand::Seq;
use std::io;

fn main() {
    println!("Welcome to the Guessing Game!");

    // Generate a random number between 1 and 100
    let secret_number = rand::thread_rng().gen_range(1..=100);

    loop {
        println!("Please enter your guess (1-100):");

        // Read user input
        let mut guess = String::new();
        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        // Convert user input to a number
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Please enter a valid number.");
                continue;
            }
        };

        // Compare the guess with the secret number
        if guess == secret_number {
            println!("Congratulations! You guessed the correct number!");
            break;
        } else if guess < secret_number {
            println!("Too low! Try again.");
        } else {
            println!("Too high! Try again.");
        }
    }
}
