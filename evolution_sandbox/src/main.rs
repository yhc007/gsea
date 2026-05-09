use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() {
    let resource_a = Arc::new(Mutex::new(0));
    let resource_b = Arc::new(Mutex::new(0));

    let a_clone = Arc::clone(&resource_a);
    let b_clone = Arc::clone(&resource_b);

    let handle1 = thread::spawn(move || {
        let _lock_a = a_clone.lock().unwrap();
        println!("Thread 1: Acquired lock A");
        thread::sleep(Duration::from_millis(50));
        let _lock_b = b_clone.lock().unwrap();
        println!("Thread 1: Acquired lock B");
    });

    let a_clone2 = Arc::clone(&resource_a);
    let b_clone2 = Arc::clone(&resource_b);

    let handle2 = thread::spawn(move || {
        // FIX: Acquire Lock A BEFORE Lock B to maintain consistent order
        let _lock_a = a_clone2.lock().unwrap();
        println!("Thread 2: Acquired lock A");
        thread::sleep(Duration::from_millis(50));
        let _lock_b = b_clone2.lock().unwrap();
        println!("Thread 2: Acquired lock B");
    });

    handle1.join().unwrap();
    let res = handle2.join();
    if let Err(e) = res {
        eprintln!("Thread 2 panicked: {:?}", e);
    }
    
    println!("Finished successfully!");
}
