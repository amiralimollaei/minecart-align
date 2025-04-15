use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::hash::{Hash, Hasher};
use std::f64;
use std::cmp::Ordering;
use std::env;
use std::process;

// Global variables initialized at start
static mut LAST_PRINT_TIME: f64 = 0.0;
static mut CONSTANT_MOVEMENT: f64 = 0.00589375;
static mut PRECISION: f64 = 1e-6;
const ORIGIN0: Point = Point { x: 0.0 };
const ORIGIN1: Point = Point { x: 1.0 };
// Helper function for printing with cooldown
fn cooldown_print(string: &str, cooldown: f64) {
    unsafe {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        if (now - LAST_PRINT_TIME) > cooldown {
            println!("{}", string);
            LAST_PRINT_TIME = now;
        }
    }
}
// Point struct
#[derive(Clone, Debug, Copy)]
struct Point {
    x: f64,
}
impl Point {
    fn new(x: f64) -> Self {
        Point { x }
    }
    
    #[inline]
    fn distance_to(&self, other: &Point) -> f64 {
        (self.x - other.x).abs()
    }
}
impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "P({})", self.x)
    }
}
impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.x.to_bits() == other.x.to_bits()
    }
}
impl Eq for Point {}
impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
    }
}
// Node for our priority queue
#[derive(Clone, Copy)]
struct Node {
    point: Point,
    f_score: f64,
}
// We need to implement Ord for the BinaryHeap
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // We want a min-heap, so we reverse the ordering
        other.f_score.partial_cmp(&self.f_score).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score
    }
}
impl Eq for Node {}
// Heuristic function for A*
#[inline]
fn heuristic(point: &Point, target: &Point) -> f64 {
    point.distance_to(target)
}
// Reconstruct the path from the cameFrom map
fn reconstruct_path(
    came_from: &HashMap<Point, (Point, &'static str)>,
    mut current: Point,
    last_action: &'static str,
) -> (Vec<Point>, Vec<&'static str>) {
    let mut total_path = vec![current];
    let mut total_actions = vec![last_action];
    
    while let Some((prev, action)) = came_from.get(&current) {
        current = *prev;
        total_path.insert(0, current);
        total_actions.insert(0, *action);
    }
    
    (total_path, total_actions)
}
// Find neighbors of a point - now returns static str references
#[inline]
fn neighbors(current: &Point) -> [(Point, &'static str); 4] {
    unsafe {
        [
            (Point::new((current.x + ORIGIN0.x) * 0.5), "half_left"),
            (Point::new((current.x + ORIGIN1.x) * 0.5), "half_right"),
            (Point::new(current.x - CONSTANT_MOVEMENT), "constant_left"),
            (Point::new(current.x + CONSTANT_MOVEMENT), "constant_right")
        ]
    }
}
// A* algorithm
fn a_star(start: Point, goal: Point) -> Option<(Vec<Point>, Vec<&'static str>)> {
    // Use a binary heap for the open set for O(log n) extract min
    let mut open_heap = BinaryHeap::new();
    open_heap.push(Node { point: start, f_score: heuristic(&start, &goal) });
    
    // Keep track of points in the open set
    let mut open_set = HashSet::new();
    open_set.insert(start);
    
    // For node n, came_from[n] is the node immediately preceding it
    let mut came_from: HashMap<Point, (Point, &'static str)> = HashMap::new();
    
    // For node n, g_score[n] is the cost of the cheapest path from start to n
    let mut g_score: HashMap<Point, f64> = HashMap::new();
    g_score.insert(start, 0.0);
    
    // For node n, f_score[n] = g_score[n] + h(n)
    let mut f_scores = HashMap::new();
    f_scores.insert(start, heuristic(&start, &goal));
    
    let mut action = "none";
    let mut min_distance_to_goal = f64::INFINITY;
    
    while let Some(Node { point: current, .. }) = open_heap.pop() {
        // If we've popped a node that's already been processed, skip it
        if !open_set.contains(&current) {
            continue;
        }
        
        let distance_to_goal = current.distance_to(&goal);
        min_distance_to_goal = min_distance_to_goal.min(distance_to_goal);
        
        cooldown_print(
            &format!("searching... distance to goal: {:.9}, size of openSet: {}", 
                     min_distance_to_goal, open_set.len()),
            0.05
        );
        
        unsafe {
            if distance_to_goal < PRECISION {
                println!("found a path with distance={}", min_distance_to_goal);
                return Some(reconstruct_path(&came_from, current, action));
            }
        }
        
        open_set.remove(&current);
        
        for (neighbor, neighbor_action) in neighbors(&current) {
            // Calculate tentative g_score
            let current_g = *g_score.get(&current).unwrap_or(&f64::INFINITY);
            let tentative_g_score = current_g + 1.0;//(heuristic(&current, &neighbor) + 1.0);
            
            if tentative_g_score < *g_score.get(&neighbor).unwrap_or(&f64::INFINITY) {
                // This path to neighbor is better than any previous one. Record it!
                came_from.insert(neighbor, (current, neighbor_action));
                g_score.insert(neighbor, tentative_g_score);
                let new_f_score = tentative_g_score + heuristic(&neighbor, &goal);
                f_scores.insert(neighbor, new_f_score);
                
                if !open_set.contains(&neighbor) {
                    open_set.insert(neighbor);
                    open_heap.push(Node { point: neighbor, f_score: new_f_score });
                }
            }
            action = neighbor_action;
        }
    }
    
    // Open set is empty but goal was never reached
    None
}

fn print_usage() {
    println!("Usage: program [OPTIONS] <target>");
    println!("Options:");
    println!("  --start FLOAT        Set the start point (default: 0.5)");
    println!("  --precision FLOAT    Set the precision (default: 1e-6)");
    println!("  --constant FLOAT     Set the constant movement value (default: 0.00589375)");
    println!("  --help               Display this help message");
    println!("");
    println!("Arguments:");
    println!("  <target>             The target point value (required)");
    println!("");
    println!("Examples:");
    println!("  program 0.75");
    println!("  program --start 0.4 0.9");
    println!("  program --precision 1e-7 --constant 0.005 0.6");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    // Default values
    let mut start_value = 0.5;
    let mut target_value = None;
    
    // Parse command line arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--start" => {
                if i + 1 < args.len() {
                    match args[i + 1].parse::<f64>() {
                        Ok(val) => start_value = val,
                        Err(_) => {
                            println!("Error: Invalid start value");
                            print_usage();
                            process::exit(1);
                        }
                    }
                    i += 2;
                } else {
                    println!("Error: Missing value for --start");
                    print_usage();
                    process::exit(1);
                }
            },
            "--precision" => {
                if i + 1 < args.len() {
                    match args[i + 1].parse::<f64>() {
                        Ok(val) => unsafe { PRECISION = val },
                        Err(_) => {
                            println!("Error: Invalid precision value");
                            print_usage();
                            process::exit(1);
                        }
                    }
                    i += 2;
                } else {
                    println!("Error: Missing value for --precision");
                    print_usage();
                    process::exit(1);
                }
            },
            "--constant" => {
                if i + 1 < args.len() {
                    match args[i + 1].parse::<f64>() {
                        Ok(val) => unsafe { CONSTANT_MOVEMENT = val },
                        Err(_) => {
                            println!("Error: Invalid constant value");
                            print_usage();
                            process::exit(1);
                        }
                    }
                    i += 2;
                } else {
                    println!("Error: Missing value for --constant");
                    print_usage();
                    process::exit(1);
                }
            },
            "--help" => {
                print_usage();
                process::exit(0);
            },
            arg if arg.starts_with("--") => {
                println!("Unknown option: {}", arg);
                print_usage();
                process::exit(1);
            },
            _ => {
                // This must be the target (positional argument)
                match args[i].parse::<f64>() {
                    Ok(val) => target_value = Some(val),
                    Err(_) => {
                        println!("Error: Invalid target value");
                        print_usage();
                        process::exit(1);
                    }
                }
                i += 1;
            }
        }
    }
    
    // Check if target was provided
    let target_value = match target_value {
        Some(val) => val,
        None => {
            println!("Error: Missing required target value");
            print_usage();
            process::exit(1);
        }
    };
    
    let start = Point::new(start_value);
    let target = Point::new(target_value);
    
    println!("Using start = {}, target = {}", start, target);

    unsafe {
        let precision = PRECISION;
        let constant_movement = CONSTANT_MOVEMENT;
        println!("Precision = {}, Constant movement = {}", precision, constant_movement);
    }
    
    let start_time = Instant::now();
    if let Some((path, actions)) = a_star(start, target) {
        println!("actions={:?}", actions);
        for p in path {
            println!("{}", p);
        }
    } else {
        println!("No path found");
    }
    println!("Execution time: {:?}", start_time.elapsed());
}