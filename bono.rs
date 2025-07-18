/// Finds the indices of the two numbers in the given vector that add up to the target.
///
/// # Arguments
/// * `nums` - A vector of integers
/// * `target` - The target sum
///
/// # Returns
/// A tuple containing the indices of the two numbers that add up to the target, or `None` if no such pair exists.
pub fn two_sum(nums: &[i32], target: i32) -> Option<(usize, usize)> {
    // Create a hashmap to store the complement of each number and its index
    let mut complement_map: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();

    // Iterate through the vector of numbers
    for (i, &num) in nums.iter().enumerate() {
        // Calculate the complement of the current number
        let complement = target - num;

        // Check if the complement is in the hashmap
        if let Some(&j) = complement_map.get(&complement) {
            // If the complement is found, return the indices of the two numbers
            return Some((j, i));
        }

        // Otherwise, add the current number and its index to the hashmap
        complement_map.insert(num, i);
    }

    // If no pair of numbers is found that add up to the target, return None
    None
}
