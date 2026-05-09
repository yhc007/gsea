/// Returns true if the slice contains only even numbers.
pub fn is_all_even(nums: &[i32]) -> bool {
    nums.iter().all(|&x| x % 2 == 0)
}