/// Checks if a string contains only alphanumeric characters.
pub fn is_alphanumeric(s: &str) -> bool {
    s.chars().all(|c| c.is_alphanumeric())
}