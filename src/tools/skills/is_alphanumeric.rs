/// Checks if a string contains only alphanumeric characters.
pub fn is_alphanumeric(s: &str) -> bool {
    s.chars().all(|c| c.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_alphanumeric() {
        assert!(is_alphanumeric("hello123"));
        assert!(!is_alphanumeric("hello world!"));
    }
}
