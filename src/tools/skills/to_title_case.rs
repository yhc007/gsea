/// Converts a string to title case (first letter of each word capitalized).
pub fn to_title_case(s: &str) -> String {
    s.split_whitespace()
        .filter_map(|word| {
            let mut chars = word.chars();
            chars.next().map(|first| {
                let capitalized = first.to_uppercase().collect::<String>();
                let rest: String = chars.collect();
                format!("{}{}", capitalized, rest.to_lowercase())
            })
        })
        .collect::<Vec<String>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_title_case() {
        assert_eq!(to_title_case("hello world"), "Hello World");
        assert_eq!(to_title_case("rust"), "Rust");
        assert_eq!(to_title_case(""), "");
        assert_eq!(to_title_case("  "), "");
    }
}
