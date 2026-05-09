/// Converts a string to title case (first letter of each word capitalized)
pub fn to_title_case(s: String) -> String {
    s.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            let first = chars.next().unwrap().to_uppercase().collect::<String>();
            let rest = chars.collect::<String>().to_lowercase();
            format!("{}{}", first, rest)
        })
        .collect::<Vec<String>>()
        .join(" ")
}