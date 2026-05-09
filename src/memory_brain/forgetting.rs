//! Ebbinghaus forgetting curve — R = e^(-t/S)
//! Controls how memories decay over time when unused.

pub struct ForgettingCurve {
    /// Half-life in hours (default: 168 = 1 week)
    half_life_hours: f64,
}

impl ForgettingCurve {
    pub fn new() -> Self {
        Self {
            half_life_hours: 168.0, // 1 week
        }
    }

    pub fn with_half_life(hours: f64) -> Self {
        Self {
            half_life_hours: hours,
        }
    }

    /// Calculate the retention rate after elapsed hours.
    /// R = e^(-t / S) where S = half_life / ln(2)
    pub fn retention(&self, elapsed_hours: f64) -> f32 {
        let s = self.half_life_hours / std::f64::consts::LN_2;
        (-elapsed_hours / s).exp() as f32
    }

    /// Get the decay factor to apply per cycle.
    /// Assumes one cycle ≈ 1 hour of virtual time.
    pub fn decay_factor(&self) -> f32 {
        self.retention(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retention_at_zero() {
        let curve = ForgettingCurve::new();
        let r = curve.retention(0.0);
        assert!((r - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_retention_decays() {
        let curve = ForgettingCurve::new();
        let r1 = curve.retention(1.0);
        let r7 = curve.retention(168.0); // 1 week
        assert!(r1 > 0.99);
        assert!((r7 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_decay_factor_between_0_and_1() {
        let curve = ForgettingCurve::new();
        let factor = curve.decay_factor();
        assert!(factor > 0.0);
        assert!(factor <= 1.0);
    }

    #[test]
    fn test_custom_half_life() {
        let curve = ForgettingCurve::with_half_life(1.0); // 1 hour
        let r1 = curve.retention(1.0);
        assert!((r1 - 0.5).abs() < 0.01);
    }
}
