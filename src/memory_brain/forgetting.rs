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
