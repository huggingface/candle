use std::ops::BitOrAssign;

pub struct ConjunctiveNormalForm<T> {
    all_of: Vec<T>,
}

impl<T> ConjunctiveNormalForm<T>
where
    T: for<'a> BitOrAssign<&'a T> + Clone,
{
    pub fn empty() -> Self {
        Self { all_of: Vec::new() }
    }

    pub fn take(self) -> Vec<T> {
        self.all_of
    }

    /// Adds a conjunction (AND) to the conjunctive normal form expression.
    ///
    /// For example, if the existing conjunctive normal form expression is represented as
    /// ```
    /// a and b and c
    /// ```
    /// the CNF of adding AND d is
    /// ```
    /// a and b and c and d
    /// ```
    pub fn add_conjunction(&mut self, item: T) {
        self.all_of.push(item)
    }

    /// Adds a bijection (OR) to the conjunctive normal form expression.
    ///
    /// For example, if the existing conjunctive normal form expression is represented as
    /// ```
    /// a and b
    /// ```
    ///
    /// the CNF of adding OR (c AND D)
    /// ```
    /// (a and b) or (c and d)
    /// ```
    /// is
    /// ```
    /// (a and b) or (c and d)
    /// (a or (c and d)) and (b or (c and d))
    /// (a or c) and (a or d) and (b or c) and (b or d)
    /// ```
    ///
    /// using the distributive law.
    pub fn add_bijection(&mut self, other: Self) {
        let mut result = vec![];

        if self.all_of.is_empty() {
            self.all_of = other.all_of;
            return;
        }

        for outer in &self.all_of {
            for inner in &other.all_of {
                let mut inner = inner.clone();
                inner |= outer;

                result.push(inner);
            }
        }

        self.all_of = result;
    }
}
