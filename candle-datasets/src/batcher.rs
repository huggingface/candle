use candle::{Result, Tensor};

pub struct Batcher<I> {
    inner: I,
    batch_size: usize,
    return_last_incomplete_batch: bool,
}

impl<I> Batcher<I> {
    fn new(inner: I) -> Self {
        Self {
            inner,
            batch_size: 16,
            return_last_incomplete_batch: false,
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn return_last_incomplete_batch(mut self, r: bool) -> Self {
        self.return_last_incomplete_batch = r;
        self
    }
}

pub struct Iter1<I: Iterator<Item = Tensor>> {
    inner: I,
}

pub struct Iter2<I: Iterator<Item = (Tensor, Tensor)>> {
    inner: I,
}

impl<I: Iterator<Item = Tensor>> Batcher<Iter1<I>> {
    pub fn new1(inner: I) -> Self {
        Self::new(Iter1 { inner })
    }
}

impl<I: Iterator<Item = (Tensor, Tensor)>> Batcher<Iter2<I>> {
    pub fn new2(inner: I) -> Self {
        Self::new(Iter2 { inner })
    }
}

pub struct IterResult1<I: Iterator<Item = Result<Tensor>>> {
    inner: I,
}

pub struct IterResult2<I: Iterator<Item = Result<(Tensor, Tensor)>>> {
    inner: I,
}

impl<I: Iterator<Item = Result<Tensor>>> Batcher<IterResult1<I>> {
    pub fn new_r1(inner: I) -> Self {
        Self::new(IterResult1 { inner })
    }
}

impl<I: Iterator<Item = Result<(Tensor, Tensor)>>> Batcher<IterResult2<I>> {
    pub fn new_r2(inner: I) -> Self {
        Self::new(IterResult2 { inner })
    }
}

impl<I: Iterator<Item = Tensor>> Iterator for Batcher<Iter1<I>> {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut items = Vec::with_capacity(self.batch_size);
        for _i in 0..self.batch_size {
            // We have two levels of inner here so that we can have two implementations of the
            // Iterator trait that are different for Iter1 and Iter2. If rust gets better
            // specialization at some point we can get rid of this.
            match self.inner.inner.next() {
                Some(item) => items.push(item),
                None => {
                    if self.return_last_incomplete_batch && !items.is_empty() {
                        break;
                    }
                    return None;
                }
            }
        }
        Some(Tensor::stack(&items, 0))
    }
}

impl<I: Iterator<Item = (Tensor, Tensor)>> Iterator for Batcher<Iter2<I>> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut xs = Vec::with_capacity(self.batch_size);
        let mut ys = Vec::with_capacity(self.batch_size);
        for _i in 0..self.batch_size {
            match self.inner.inner.next() {
                Some((x, y)) => {
                    xs.push(x);
                    ys.push(y)
                }
                None => {
                    if self.return_last_incomplete_batch && !xs.is_empty() && !ys.is_empty() {
                        break;
                    }
                    return None;
                }
            }
        }
        let xs = Tensor::stack(&xs, 0);
        let ys = Tensor::stack(&ys, 0);
        Some(xs.and_then(|xs| ys.map(|ys| (xs, ys))))
    }
}

impl<I: Iterator<Item = Result<Tensor>>> Iterator for Batcher<IterResult1<I>> {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut items = Vec::with_capacity(self.batch_size);
        for _i in 0..self.batch_size {
            // We have two levels of inner here so that we can have two implementations of the
            // Iterator trait that are different for Iter1 and Iter2. If rust gets better
            // specialization at some point we can get rid of this.
            match self.inner.inner.next() {
                Some(item) => items.push(item),
                None => {
                    if self.return_last_incomplete_batch && !items.is_empty() {
                        break;
                    }
                    return None;
                }
            }
        }
        let items = items.into_iter().collect::<Result<Vec<Tensor>>>();
        Some(items.and_then(|items| Tensor::stack(&items, 0)))
    }
}

impl<I: Iterator<Item = Result<(Tensor, Tensor)>>> Iterator for Batcher<IterResult2<I>> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut xs = Vec::with_capacity(self.batch_size);
        let mut ys = Vec::with_capacity(self.batch_size);
        let mut errs = vec![];
        for _i in 0..self.batch_size {
            match self.inner.inner.next() {
                Some(Ok((x, y))) => {
                    xs.push(x);
                    ys.push(y)
                }
                Some(Err(err)) => errs.push(err),
                None => {
                    if self.return_last_incomplete_batch && !xs.is_empty() && !ys.is_empty() {
                        break;
                    }
                    return None;
                }
            }
        }
        if !errs.is_empty() {
            return Some(Err(errs.swap_remove(0)));
        }
        let xs = Tensor::stack(&xs, 0);
        let ys = Tensor::stack(&ys, 0);
        Some(xs.and_then(|xs| ys.map(|ys| (xs, ys))))
    }
}
