use candle::{CpuStorage, CustomOp2, Layout, Result, Shape};
use half::{bf16, f16};
use rayon::prelude::*;
use std::fmt::Debug;
use std::marker::{Send, Sync};
pub struct SearchSorted {
    pub right: bool,
}

pub trait Sortable<T: PartialOrd + Debug + Sync + Send> {
    fn search_sorted(
        &self,
        innerdim_bd: usize,
        values: &[T],
        innerdim_val: usize,
        is_1d_bd: bool,
        is_1d_vals: bool,
        right: bool,
    ) -> Vec<i64>;
}
macro_rules! match_cpu_storage {
    ($s1:expr, $s2:expr, $code:expr) => {
        match $s1 {
            CpuStorage::U8(vs) => match $s2 {
                CpuStorage::U8(values) => $code(vs, values),
                _ => panic!("Unsupported data type"),
            },
            CpuStorage::U32(vs) => match $s2 {
                CpuStorage::U32(values) => $code(vs, values),
                _ => panic!("Unsupported data type"),
            },
            CpuStorage::I64(vs) => match $s2 {
                CpuStorage::I64(values) => $code(vs, values),
                _ => panic!("Unsupported data type"),
            },
            CpuStorage::BF16(vs) => match $s2 {
                CpuStorage::BF16(values) => $code(vs, values),
                _ => panic!("Unsupported data type"),
            },
            CpuStorage::F16(vs) => match $s2 {
                CpuStorage::F16(values) => $code(vs, values),
                _ => panic!("Unsupported data type"),
            },
            CpuStorage::F32(vs) => match $s2 {
                CpuStorage::F32(values) => $code(vs, values),
                _ => panic!("Unsupported data type"),
            },
            CpuStorage::F64(vs) => match $s2 {
                CpuStorage::F64(values) => $code(vs, values),
                _ => panic!("Unsupported data type"),
            },
            _ => panic!("Unsupported data type"),
        }
    };
}

fn binary_search<T: PartialOrd>(slice: &[T], value: &T, right: bool) -> i64 {
    let mut start: usize = 0;
    let mut end: usize = slice.len();
    while start < end {
        let mid = start + ((end - start) >> 1);
        let mid_val = &slice[mid];
        let pred = if right {
            !(mid_val > value)
        } else {
            !(mid_val >= value)
        };
        if pred {
            start = mid + 1;
        } else {
            end = mid;
        }
    }
    start as i64
}
impl<T: PartialOrd + Debug + Sync + Send> Sortable<T> for Vec<T> {
    fn search_sorted(
        &self,
        innerdim_bd: usize,
        values: &[T],
        innerdim_val: usize,
        is_1d_bd: bool,
        is_1d_vals: bool,
        right: bool,
    ) -> Vec<i64> {
        // assert!(self.len() % innerdim_bd == 0);
        // assert!(values.len() % innerdim_val == 0);
        // let mut indices: Vec<i64> = Vec::new();

        let indices: Vec<i64> = match (is_1d_bd, is_1d_vals) {
            // 1-d sorted seq, n-d vals --> apply each "row" of vals to the sorted seq
            (true, false) => {
                let num_val_its = values.len() / innerdim_val;
                (0..num_val_its)
                    .into_par_iter()
                    .map(|i| {
                        let slice = &self[..];
                        let vals = &values[i * innerdim_val..(i + 1) * innerdim_val];
                        let mut inner_vec: Vec<i64> = Vec::new();
                        for v in vals {
                            let found = binary_search(slice, v, right);
                            inner_vec.push(found as i64);
                        }
                        inner_vec
                    })
                    .flatten()
                    .collect()
            }
            // n-d sorted seq, 1-d vals --> search for vals in each row of sorted seq
            (false, true) => {
                println!("Matched n-d sort, 1-d vals");
                let num_it = self.len() / innerdim_bd;
                let matches: Vec<i64> = (0..num_it)
                    .into_par_iter()
                    // .step_by(innerdim_bd)
                    .map(|i| {
                        let slice = &self[i * innerdim_bd..(i + 1) * innerdim_bd];
                        println!("Slice: {:?}", slice);
                        let vals = &values[..];
                        let mut inner_vec: Vec<i64> = Vec::new();
                        for v in vals {
                            let found = binary_search(slice, v, right);
                            inner_vec.push(found as i64);
                        }
                        println!("inner matches: {:?}", inner_vec);
                        inner_vec
                    })
                    .flatten()
                    .collect();
                println!("matches: {:?}", matches);
                matches
            }
            // N-d sorted seq, N-d vals --> num "rows" of vals must be equal to the num "rows" of sorted seq
            // each row of vals is applied to the corresponding row of sorted seq
            _ => {
                assert!(self.len() / innerdim_bd == values.len() / innerdim_val);

                let num_it = self.len() / innerdim_bd;
                let matches: Vec<i64> = (0..num_it)
                    .into_par_iter()
                    // .step_by(innerdim_bd)
                    .map(|i| {
                        let mut inner_vec: Vec<i64> = Vec::new();
                        let slice = &self[i * innerdim_bd..(i + 1) * innerdim_bd];
                        let vals = &values[i * innerdim_val..(i + 1) * innerdim_val];
                        println!("slice: {:?}", slice);
                        println!("vals: {:?}", vals);
                        for v in vals {
                            let found = binary_search(slice, v, right);
                            inner_vec.push(found as i64);
                        }
                        inner_vec
                    })
                    .flatten()
                    .collect();
                matches
            }
        };
        println!("indices: {:?}", indices);
        indices
    }
}

impl CustomOp2 for SearchSorted {
    fn name(&self) -> &'static str {
        "search-sorted"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let rank_bd = l1.shape().rank();
        let l1_dims = l1.shape().dims().to_vec();
        let l2_dims = l2.shape().dims().to_vec();
        let (innerdim_bd, leadingdims_bd) = l1_dims.split_last().unwrap();
        let (innerdim_val, leadingdims_val) = l2_dims.split_last().unwrap();

        // let innerdim_bd = l1.shape().dims()[rank_bd - 1];
        let numels_bd = l1.shape().elem_count();
        assert!(numels_bd % innerdim_bd == 0);
        let num_rows_bd = numels_bd / innerdim_bd;

        let rank_val = l2.shape().rank();
        let numels_val = l2.shape().elem_count();
        assert!(numels_val % innerdim_val == 0);
        let num_rows_val = numels_val / innerdim_val;

        if (rank_bd != 1 && rank_val != 1) {
            //Check that leading dims are the same
            assert!(leadingdims_bd == leadingdims_val);
        }

        //Check that sorted seq is sorted
        //Check contiguity

        let is_1d_bd = l1.shape().rank() == 1;
        let is_1d_vals = l2.shape().rank() == 1;

        let indices = match_cpu_storage!(s1, s2, |vs: &Vec<_>, values: &Vec<_>| {
            let indices = vs.search_sorted(
                *innerdim_bd,
                values,
                *innerdim_val,
                is_1d_bd,
                is_1d_vals,
                self.right,
            );
            CpuStorage::I64(indices)
        });
        let output_dims = match is_1d_bd {
            true => [&leadingdims_val[..], &[*innerdim_val]].concat(),
            false => [&leadingdims_bd[..], &[*innerdim_val]].concat(),
        };
        let output_shape = Shape::from_dims(&output_dims);

        Ok((indices, output_shape))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    #[test]
    fn test_ss_1d_vals_1d() {
        let device = Device::Cpu;

        let ss: Vec<u32> = (2..=10).step_by(2).collect();
        let ss_shape = Shape::from_dims(&[5]);

        let vals: Vec<u32> = vec![3, 6, 9];
        let vals_shape = Shape::from_dims(&[3]);

        let t1 = Tensor::from_vec(ss, &ss_shape, &device).unwrap();
        let t2 = Tensor::from_vec(vals.clone(), &vals_shape, &device).unwrap();
        let expected_indices: Vec<i64> = vec![1, 2, 4];
        let expected_shape = Shape::from_dims(&[3]);
        //Test left
        let t3 = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();
        let actual_shape = t3.shape();
        let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            actual_indices == expected_indices,
            "Expected {:?}, got {:?}",
            expected_indices,
            actual_indices
        );
        assert!(
            actual_shape.dims() == expected_shape.dims(),
            "Expected shape {:?}, got {:?}",
            expected_shape,
            actual_shape
        );

        //Test right
        let expected_indices: Vec<i64> = vec![1, 3, 4];
        let expected_shape = Shape::from_dims(&[3]);
        let t3 = t1.apply_op2(&t2, SearchSorted { right: true }).unwrap();
        let actual_shape = t3.shape();
        let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            actual_indices == expected_indices,
            "Expected {:?}, got {:?}",
            expected_indices,
            actual_indices
        );
        assert!(
            actual_shape.dims() == expected_shape.dims(),
            "Expected shape {:?}, got {:?}",
            expected_shape,
            actual_shape
        );
    }
    #[test]
    fn test_ss_1d_vals_2d() {
        let device = Device::Cpu;

        let ss: Vec<u32> = (1..10).step_by(2).collect();
        let ss_shape = Shape::from_dims(&[5]);

        let vals: Vec<u32> = vec![3, 6, 9, 3, 6, 9];
        let vals_shape = Shape::from_dims(&[2, 3]);

        // Test left
        let t1 = Tensor::from_vec(ss.clone(), &ss_shape, &device).unwrap();
        let t2 = Tensor::from_vec(vals.clone(), &vals_shape, &device).unwrap();
        let t3 = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();
        let expected_indices: Vec<i64> = vec![1, 3, 4, 1, 3, 4];
        let expected_shape = Shape::from_dims(&[2, 3]);

        let actual_shape = t3.shape();
        let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            actual_indices == expected_indices,
            "Expected {:?}, got {:?}",
            expected_indices,
            actual_indices
        );
        assert!(
            actual_shape.dims() == expected_shape.dims(),
            "Expected shape {:?}, got {:?}",
            expected_shape,
            actual_shape
        );

        // Test right
        let expected_indices: Vec<i64> = vec![2, 3, 5, 2, 3, 5];
        let expected_shape = Shape::from_dims(&[2, 3]);
        let t3 = t1.apply_op2(&t2, SearchSorted { right: true }).unwrap();
        let actual_shape = t3.shape();
        let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            actual_indices == expected_indices,
            "Expected {:?}, got {:?}",
            expected_indices,
            actual_indices
        );
        assert!(
            actual_shape.dims() == expected_shape.dims(),
            "Expected shape {:?}, got {:?}",
            expected_shape,
            actual_shape
        );
    }

    #[test]
    fn test_ss_2d_vals_1d() {
        let device = Device::Cpu;

        let ss: Vec<u32> = [
            (1..10).step_by(2).collect::<Vec<u32>>(),
            (2..11).step_by(2).collect::<Vec<u32>>(),
        ]
        .concat();
        let ss_shape = Shape::from_dims(&[2, 5]);

        let vals: Vec<u32> = vec![3, 6, 9];
        let vals_shape = Shape::from_dims(&[3]);

        // Test left
        let t1 = Tensor::from_vec(ss.clone(), &ss_shape, &device).unwrap();
        let t2 = Tensor::from_vec(vals.clone(), &vals_shape, &device).unwrap();
        let t3 = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();
        let expected_indices: Vec<i64> = vec![1, 3, 4, 1, 2, 4];
        let expected_shape = Shape::from_dims(&[2, 3]);

        let actual_shape = t3.shape();
        let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            actual_indices == expected_indices,
            "Expected {:?}, got {:?}",
            expected_indices,
            actual_indices
        );
        assert!(
            actual_shape.dims() == expected_shape.dims(),
            "Expected shape {:?}, got {:?}",
            expected_shape,
            actual_shape
        );

        // Test right
        let expected_indices: Vec<i64> = vec![2, 3, 5, 1, 3, 4];
        let expected_shape = Shape::from_dims(&[2, 3]);
        let t3 = t1.apply_op2(&t2, SearchSorted { right: true }).unwrap();
        let actual_shape = t3.shape();
        let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            actual_indices == expected_indices,
            "Expected {:?}, got {:?}",
            expected_indices,
            actual_indices
        );
        assert!(
            actual_shape.dims() == expected_shape.dims(),
            "Expected shape {:?}, got {:?}",
            expected_shape,
            actual_shape
        );
    }

    #[test]
    fn test_ss_2d_vals_same_2d() {
        let device = Device::Cpu;

        let ss: Vec<u32> = [
            (1..10).step_by(2).collect::<Vec<u32>>(),
            (2..11).step_by(2).collect::<Vec<u32>>(),
        ]
        .concat();
        let ss_shape = Shape::from_dims(&[2, 5]);

        let vals: Vec<u32> = vec![3, 6, 9, 3, 6, 9];
        let vals_shape = Shape::from_dims(&[2, 3]);

        // Test left
        let t1 = Tensor::from_vec(ss.clone(), &ss_shape, &device).unwrap();
        let t2 = Tensor::from_vec(vals.clone(), &vals_shape, &device).unwrap();
        let t3 = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();
        let expected_indices: Vec<i64> = vec![1, 3, 4, 1, 2, 4];
        let expected_shape = Shape::from_dims(&[2, 3]);

        let actual_shape = t3.shape();
        let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            actual_indices == expected_indices,
            "Expected {:?}, got {:?}",
            expected_indices,
            actual_indices
        );
        assert!(
            actual_shape.dims() == expected_shape.dims(),
            "Expected shape {:?}, got {:?}",
            expected_shape,
            actual_shape
        );

        // Test right
        let expected_indices: Vec<i64> = vec![2, 3, 5, 1, 3, 4];
        let expected_shape = Shape::from_dims(&[2, 3]);
        let t3 = t1.apply_op2(&t2, SearchSorted { right: true }).unwrap();
        let actual_shape = t3.shape();
        let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            actual_indices == expected_indices,
            "Expected {:?}, got {:?}",
            expected_indices,
            actual_indices
        );
        assert!(
            actual_shape.dims() == expected_shape.dims(),
            "Expected shape {:?}, got {:?}",
            expected_shape,
            actual_shape
        );
    }
    #[test]
    fn test_ss_2d_vals_diff_2d() {
        let device = Device::Cpu;

        let ss: Vec<u32> = [
            (1..10).step_by(2).collect::<Vec<u32>>(),
            (2..11).step_by(2).collect::<Vec<u32>>(),
        ]
        .concat();
        let ss_shape = Shape::from_dims(&[2, 5]);
        let vals: Vec<u32> = vec![3, 6, 9, 1, 2, 3];
        let vals_shape = Shape::from_dims(&[2, 3]);

        // Test left
        let t1 = Tensor::from_vec(ss.clone(), &ss_shape, &device).unwrap();
        let t2 = Tensor::from_vec(vals.clone(), &vals_shape, &device).unwrap();
        let t3 = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();

        let expected_indices: Vec<i64> = vec![1, 3, 4, 0, 0, 1];
        let expected_shape = Shape::from_dims(&[2, 3]);
        let actual_shape = t3.shape();
        let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            actual_indices == expected_indices,
            "Expected {:?}, got {:?}",
            expected_indices,
            actual_indices
        );
        assert!(
            actual_shape.dims() == expected_shape.dims(),
            "Expected shape {:?}, got {:?}",
            expected_shape,
            actual_shape
        );

        let t3 = t1.apply_op2(&t2, SearchSorted { right: true }).unwrap();
        let expected_indices: Vec<i64> = vec![2, 3, 5, 0, 1, 1];
        let expected_shape = Shape::from_dims(&[2, 3]);
        let actual_shape = t3.shape();
        let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            actual_indices == expected_indices,
            "Expected {:?}, got {:?}",
            expected_indices,
            actual_indices
        );
        assert!(
            actual_shape.dims() == expected_shape.dims(),
            "Expected shape {:?}, got {:?}",
            expected_shape,
            actual_shape
        );
    }
    macro_rules! test_ss_2d_vals_diff_2d {
        ($t:ty, $ss:expr, $vals:expr) => {
            let device = Device::Cpu;
            let ss: Vec<$t> = $ss;
            let ss_shape = Shape::from_dims(&[2, 5]);
            let vals: Vec<$t> = $vals;
            let vals_shape = Shape::from_dims(&[2, 3]);

            // Test left
            let t1 = Tensor::from_vec::<_, $t>(ss, &ss_shape, &device).unwrap();
            let t2 = Tensor::from_vec::<_, $t>(vals, &vals_shape, &device).unwrap();
            let t3 = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();

            let expected_indices: Vec<i64> = vec![1, 3, 4, 0, 0, 1];
            let expected_shape = Shape::from_dims(&[2, 3]);
            let actual_shape = t3.shape();
            let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                actual_indices == expected_indices,
                "Expected {:?}, got {:?}",
                expected_indices,
                actual_indices
            );
            assert!(
                actual_shape.dims() == expected_shape.dims(),
                "Expected shape {:?}, got {:?}",
                expected_shape,
                actual_shape
            );

            let t3 = t1.apply_op2(&t2, SearchSorted { right: true }).unwrap();
            let expected_indices: Vec<i64> = vec![2, 3, 5, 0, 1, 1];
            let expected_shape = Shape::from_dims(&[2, 3]);
            let actual_shape = t3.shape();
            let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                actual_indices == expected_indices,
                "Expected {:?}, got {:?}",
                expected_indices,
                actual_indices
            );
            assert!(
                actual_shape.dims() == expected_shape.dims(),
                "Expected shape {:?}, got {:?}",
                expected_shape,
                actual_shape
            );
        };
    }

    macro_rules! test_ss_2d_vals_diff_2d_half {
        ($t:ty, $ss:expr, $vals:expr) => {
            let device = Device::Cpu;
            let ss: Vec<$t> = $ss.iter().map(|x| <$t>::from_f32(*x)).collect();
            let ss_shape = Shape::from_dims(&[2, 5]);
            let vals: Vec<$t> = $vals.iter().map(|x| <$t>::from_f32(*x)).collect();
            let vals_shape = Shape::from_dims(&[2, 3]);

            // Test left
            let t1 = Tensor::from_vec::<_, $t>(ss, &ss_shape, &device).unwrap();
            let t2 = Tensor::from_vec::<_, $t>(vals, &vals_shape, &device).unwrap();
            let t3 = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();

            let expected_indices: Vec<i64> = vec![1, 3, 4, 0, 0, 1];
            let expected_shape = Shape::from_dims(&[2, 3]);
            let actual_shape = t3.shape();
            let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                actual_indices == expected_indices,
                "Expected {:?}, got {:?}",
                expected_indices,
                actual_indices
            );
            assert!(
                actual_shape.dims() == expected_shape.dims(),
                "Expected shape {:?}, got {:?}",
                expected_shape,
                actual_shape
            );

            let t3 = t1.apply_op2(&t2, SearchSorted { right: true }).unwrap();
            let expected_indices: Vec<i64> = vec![2, 3, 5, 0, 1, 1];
            let expected_shape = Shape::from_dims(&[2, 3]);
            let actual_shape = t3.shape();
            let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                actual_indices == expected_indices,
                "Expected {:?}, got {:?}",
                expected_indices,
                actual_indices
            );
            assert!(
                actual_shape.dims() == expected_shape.dims(),
                "Expected shape {:?}, got {:?}",
                expected_shape,
                actual_shape
            );
        };
    }
    #[test]
    fn test_ss_2d_vals_diff_2d_u8() {
        test_ss_2d_vals_diff_2d!(
            u8,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9, 1, 2, 3]
        );
    }
    #[test]
    fn test_ss_2d_vals_diff_2d_u32() {
        test_ss_2d_vals_diff_2d!(
            u32,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9, 1, 2, 3]
        );
    }
    #[test]
    fn test_ss_2d_vals_diff_2d_i64() {
        test_ss_2d_vals_diff_2d!(
            i64,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9, 1, 2, 3]
        );
    }
    #[test]
    fn test_ss_2d_vals_diff_2d_fp64() {
        test_ss_2d_vals_diff_2d!(
            f32,
            vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.],
            vec![3., 6., 9., 1., 2., 3.]
        );
    }
    #[test]
    fn test_ss_2d_vals_diff_2d_f16() {
        test_ss_2d_vals_diff_2d_half!(
            f16,
            vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.],
            vec![3., 6., 9., 1., 2., 3.]
        );
    }
    #[test]
    fn test_ss_2d_vals_diff_2d_bf16() {
        test_ss_2d_vals_diff_2d_half!(
            bf16,
            vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.],
            vec![3., 6., 9., 1., 2., 3.]
        );
    }
}
