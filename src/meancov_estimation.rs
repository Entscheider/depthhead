use std::ops::{Add, Mul, Sub, Deref, Div, Neg};
use std::iter::Sum;
use std::mem;

/// Trait for n-dimensional Vectors
pub trait Vecn: Add<Output = Self> + Sized + Copy {
    type MatrixOut;
    fn dim(&self) -> usize;
    fn transposed_matrix(&self) -> Self::MatrixOut;
}

/// Trait which defines function which are needed for matrices.
pub trait MatrixFuncSimple{
    type Output;
    fn trace(&self) -> Self::Output;
}

/// Creates a n-dimensional vector and a nxn-matrix
/// and implements useful functions.
///
/// # Arguments:
/// * `$Vec` name of the created vector type
/// * `$Mat` name of the creates matrix type
/// * `$n` number of the dimension
///
/// # Example:
/// `VecMat!(Vec2, Mat2, 2)`
///
macro_rules! VecMat {
    ($Vec: ident, $Mat: ident, $n: expr) => {

        #[derive(Serialize, Deserialize, Debug)]
        pub struct $Vec<T>(pub [T; $n]);

        impl<T> $Vec<T>{
            pub fn new(coord: [T;$n]) -> $Vec<T>{ $Vec(coord) }
        }

        impl<T> Into<[T;$n]> for $Vec<T>{
            fn into(self) -> [T;$n]{
                self.0
            }
        }

        // impl<T> Into<$Vec<T>> for [T;$n]{
        impl Into<$Vec<f32>> for [f32;$n]{
            fn into(self) -> $Vec<f32>{
                $Vec(self)
            }
        }

        impl<T> Deref for $Vec<T>{
            type Target = [T;$n];

            fn deref(&self) -> &[T;$n]{
                &self.0
            }
        }

        impl<T> Copy for $Vec<T> where T: Copy +Sized{ }
        impl<T> Clone for $Vec<T> where T: Copy +Sized{
            fn clone(&self) -> $Vec<T>{
                *self
            }
        }

        #[derive(Serialize, Deserialize, Debug)]
        pub struct $Mat<T>(pub [[T;$n];$n]);
        impl<T> Copy for $Mat<T> where T: Copy +Sized{ }
        impl<T> Clone for $Mat<T> where T: Copy +Sized{
            fn clone(&self) -> $Mat<T>{
                *self
            }
        }

        impl<T> Add for $Vec<T> where T: Add<Output=T>+Copy{
            type Output = $Vec<T>;
            fn add(self, rhs: $Vec<T>) -> Self::Output{
                let mut me = self;
                for i in 0..$n{
                    me.0[i] = me.0[i] + rhs.0[i];
                }
                me
            }
        }

        impl<T> Sum for $Vec<T> where T: Add<Output=T>+Sum<T>+Copy{
            fn sum<I>(iter: I) -> Self where I: Iterator<Item=Self>{
                let mut iter = iter;
                match iter.next(){
                    Some(first) => iter.fold(first, Add::add),
                    None => {
                        let tmp: Vec<T> = vec![];
                        let val = tmp.into_iter().sum();
                        $Vec::new([val; $n])
                    }
                }
            }
        }
        impl<'a, T> Sum<&'a $Vec<T>> for $Vec<T> where T: Add<Output=T>+Sum+Copy{
            fn sum<I>(iter: I) -> Self where I: Iterator<Item=&'a Self>{
                iter.map(|x| *x).sum()
            }
        }

        impl<T> Sub for $Vec<T> where T: Sub<Output=T>+Copy{
            type Output = $Vec<T>;
            fn sub(self, rhs: $Vec<T>) -> Self::Output{
                let mut me = self;
                for i in 0..$n{
                    me.0[i] = me.0[i] - rhs.0[i];
                }
                me
            }
        }
        impl<T> Div<T> for $Vec<T> where T: Div<T, Output=T>+Copy{
            type Output = $Vec<T>;
            fn div(self, rhs: T) -> $Vec<T>{
                let mut me = self;
                for i in 0..$n{
                    me.0[i] = me.0[i] / rhs;
                }
                me
            }
        }

        // impl<T> Mul<f64> for $Vec<T> where T: Mul<f64, Output=T> + Copy{
        //     type Output = $Vec<T>;
        //     fn mul(self, rhs: f64) -> $Vec<T>{
        //         let mut me = self;
        //         for i in 0..$n{
        //             me.0[i] = me.0[i] * rhs;
        //         }
        //         me

        //     }
        // }
        impl<T> Mul<T> for $Vec<T> where T: Mul<T, Output=T> + Copy{
            type Output = $Vec<T>;
            fn mul(self, rhs: T) -> $Vec<T>{
                let mut me = self;
                for i in 0..$n{
                    me.0[i] = me.0[i] * rhs;
                }
                me

            }
        }
        impl<T> Add for $Mat<T> where T: Add<Output=T> + Copy{
            type Output = $Mat<T>;
            fn add(self, rhs: $Mat<T>) -> Self::Output{
                let mut me = self;
                for i in 0..$n{
                    for j in 0..$n{
                        me.0[i][j] = me.0[i][j] + rhs.0[i][j];
                    }
                }
                me
            }
        }

        impl<T> Div<T> for $Mat<T> where T: Div<T, Output=T>+Copy{
            type Output = $Mat<T>;
            fn div(self, rhs: T) -> $Mat<T>{
                let mut me = self;
                for i in 0..$n{
                    for j in 0..$n{
                        me.0[j][i] = me.0[j][i] / rhs;
                    }
                }
                me
            }
        }

        impl<T> Mul<T> for $Mat<T> where T: Mul<T, Output=T>+Copy{
            type Output = $Mat<T>;
            fn mul(self, rhs: T) -> $Mat<T>{
                let mut me = self;
                for i in 0..$n{
                    for j in 0..$n{
                        me.0[j][i] = me.0[j][i] * rhs;
                    }
                }
                me
            }
        }

        // impl<T> Mul<f64> for $Mat<T> where T: Add<Output=T> + Mul<f64, Output=T> + Copy{
        //     type Output = $Mat<T>;
        //     fn mul(self, rhs: f64) -> Self::Output{
        //         let mut me = self;
        //         for i in 0..$n{
        //             for j in 0..$n{
        //                 me.0[i][j] = me.0[i][j] * rhs;
        //             }
        //         }
        //         me
        //     }
        // }

        impl<T> Mul<$Vec<T>> for $Mat<T> where T: Add<Output=T> + Mul<T, Output=T> + Copy{
            type Output=$Vec<T>;
            fn mul(self, rhs: $Vec<T>) -> $Vec<T>{
                unsafe{
                    let mut res = $Vec(mem::uninitialized());
                    for j in 0..$n{
                        let mut tmp = rhs.0[0] * self.0[j][0];
                        for i in 1..$n{
                            tmp = tmp+ rhs.0[i] * self.0[j][i];
                        }
                        res.0[j] = tmp;
                    }
                    res
                }
            }
        }

        impl<T> Sub for $Mat<T> where T: Sub<T, Output=T> + Copy{
            type Output = $Mat<T>;
            fn sub(self, rhs: $Mat<T>) -> Self::Output{
                let mut me = self;
                for i in 0..$n{
                    for j in 0..$n{
                        me.0[i][j] = me.0[i][j] - rhs.0[i][j];
                    }
                }
                me

            }
        }

        impl<T> Into<[[T;$n];$n]> for $Mat<T>{
            fn into(self) -> [[T;$n];$n]{
                self.0
            }
        }

        // impl<T> Into<$Mat<T>> for [[T;$n];$n]{
        impl Into<$Mat<f32>> for [[f32;$n];$n]{
            fn into(self) -> $Mat<f32>{
                $Mat(self)
            }
        }

        // impl<T> Into<$Mat<T>> for [T;$n*$n]{
        impl Into<$Mat<f32>> for [f32;$n*$n]{
            fn into(self) -> $Mat<f32>{
                unsafe {
                    let mut res = $Mat(mem::uninitialized());
                    for j in 0..$n{
                        for i in 0..$n{
                            res.0[j][i] = self[j*$n+i];
                        }
                    }
                    res
                }
            }
        }

        impl<T> MatrixFuncSimple for $Mat<T> where T: Sum + Copy{
            type Output = T;
            fn trace(&self) -> T{
                (0..$n).map(|i| self.0[i][i]).sum()
            }
        }
        
        impl<T> Vecn for $Vec<T> where T: Add<Output=T> + Mul<T,Output=T>+ Copy{

            type MatrixOut = $Mat<T>;
            fn dim(&self) -> usize{ $n}
            fn transposed_matrix(&self) -> Self::MatrixOut{
                unsafe{
                    let mut res = $Mat(mem::uninitialized());
                    for i in 0..$n{
                        for j in 0..$n{
                            res.0[i][j] = self.0[i] * self.0[j];
                        }
                    }
                    res
                }
            }
        }
    }
}

VecMat!(Vec2, Mat2, 2);
VecMat!(Vec3, Mat3, 3);

/// Implement f64 division and multiplication for $T<f32>
macro_rules! implementf64forf32 {
    ($T: ident) => {
        impl Div<f64> for $T<f32>{
            type Output=Self;
            fn div(self, other: f64)->Self{
                self / other as f32
            }
        }
        impl Mul<f64> for $T<f32>{
            type Output=Self;
            fn mul(self, other: f64)->Self{
                self * other as f32
            }
        }
    }
}
implementf64forf32!(Mat3);
implementf64forf32!(Vec3);

VecMat!(Vec4, Mat4, 4);

/// This functions are needed for estimate_mean_conv
pub trait MatrixFunc {
    type Output;
    /// Computes the determinant of a matrix
    fn det(&self) -> Self::Output;
    /// Computes the inverse of a matrix
    fn inv(&self) -> Self;
}

impl<T> MatrixFunc for Mat2<T>
where T: Sub<T, Output = T> + Mul<T, Output = T> + Add<Output = T> + Div<T, Output=T>+ Copy + Neg<Output=T>
{
    type Output = T;
    fn det(&self) -> T {
        self.0[0][0] * self.0[1][1] - self.0[0][1] * self.0[1][0]
    }

    fn inv(&self)  -> Self{
        let (a,b) = (self.0[0][0], self.0[0][1]);
        let (c,d) = (self.0[1][0], self.0[1][1]);
        Mat2([[d,-b],[-c,a]]) / self.det()
    }
}

impl<T> MatrixFunc for Mat3<T>
where T: Sub<T, Output = T> + Mul<T, Output = T> + Add<Output = T> + Div<T, Output=T> + Copy + Neg<Output=T>
{
    type Output = T;
    fn det(&self) -> T {
        self.0[0][0] * (self.0[1][1] * self.0[2][2] - self.0[1][2] * self.0[2][1]) -
            self.0[1][0] * (self.0[0][1] * self.0[2][2] - self.0[0][2] * self.0[2][1]) +
            self.0[2][0] * (self.0[0][1] * self.0[1][2] - self.0[0][2] * self.0[1][1])
    }
    fn inv(&self) -> Mat3<T>{
        let (a,b,c) = (self.0[0][0], self.0[0][1], self.0[0][2]);
        let (d,e,f) = (self.0[1][0], self.0[1][1], self.0[1][2]);
        let (g,h,i) = (self.0[2][0], self.0[2][1], self.0[2][2]);
        return Mat3([ [e*i-f*h,c*h-b*i,b*f-c*e],
                    [f*g-d*i,a*i-c*g,c*d-a*f],
                    [d*h-e*g,b*g-a*h,a*e-b*d]])
            / self.det();
    }
}


// TODO: Use iterators instead of slices
/// Estimate the mean value and the covariance matrix
/// of the given set.
pub fn estimate_mean_cov<V, M>(set: &[V]) -> Option<(V, M)>
    where V: Vecn<MatrixOut = M> + Sub<V,Output=V> + Div<f64, Output = V>,
          M: Add<M, Output = M> + Div<f64, Output = M> + Sub<M, Output = M>
{
    if set.is_empty() {
        return None;
    }
    let mut mean = set[0];
    for i in 1..set.len() {
        mean = mean + set[i];
    }
    let mean = mean  / set.len() as f64;

    let mut cov = (set[0] - mean).transposed_matrix();
    for i in 1..set.len(){
       cov = cov + (set[i]-mean).transposed_matrix();
    }
    let cov = cov / (set.len() - 1) as f64;
    Some((mean, cov))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper trait for assertion with tolerance
    trait FEq {
        fn feq(a: Self, b: Self, tolerance: f64) -> bool;
    }

    impl FEq for f64 {
        fn feq(a: Self, b: Self, tolerance: f64) -> bool {
            (a - b).abs() < tolerance
        }
    }

    macro_rules! FEqVec {
        ($Vec: ident, $n: expr) => (
            impl FEq for $Vec<f64>{
                fn feq(a: Self, b: Self, tolerance: f64) -> bool{
                    for i in 0..$n{
                        if !FEq::feq(a.0[i],b.0[i],tolerance) {
                            return false;
                        }
                    }
                    true
                }
            }
            )
    }

    macro_rules! FEqMat {
        ($Mat: ident, $n: expr) => (
            impl FEq for $Mat<f64>{
                fn feq(a: Self, b: Self, tolerance: f64) -> bool{
                    for j in 0..$n{
                        for i in 0..$n{
                            if !FEq::feq(a.0[j][i],b.0[j][i],tolerance) {
                                return false;
                            }
                        }
                    }
                    true
                }
            }
            )
    }

    FEqMat!(Mat2, 2);
    FEqMat!(Mat3, 3);
    FEqVec!(Vec2, 2);
    FEqVec!(Vec3, 3);

    /// Assert with tolerance
    macro_rules! assert_feq {
        ($a: expr, $b: expr, $tol: expr) => (
            // assert!(($b-$a).abs() < $tol)
            assert!(FEq::feq($a,$b,$tol));
            )
    }

    //     #[test]
    //     fn test_mean_conv2(){
    //         let v: Vec<_> = (0..10).map(|j| Vec2([j as f64,(2*j) as f64])).collect();
    //         let (m,c) = estimate_mean_cov(&v[..]).unwrap();
    //         assert_eq!( m.0[0], 4.5f64);
    //         assert_eq!( m.1[1], 9f64);
    //         assert_feq!( c.0[0][0], 9.166, 0.01 );
    //         assert_feq!( c.0[1][1], 36.666, 0.01);
    //     }

    #[test]
    fn test_mean_cov2() {
        let v = [Vec2([2f64, 6f64]), Vec2([3f64, 4f64]), Vec2([3f64, 8f64]), Vec2([4f64, 6f64])];
        let (m, c) = estimate_mean_cov(&v[..]).unwrap();
        assert_feq!(m.0[0], 3f64, 0.001);
        assert_feq!(m.0[1], 6f64, 0.001);
        assert_feq!(c.0[0][0], 0.66666f64, 0.001);
        assert_feq!(c.0[1][1], 2.6666f64, 0.001);
        assert_feq!(c.0[0][1], 0f64, 0.001);
        assert_feq!(c.0[1][0], 0f64, 0.001);
    }
    #[test]
    fn test_mean_cov3() {
        let v = [Vec3([1.0,2.0,3.0]), Vec3([1.2,1.0,3.2]), Vec3([-1.0,-2.1,3.0]), Vec3([0.0,1.0,0.0])];
        let (m, c) = estimate_mean_cov(&v[..]).unwrap();
        assert_feq!(m.0[0], 0.3f64, 0.001);
        assert_feq!(m.0[1], 0.475f64, 0.001);
        assert_feq!(m.0[2], 2.3f64, 0.001);
        assert_feq!(c.0[0][0], 1.0266666, 0.001);
        assert_feq!(c.0[1][1], 3.1691666, 0.001);
        assert_feq!(c.0[2][2], 2.36, 0.001);
        assert_feq!(c.0[0][1], 1.576666f64, 0.001);
        assert_feq!(c.0[1][0], 1.576666f64, 0.001);
        assert_feq!(c.0[0][2], 0.36, 0.001);
        assert_feq!(c.0[2][0], 0.36, 0.001);
        assert_feq!(c.0[1][2], -0.49, 0.001);
        assert_feq!(c.0[2][1], -0.49, 0.001);

        let v = [Vec3([-32.48225021362305, 24.72743034362793, -3.9425208568573]), Vec3([-25.82341957092285, -25.307233810424805, 1.955498456954956]), Vec3([35.37421417236328, -18.529083251953125, -5.888242721557617]), Vec3([43.30265808105469, -60.69481658935547, -15.176074028015137]), Vec3([32.97354507446289, -7.171285629272461, -3.897606134414673])];
        let (_, c) = estimate_mean_cov(&v[..]).unwrap();
        assert_feq!(c.0[0][0], 1341.63076476, 0.001);
        assert_feq!(c.0[1][1], 954.396794746, 0.001);
        assert_feq!(c.0[2][2],  38.573568346, 0.001);
        assert_feq!(c.0[0][1],  -685.47821414, 0.001);
        assert_feq!(c.0[1][0],  -685.478214144, 0.001);
        assert_feq!(c.0[0][2], -157.223241746, 0.001);
        assert_feq!(c.0[2][0],  -157.223241746, 0.001);
        assert_feq!(c.0[1][2], 110.60252659, 0.001);
        assert_feq!(c.0[2][1],110.60252659, 0.001);
        assert_feq!(c.det(), 15102509.494226849, 0.0001);
    }

    #[test]
    fn test_det_2_3_trace() {
        let m2 = Mat2([[1f64, 3f64], [2f64, 44f64]]);
        assert_feq!(m2.det(), 38f64, 0.001);
        assert_feq!(m2.trace(), 45f64, 0.001);
        let m3 = Mat3([[1f64, 3f64, 22f64], [2f64, 44f64, 1f64], [2f64, 0f64, 3.1]]);
        assert_feq!(m3.det(), -1812.199, 0.001);
        assert_feq!(m3.trace(), 48.1f64, 0.001);
    }

    #[test]
    fn test_inverse() {
        let m2 = Mat2([[4.3, 2.4], [2.1, 424.11]]);
        assert_feq!(m2.det(), 1818.633, 0.001);
        assert_feq!(m2.inv(),
                    Mat2([[0.23320263, -0.00131967], [-0.00115471, 0.00236441]]),
                    0.001);
        let m3 = Mat3([[2.3, 1.4, 12.11], [2.1, 44.11, 2.11], [1.3, 4.1, 19.0]]);
        assert_feq!(m3.inv(),
                    Mat3([[0.65540671, 0.01821446, -0.4197583],
                          [-0.02936075, 0.02209108, 0.01626034],
                          [-0.03850788, -0.00601328, 0.07784307]]),
                    0.001);
    }

    #[test]
    fn test_mat_vec_mul() {
        let m2 = Mat2([[1.3, 12.1], [3.1, 33.1]]);
        let v2 = Vec2([11.0, 12.0]);
        assert_feq!(m2 * v2, Vec2([159.5, 431.3]), 0.001);
        let m3 = Mat3([[1.3, 12.1, 2.3], [3.1, 33.1, 14.1], [1.0, 2.0, 3.0]]);
        let v3 = Vec3([11.0, 12.0, 32.2]);
        assert_feq!(m3 * v3, Vec3([233.56, 885.32, 131.6]), 0.001);
    }

    #[test]
    fn test_transposed_matrix(){
        let v = Vec3([2.0,1.1,4.3]);
        assert_feq!(v.transposed_matrix(), Mat3([[4.0,2.2,8.6],[2.2,1.21,4.73],[8.6,4.73,18.49]]),0.001);
        let v = Vec2([3.1,2.2]);
        assert_feq!(v.transposed_matrix(), Mat2([[9.61,6.82],[6.82,4.84]]), 0.001);
    }
}
