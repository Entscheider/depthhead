use std::ops::{Index, IndexMut};
use meancov_estimation::Vec3;
use std::collections::HashMap;
use std::cell::UnsafeCell;
use std::collections::hash_map::Iter as HashIter;

pub trait Array3D<T, Idx: Sized>
    : Index<(Idx, Idx, Idx), Output = T> + IndexMut<(Idx, Idx, Idx), Output = T> {
}

/// Use a HashMap to save 3d points.
/// This form of representation should save memory and can handle a very large range
/// of coordinates
pub struct SparseArray3D<T>
    where T: Clone
{
    default: T,
    data: UnsafeCell<HashMap<(i32, i32, i32), T>>, // To make the borrow-checker happy
}

impl<T> Index<(i32, i32, i32)> for SparseArray3D<T>
    where T: Clone
{
    type Output = T;
    fn index<'a>(&'a self, idx: (i32, i32, i32)) -> &'a T {
        unsafe { (*self.data.get()).get(&idx).unwrap_or(&self.default) }
    }
}


impl<T> IndexMut<(i32, i32, i32)> for SparseArray3D<T>
    where T: Clone
{
    fn index_mut<'a>(&'a mut self, idx: (i32, i32, i32)) -> &'a mut T {

        if let Some(reference) = unsafe { (*self.data.get()).get_mut(&idx) } {
            return reference;
        }
        unsafe { self.insert_default_at_index(idx) }
    }
}

impl<T> Array3D<T, i32> for SparseArray3D<T> where T: Clone {}

impl<T> SparseArray3D<T>
    where T: Clone
{
    /// Creates a SparseArray3D of type `T`.
    /// The value `default` will be returnes by this array,
    /// if no value is saved for a request position.
    pub fn with_default_value(default: T) -> SparseArray3D<T> {
        SparseArray3D {
            default: default,
            data: UnsafeCell::new(HashMap::new()),
        }
    }

    #[inline]
    unsafe fn insert_default_at_index(&mut self, idx: (i32, i32, i32)) -> &mut T {
        (*self.data.get()).insert(idx, self.default.clone());
        (*self.data.get()).get_mut(&idx).unwrap()
    }

    pub fn iter(&self) -> HashIter<(i32,i32,i32),T> {
        let map: &HashMap<(i32,i32,i32),T> = unsafe{ &*self.data.get()};
        map.iter()
    }
}

/// Stores 3d points and values within a big array.
pub struct FullArray3D<T> {
    width: u32,
    height: u32,
    depth: u32,
    data: Vec<T>,
}

impl<T> Index<(u32, u32, u32)> for FullArray3D<T> {
    type Output = T;
    fn index<'a>(&'a self, idx: (u32, u32, u32)) -> &'a T {
        let (x, y, z) = idx;
        let x = x as usize;
        let y = y as usize;
        let z = z as usize;
        let w = self.width as usize;
        let h = self.height as usize;
        &self.data[(z * (w * h) + y * w + x)]
    }
}

impl<T> IndexMut<(u32, u32, u32)> for FullArray3D<T> {
    fn index_mut<'a>(&'a mut self, idx: (u32, u32, u32)) -> &'a mut T {
        let (x, y, z) = idx;
        let x = x as usize;
        let y = y as usize;
        let z = z as usize;
        let w = self.width as usize;
        let h = self.height as usize;
        &mut self.data[(z * (w * h) + y * w + x)]
    }
}

impl<T> Array3D<T, u32> for FullArray3D<T> {}

pub struct FullArray3DIter<'a, T>
    where T: 'a
{
    arr: &'a FullArray3D<T>,
    x: u32,
    y: u32,
    z: u32,
}

impl<'a, T> Iterator for FullArray3DIter<'a, T>
    where T: 'a
{
    type Item = (u32, u32, u32, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        let d = self.arr.depth();
        if self.z >= d {
            return None;
        }
        let w = self.arr.width();
        let h = self.arr.height();
        let (x, y, z) = (self.x, self.y, self.z);
        if self.x >= w - 1 {
            self.x = 0;
            if self.y >= h - 1 {
                self.y = 0;
                self.z += 1;
            } else {
                self.y += 1;
            }
        } else {
            self.x += 1;
        }
        Some((x, y, z, &self.arr[(x, y, z)]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.arr.height() as usize * self.arr.width() as usize *
                   self.arr.depth() as usize;
        (size, Some(size))
    }
}

impl<T> FullArray3D<T> {
    /// Creates a `FullArray3D` and initializes its value with the given function.
    ///
    /// # Arguments
    /// * `widht` - width of the array
    /// * `height` - height of the array
    /// * `depth` - depth of the array
    /// * `f` - the coordinate (x,y,z) will be initialized with f(x,y,z)
    pub fn from_fn<F>(width: u32, height: u32, depth: u32, f: F) -> FullArray3D<T>
        where F: Fn(u32, u32, u32) -> T
    {
        let vec: Vec<_> = (0..width * height * depth)
            .map(|i| {
                let z = i / (width * height);
                let rest = i % (width * height);
                let y = rest / width;
                let x = rest % width;
                f(x, y, z)
            })
            .collect();
        FullArray3D {
            width: width,
            height: height,
            depth: depth,
            data: vec,
        }
    }


    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn depth(&self) -> u32 {
        self.depth
    }

    pub fn iter<'a>(&'a self) -> FullArray3DIter<'a, T> {
        FullArray3DIter {
            arr: self,
            x: 0,
            y: 0,
            z: 0,
        }
    }
}

impl<T> FullArray3D<T>
    where T: Clone
{
    /// Creates a FullArray3D and use the given value to initialize every entry.
    /// # Arguments
    /// * `widht` - width of the array
    /// * `height` - height of the array
    /// * `depth` - depth of the array
    /// * `value` - the value every entry in the array will get
    pub fn from_value(width: u32, height: u32, depth: u32, value: T) -> FullArray3D<T> {
        // use std::iter;
        FullArray3D {
            width: width,
            height: height,
            depth: depth,
            // data: iter::repeat(value).collect()
            data: vec![value; (height as usize)*(width as usize)*(depth as usize)],
        }
    }

    /// Sets every entry in the array to the given `value`
    pub fn clear_value(&mut self, value: T) {
        self.data = vec![value; (self.height as usize)*(self.width as usize)*(self.depth as usize)];
    }
}

/// Computes the gaussian kernel for mean shifting.
///
/// # Arguments:
/// * `pos` - the coordinate the returned value is for. (Positive and negative values expected)
/// * `variance` - the variance to use
#[inline]
fn kernel_function(pos: (i32, i32, i32), variance: f32) -> f32 {
    let (x, y, z) = pos;
    let norm = x.pow(2) + y.pow(2) + z.pow(2);
    (-1.0 * (norm as f32) / (2.0 * variance)).exp()
}

impl FullArray3D<f32> {
    /// Creates an array of a gaussian kernel.
    /// The result is a 3d-array whose central position
    /// (neigbour_size / 2, neighbour_size / 2, neighbour_size / 2)
    /// has the value of the gaussian kernel at position (0,0,0).
    ///
    /// # Arguments
    /// * `neighbour_size` - size of the neighbourhood (=> (size x size x size) is the length of the result)
    /// * `variance` - variance used to compute the kernel
    ///
    pub fn build_kernel(neighbour_size: u32, variance: f32) -> FullArray3D<f32> {
        let half = (neighbour_size / 2) as i32;
        FullArray3D::from_fn(neighbour_size, neighbour_size, neighbour_size, |x, y, z| {
            let dx = (x as i32) - half;
            let dy = (y as i32) - half;
            let dz = (z as i32) - half;
            kernel_function((dx, dy, dz), variance)
        })
    }
}

pub trait MeanShift<Idx> {
    /// Performs the mean shifting.
    /// Returns the coordinate of the points with the highest dense.
    ///
    /// # Arguments
    /// * `init` - position to start mean shifting
    /// * `kernel` - kernel to use for mean shifting
    /// * `iterations` - maximum of iterations which should be performed
    fn meanshift(&self,
                 init: (Idx, Idx, Idx),
                 kernel: &FullArray3D<f32>,
                 iterations: u32)
                 -> (Idx, Idx, Idx);
}

use std::ops::Add;
use std::cmp::PartialOrd;

/// Some utility function for numbers which are needed for mean shift
pub trait NumExt: Sized + Copy + PartialOrd {
    type SignedType: NumExt + Add<i32, Output = Self::SignedType>;

    fn min_value() -> Self;
    fn to_signed(self) -> Self::SignedType;
    fn from_signed(num: Self::SignedType) -> Self;
    fn to_f32(self) -> f32;
    fn from_f32(num: f32) -> Self;
}

impl NumExt for i32 {
    type SignedType = Self;
    fn min_value() -> Self {
        i32::min_value()
    }
    fn to_signed(self) -> Self::SignedType {
        self
    }
    fn from_signed(num: Self::SignedType) -> Self {
        num
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f32(num: f32) -> Self {
        num as i32
    }
}

impl NumExt for u32 {
    type SignedType = i32;
    fn min_value() -> Self {
        u32::min_value()
    }
    fn to_signed(self) -> Self::SignedType {
        self as i32
    }
    fn from_signed(num: Self::SignedType) -> Self {
        num as u32
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f32(num: f32) -> Self {
        num as u32
    }
}

impl<T, Idx> MeanShift<Idx> for T
    where T: Array3D<u32, Idx>,
          Idx: NumExt,
          Vec3<Idx::SignedType>: Add<Vec3<i32>, Output = Vec3<Idx::SignedType>>
{

    fn meanshift(&self,
                 init: (Idx, Idx, Idx),
                 kernel: &FullArray3D<f32>,
                 iterations: u32)
                 -> (Idx, Idx, Idx) {
        // Start with the guessed init value
        let mut pos = Vec3::new([init.0.to_signed(), init.1.to_signed(), init.2.to_signed()]);
        // Make some iterations
        for _ in 0..iterations {
            let mut mean_numerator = Vec3::new([0.0, 0.0, 0.0]);
            let mut mean_denominator = 0.0;
            let (w, h, d) = (kernel.width() as i32, kernel.height() as i32, kernel.depth() as i32);
            let halfx = w / 2;
            let halfy = h / 2;
            let halfz = d / 2;
            // Sum over the neighborhood
            for x in -halfx..w - halfx {
                for y in -halfy..h - halfy {
                    for z in -halfz..d - halfz {
                        // Valid coordinates?
                        if pos.0[0] + x < Idx::min_value().to_signed() {
                            continue;
                        };
                        if pos.0[1] + y < Idx::min_value().to_signed() {
                            continue;
                        };
                        if pos.0[2] + z < Idx::min_value().to_signed() {
                            continue;
                        };
                        // Compute absolut position and get voting
                        let offset = Vec3::new([x, y, z]);
                        let abs_pos = pos + offset;
                        let factor = self[(Idx::from_signed(abs_pos.0[0]),
                                           Idx::from_signed(abs_pos.0[1]),
                                           Idx::from_signed(abs_pos.0[2]))];
                        // No voting => Skip
                        if factor == 0 {
                            continue;
                        }

                        // Use kernel to get the influence for the estimated dense
                        // Note: the kernel value for (0,0,0) is in the middle.
                        let influence =
                            kernel[((x + halfx) as u32, (y + halfy) as u32, (z + halfz) as u32)];

                        let abs_pos = Vec3::new([abs_pos.0[0].to_f32(),
                                                 abs_pos.0[1].to_f32(),
                                                 abs_pos.0[2].to_f32()]);
                        // Update values
                        let factor = factor as f32;
                        mean_numerator = mean_numerator + abs_pos * (influence * factor);
                        mean_denominator = mean_denominator + influence * factor;
                    }
                }
            }

            // Very bad situation
            if mean_denominator == 0.0 {
                warn!("Breaking meanshift - zero sum");
                break;
            }

            // Update the position
            let pos_float = mean_numerator / mean_denominator;
            pos = Vec3::new([Idx::SignedType::from_f32(pos_float.0[0]),
                             Idx::SignedType::from_f32(pos_float.0[1]),
                             Idx::SignedType::from_f32(pos_float.0[2])]);
        }
        // Force to be positive
        if pos.0[0] < Idx::min_value().to_signed() {
            pos.0[0] = Idx::min_value().to_signed();
        }
        if pos.0[1] < Idx::min_value().to_signed() {
            pos.0[1] = Idx::min_value().to_signed();
        };
        if pos.0[2] < Idx::min_value().to_signed() {
            pos.0[2] = Idx::min_value().to_signed();
        }
        (Idx::from_signed(pos.0[0]), Idx::from_signed(pos.0[1]), Idx::from_signed(pos.0[2]))
    }
}
