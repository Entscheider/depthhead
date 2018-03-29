use image::{ImageBuffer, Luma, DynamicImage, GenericImage, Primitive, Pixel};
use std::sync::Arc;
use rand::{Rng, ThreadRng, thread_rng};
use std::cell::RefCell;

// TODO: Get rid of this dependency
use hough::houghforest;
use meancov_estimation::{Mat3, Vec3, Vec2};

pub type DepthImage = ImageBuffer<Luma<u16>, Vec<u16>>;
pub type MaskImage = ImageBuffer<Luma<u8>, Vec<u8>>;
pub type PersonImage = DynamicImage;

pub trait MaskOps {
    fn set_in_mask(&mut self, x: u32, y: u32, val: bool);
    fn is_in_mask(&self, x: u32, y: u32) -> bool;
}

impl MaskOps for MaskImage {
    fn set_in_mask(&mut self, x: u32, y: u32, val: bool) {
        self[(x, y)] = if val { Luma([1u8]) } else { Luma([0u8]) }
    }
    fn is_in_mask(&self, x: u32, y: u32) -> bool {
        match self[(x, y)].data[0] {
            0 => false,
            1 => true,
            _ => panic!("Invalid value in mask"),
        }
    }
}

/// Represents a rectangle
#[derive(Serialize, Deserialize, Clone,Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    topleft: [u32; 2],
    bottomright: [u32; 2],
}

impl Rect {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Rect {
        Rect {
            topleft: [x, y],
            bottomright: [x + width, y + height],
        }
    }

    pub fn width(&self) -> u32 {
        self.bottomright[0] - self.topleft[0]
    }
    pub fn height(&self) -> u32 {
        self.bottomright[1] - self.topleft[1]
    }
    pub fn x(&self) -> u32 {
        self.topleft[0]
    }
    pub fn y(&self) -> u32 {
        self.topleft[1]
    }
    pub fn size(&self) -> u64 {
        (self.width() as u64) * (self.height() as u64)
    }

    /// Clone the rectangle and updates its x value
    pub fn with_x(&self, x: u32) -> Rect {
        Rect::new(x, self.y(), self.width(), self.height())
    }
    /// Clone the rectangle and updates its y value
    pub fn with_y(&self, y: u32) -> Rect {
        Rect::new(self.x(), y, self.width(), self.height())
    }
    /// Clone the rectangle and updates its witdth
    pub fn with_width(&self, w: u32) -> Rect {
        Rect::new(self.x(), self.y(), w, self.height())
    }
    /// Clone the rectangle and updates its height
    pub fn with_height(&self, h: u32) -> Rect {
        Rect::new(self.x(), self.y(), self.width(), h)
    }

    /// Returns a new rectangle that is translated and rotated relatively, but still
    /// remains in this rectangle.
    pub fn scale_and_replace(&self, scale: f64, rel_x_offset: f64, rel_y_offset: f64) -> Rect {
        if scale > 1.0 {
            return *self;
        }
        let nw = (self.width() as f64) * scale;
        let nh = (self.height() as f64) * scale;
        let nx = (self.x() as f64) + rel_x_offset * ((self.width() as f64) - nw);
        let ny = (self.y() as f64) + rel_y_offset * ((self.height() as f64) - nh);
        Rect::new(nx as u32, ny as u32, nw as u32, nh as u32)
    }

    /// Returns a iterator which randomly generates some sub-rectangles and thresholds.
    ///
    /// # Arguments
    /// * `min_subrect_factor` - minimal relative size a sub-rectangle may have (between 0.0-1.0)
    /// * `max_subrect_factor` - maximum relative size a sub-rectangle may have (between 0.0-1.0)
    /// * `threshold_range_start` - lowest value a threshold may have.
    /// * `threshold_range_end` - the threshold may not have this value or any higher
    pub fn random_subrect_iterator(&self,
                                   min_subrect_factor: f64,
                                   max_subrect_factor: f64,
                                   threshold_range_start: f64,
                                   threshold_range_end: f64)
                                   -> Option<ThreadRngSubRectThresholdIterator> {
        if min_subrect_factor <= 0.0 || min_subrect_factor > 1.0 || max_subrect_factor <= 0.0 ||
           max_subrect_factor > 1.0 ||
           min_subrect_factor > max_subrect_factor {
            return None;
        }
        let rng = thread_rng();
        Some(RandomSubRectThresholdIterator {
            img_rect: *self,
            min_subrect_factor: min_subrect_factor,
            max_subrect_factor: max_subrect_factor,
            threshold_range_start: threshold_range_start,
            threshold_range_end: threshold_range_end,
            rng: rng,
        })
    }
}

pub type ThreadRngSubRectThresholdIterator = RandomSubRectThresholdIterator<ThreadRng>;

pub struct RandomSubRectThresholdIterator<R>
    where R: Rng
{
    img_rect: Rect,
    min_subrect_factor: f64,
    max_subrect_factor: f64,
    threshold_range_start: f64,
    threshold_range_end: f64,
    rng: R,
}

impl<R> Iterator for RandomSubRectThresholdIterator<R>
    where R: Rng
{
    type Item = (Rect, Rect, f64);
    fn next(&mut self) -> Option<Self::Item> {
        let rng = &mut self.rng;
        let factor = {
            if self.min_subrect_factor == self.max_subrect_factor {
                self.min_subrect_factor
            } else {
                rng.gen_range(self.min_subrect_factor, self.max_subrect_factor)
            }
        };
        let (r1, r2) = {
            let mut offsets = (0..4).map(|_| rng.gen_range(0f64, 1f64));
            (self.img_rect
                 .scale_and_replace(factor, offsets.next().unwrap(), offsets.next().unwrap()),
             self.img_rect
                 .scale_and_replace(factor, offsets.next().unwrap(), offsets.next().unwrap()))
        };
        let th = rng.gen_range(self.threshold_range_start, self.threshold_range_end);
        Some((r1, r2, th))
    }
}



/// This struct represent an immutable section of an image
#[derive(Serialize)]
pub struct SubImage<T: GenericImage> {
    // Skip every value for serialization, because we do not need them

    #[serde(skip_serializing)]
    subrect: Rect,
    #[serde(skip_serializing)]
    img: Arc<T>,
}

pub trait PixelHelper: Copy + 'static {
    fn to_u64(self) -> u64;
}

impl PixelHelper for u32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}
impl PixelHelper for u16 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}
impl PixelHelper for u8 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}
impl<X> PixelHelper for Luma<X>
    where X: PixelHelper + Primitive
{
    fn to_u64(self) -> u64 {
        self.data[0].to_u64()
    }
}

pub type InMutSubImage = SubImage<DepthImage>;

/// Iterate over an every pixel in a subimage.
pub struct SubImageIter<'a, T>
    where T: GenericImage + 'a
{
    subimg: &'a SubImage<T>,
    x: u32,
    y: u32,
}

impl<'a, T> Iterator for SubImageIter<'a, T>
    where T: GenericImage
{
    type Item = T::Pixel;

    #[inline]
    fn next(&mut self) -> Option<T::Pixel> {
        let w = self.subimg.subrect.width();
        if self.x >= w {
            return None;
        }
        let h = self.subimg.subrect.height();
        let x = self.subimg.subrect.x() + self.x;
        let y = self.subimg.subrect.y() + self.y;
        if self.y >= h - 1 {
            self.y = 0;
            self.x += 1;
        } else {
            self.y += 1;
        }
       #[cfg(feature="reduce_bound_checks")]
        {
            unsafe { Some(self.subimg.img.unsafe_get_pixel(x, y)) }
        }
       #[cfg(not(feature="reduce_bound_checks"))]
        {
            Some(self.subimg.img.get_pixel(x, y))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.subimg.subrect.size() as usize;
        (size, Some(size))
    }
}

impl<T: GenericImage> SubImage<T> where T::Pixel: 'static
    //where T::Pixel: PixelHelper
{
    /// Creates a subimage of the image `img` using the rectangle which are
    /// described by `x`, `y`, `w` (width), `h` (height).
    pub fn new(x: u32, y: u32, w: u32, h: u32, img: Arc<T>) -> Option<SubImage<T>> {
        if x + w > img.width() || y + h > img.height() {
            return None;
        }
        Some(SubImage {
            subrect: Rect::new(x, y, w, h),
            img: img,
        })
    }

    /// Returns an iterator which iterates over every pixel
    /// of this subimage.
    pub fn iter<'a>(&'a self) -> SubImageIter<'a, T> {
        SubImageIter {
            subimg: self,
            x: 0,
            y: 0,
        }
    }

    /// Copy the data of this subimage and creates a new independent Image.
    pub fn to_image(&self) -> ImageBuffer<T::Pixel,Vec<<T::Pixel as Pixel>::Subpixel>>{
        let mut out = ImageBuffer::new(self.subrect.width(), self.subrect.height());
        let begx = self.subrect.x(); 
        let begy = self.subrect.y(); 
        for y in 0..self.subrect.width(){
            for x in 0..self.subrect.height(){
                let p = self.img.get_pixel(begx+x,begy+y);
                out.put_pixel(x,y,p);
            }
        }
        out
    }


    /// Returns true if this subimage is the whole image,
    /// false otherwise.
    fn is_whole_image(&self) -> bool{
        self.subrect.x() == 0 && self.subrect.y() == 0
        && self.subrect.width() == self.img.width() 
        && self.subrect.height() == self.img.height()
    }
}

impl<P: Pixel> SubImage<ImageBuffer<P, Vec<P::Subpixel>>> where P: 'static{
    /// Clone the content of this subimage into a independent one.
    /// Then a canonical subimage (representing the whole new image) wrap it
    /// and is returned.
    /// If this subimage is the whole image no data will be copied.
    pub fn to_cropped_subimage(&self) -> SubImage<ImageBuffer<P,Vec<P::Subpixel>>>{
      if self.is_whole_image() {
          SubImage{subrect: self.subrect.clone(), img: self.img.clone()}
       }else{
          let img = Arc::new(self.to_image());
          assert_eq!(img.width(), self.subrect.width());
          assert_eq!(img.height(), self.subrect.height());
          SubImage{subrect: Rect::new(0,0,img.width(),img.height()), img: img}
       }
    }
}

impl<T: GenericImage> houghforest::HoughForestImage for SubImage<T>
    where T::Pixel: PixelHelper
{
    fn average_value_in_rect(&self, rect: Rect) -> f64 {
        let mut sum = 0u64;
        let mut count = 0u64;
        for y in (rect.y() + self.subrect.y())..(self.subrect.y() + rect.y() + rect.height()) {
            for x in (rect.x() + self.subrect.x())..(self.subrect.x() + rect.x() + rect.width()) {
                count += 1;
                #[cfg(feature="reduce_bound_checks")]
                {
                    unsafe {
                        sum += self.img.unsafe_get_pixel(x, y).to_u64();
                    }
                }
                #[cfg(not(feature="reduce_bound_checks"))]
                {
                    sum += self.img.get_pixel(x, y).to_u64();
                }
            }
        }
        if count == 0 {
            return 0f64;
        }
        (sum as f64) / (count as f64)
    }
}

/// Does a sliding window over the image `img` and calls f for every resulting
/// subimage.
///
/// # Arguments
/// * `img` - the image to iterate over
/// * `subimage_widht' - the width of a subimage
/// * `subimage_height`- the height of a subimage
/// * `step_x` - the step size in the x-direction
/// * `step_y` - the step size in the y-direction
/// * `f` - a  function that is called for every subimage with the current midpoint position
pub fn iterate_subimage<F, T: GenericImage>(img: Arc<T>,
                                            subimage_width: u32,
                                            subimage_height: u32,
                                            step_x: u32,
                                            step_y: u32,
                                            mut f: F)
    where F: FnMut(SubImage<T>,
                   // mid_x
                   u32,
                   // mid_y:
                   u32),
          T::Pixel: PixelHelper
{
    let left_w = subimage_width / 2;
    let right_w = subimage_width - left_w;
    let left_h = subimage_height / 2;
    let right_h = subimage_height - left_h;
    let mut y = left_h;
    while y < img.height() - right_h {
        let mut x = left_w;
        while x < img.width() - right_w {
            let subimg = SubImage::new(x - left_w,
                                       y - left_h,
                                       subimage_width,
                                       subimage_height,
                                       img.clone())
                .unwrap();
            f(subimg, x, y);
            x += step_x;
        }
        y += step_y;
    }
}



/// Permute a vector randomly
pub fn rand_perm<T>(v: &mut Vec<T>, rng: &mut ThreadRng) {
    let len = v.len();
    let slice = v.as_mut_slice();
    for _ in 0..len {
        let (a, b) = (rng.gen_range(0, len), rng.gen_range(0, len));
        if a == b {
            continue;
        }
        slice.swap(a, b);
    }
}


type IFloat = f32;
/// Represent a intrinsic matrix
#[derive(Debug)]
pub struct IntrinsicMatrix(pub Mat3<IFloat>, RefCell<Option<Mat3<IFloat>>>);
impl Clone for IntrinsicMatrix {
    fn clone(&self) -> Self {
        IntrinsicMatrix::new(self.0)
    }
}

impl IntrinsicMatrix {
    pub fn new<T: Into<Mat3<IFloat>>>(mat: T) -> IntrinsicMatrix {
        IntrinsicMatrix(mat.into(), RefCell::new(None))
    }

    /// Returns the default intrinsic matrix for a kinect
    pub fn default_kinect_intrinsic() -> IntrinsicMatrix {
        IntrinsicMatrix::new([[560.0, 0.0, 320.0], [0.0, 560.0, 240.0], [0.0, 0.0, 1.0]])
    }


    /// Maps the space coordinate to the corresponding image coordinate using this intrinsic matrix.
    pub fn space_to_img_coord<T: Into<Vec3<IFloat>>>(&self, space_coord: T) -> Vec2<IFloat> {
        let res = self.0 * space_coord.into();
        let c = res.0[2];
        Vec2::new([res[0] / c, res[1] / c])
    }

    /// Maps the image coordinate to the corresponding space coordinate using this intrinsic matrix
    /// and the given z - value.
    pub fn img_to_space_coord<T: Into<Vec2<IFloat>>>(&self, img_coord: T, z: IFloat) -> Vec3<f32> {
        use meancov_estimation::MatrixFunc;
        let v2 = img_coord.into();
        let v3 = Vec3::new([v2.0[0], v2.0[1], 1.0]);
        let inv = {
            if self.1.borrow().is_none() {
                *self.1.borrow_mut() = Some(self.0.inv());
            }
            self.1.borrow().unwrap()
        };
        let res = inv * v3;
        let c = z / res.0[2];
        res * c
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use meancov_estimation::{Mat3, Vec3};

    #[test]
    fn test_rect() {
        let rect = Rect::new(1, 2, 10, 20);
        assert_eq!(rect.x(), 1);
        assert_eq!(rect.y(), 2);
        assert_eq!(rect.width(), 10);
        assert_eq!(rect.height(), 20);
        assert_eq!(rect.size(), 10 * 20);
        assert_eq!(rect.scale_and_replace(1.0, 0.0, 0.0), rect);
        assert_eq!(rect.scale_and_replace(1.0, 0.5, 0.5), rect);
        assert_eq!(rect.scale_and_replace(0.5, 0.0, 0.0),
                   Rect::new(1, 2, 5, 10));
        assert_eq!(rect.scale_and_replace(0.5, 1.0, 1.0),
                   Rect::new(6, 12, 5, 10));
        assert_eq!(rect.scale_and_replace(0.25, 0.5, 0.2),
                   Rect::new(4, 5, 2, 5));
        assert_eq!(rect.with_x(5), Rect::new(5, 2, 10, 20));
        assert_eq!(rect.with_y(4), Rect::new(1, 4, 10, 20));
        assert_eq!(rect.with_width(100), Rect::new(1, 2, 100, 20));
        assert_eq!(rect.with_height(30), Rect::new(1, 2, 10, 30));
    }

    #[test]
    fn test_intrinsic() {
        let inmatrix = Mat3([[22.0, 11.4, 12.11], [2.1, 4.1, 2.11], [1.3, 3.1, 19.0]]);
        let inmatrix = IntrinsicMatrix::new(inmatrix);
        let p3 = Vec3([11.0, 12.0, 32.2]);
        let p2 = inmatrix.space_to_img_coord(p3);
        assert!((p2.0[0] - 1.15896578).abs() < 0.0001);
        assert!((p2.0[1] - 0.21143073).abs() < 0.0001);
        let p3v2 = inmatrix.img_to_space_coord(p2, p3.0[2]);
        assert!((p3v2.0[0] - p3.0[0]).abs() < 0.0001);
        assert!((p3v2.0[1] - p3.0[1]).abs() < 0.0001);
        assert!((p3v2.0[2] - p3.0[2]).abs() < 0.0001);
    }
}
