use types::*;
use std::iter::Iterator;
use std::result::Result;
use image;

/// Trait for reading training data from a database.
pub trait DepthReader {
    type Err;
    /// Iterator for the ground truth data of a person within the database
    type Iter: Iterator<Item = Result<DepthTrue, Self::Err>>;

    /// Returns the number of person within the database
    fn person_count(&self) -> Result<usize, Self::Err>;

    /// Returns an iterator for ground truth data of the given person.
    /// `nr` is the number of this person.
    fn person(&self, nr: usize) -> Result<Self::Iter, Self::Err>;

    /// Checks if the database is valid
    fn is_valid(&self) -> Result<bool, Self::Err>;
}

#[derive(Debug, Clone, Copy)]
pub struct HeadTransformation {
    /// The central head position in 3d
    pub pos3d: [f32; 3],
    /// The central head position in 2d (coordinates of the picture)
    pub pos2d: [f32; 2],
    /// The head rotation parameters
    pub rot: [f32; 3],
}

impl HeadTransformation {
    /// Returns the x position in 3d
    pub fn x(&self) -> f32 {
        self.pos3d[0]
    }
    /// Returns the y position in 3d
    pub fn y(&self) -> f32 {
        self.pos3d[1]
    }
    /// Returns the z position in 3d
    pub fn z(&self) -> f32 {
        self.pos3d[2]
    }
    /// Returns the x position in 2d
    pub fn flat_x(&self) -> f32 {
        self.pos2d[0]
    }
    /// Returns the x position in 2d
    pub fn flat_y(&self) -> f32 {
        self.pos2d[1]
    }
    /// Returns the pith rotation
    pub fn pith(&self) -> f32 {
        self.rot[0]
    }
    /// Returns the yaw rotation
    pub fn yaw(&self) -> f32 {
        self.rot[1]
    }
    /// Returns the roll rotation
    pub fn roll(&self) -> f32 {
        self.rot[2]
    }
}

/// Represents the ground truth information for learning a hough forest
pub struct DepthTrue {
    /// The transfomation parameters of the head
    pub trans: HeadTransformation,
    /// The depth image
    pub depth: DepthImage,
    /// The mask image (white where the head is)
    pub mask: MaskImage,
    /// Intrinsic matrix of the camera used
    pub intrinsic: IntrinsicMatrix,
}

/// Ground truth information for learning a gradient boosting decision tree
/// of the head
pub struct PoseTruth {
    /// The transfomation parameters
    pub trans: HeadTransformation,
    /// The gray value image
    pub img: image::GrayImage,
}


