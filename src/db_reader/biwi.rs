use byteorder::ReadBytesExt;
use byteorder;
use std::io::{Read, Result as IOResult, BufRead, BufReader};
use std::result::Result;
use super::reader::HeadTransformation as GT;
use types::*;
use image::Luma;
use image;
use super::reader::*;
use std::fs::DirEntry;
use regex::Regex;
use std::num::ParseFloatError;

use std::fs::{read_dir, File};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::cell::Cell;
use std::io;
use std::error::Error;
use std::fmt;

pub type BResult<T> = Result<T, BiwiReadError>;

// Parse Functions

/// Parse Calibration-File and extract the intrinsic matrix
fn read_cal<R: BufRead>(mut reader: R) -> BResult<IntrinsicMatrix> {
    use std::f32;
    use std::str::FromStr;
    lazy_static! {
        static ref FLOAT: Regex = Regex::new(r"(\d+[\.\d+]*)").unwrap();
    }
    let mut res = [[0.0; 3]; 3];
    let mut text = String::new();
    for j in 0..3 {
        let mut found = 0;
        reader.read_line(&mut text)?;
        for c in FLOAT.captures_iter(&*text) {
            let float_text = match c.get(1) {
                Some(x) => x.as_str(),
                None => {
                    return Err(BiwiReadError::Unspecific(Some("Unsupported Calibration-File"
                        .to_string())))
                }
            };
            res[j][found] = f32::from_str(float_text)?;
            if found == 3 {
                return Err(BiwiReadError::Unspecific(Some("Unsupported Calibration-File"
                    .to_string())));

            }
            found += 1;
        }
        if found != 3 {
            return Err(BiwiReadError::Unspecific(Some("Unsupported Calibration-File".to_string())));
        }
        text.clear();
    }
    Ok(IntrinsicMatrix::new(res))
}

/// Parse the ground truth file
fn read_gt<R: Read>(mut reader: R, intrinsic: &IntrinsicMatrix) -> IOResult<GT> {
    type End = byteorder::LittleEndian;
    let mut res = [0f32; 6];
    for i in 0..6 {
        res[i] = reader.read_f32::<End>()?;
    }
    let p3 = [res[0], res[1], res[2]];
    let p2 = intrinsic.space_to_img_coord(p3);

    return Ok(GT {
        pos3d: p3,
        pos2d: p2.into(),
        rot: [res[3], res[4], res[5]],
    });
}


/// Parse a file to extract a depth image
pub fn read_depth<R: Read>(mut reader: R) -> IOResult<DepthImage> {
    type End = byteorder::LittleEndian;
    let width = reader.read_u32::<End>()? as u32;
    let height = reader.read_u32::<End>()? as u32;
    let mut p = 0usize;
    let mut depth = DepthImage::new(width, height);
    {
        let mut it = depth.pixels_mut();
        while p < (width * height) as usize {
            let num_empty = reader.read_u32::<End>()?;
            for _ in 0..num_empty {
                *it.next().unwrap() = Luma([0u16]);
            }
            let num_nonempty = reader.read_u32::<End>()?;
            for _ in 0..num_nonempty {
                let next = reader.read_u16::<End>()?;
                *it.next().unwrap() = Luma([next]);
            }
            p += num_empty as usize + num_nonempty as usize;
        }
    }
    Ok(depth)
}

// Error Definitions

#[derive(Debug)]
pub enum BiwiReadError {
    DirectoryExpected(Option<String>),
    IoError(io::Error),
    ParseFloatError(ParseFloatError),
    InvalidDirStructure(Option<String>),
    ImageError(image::ImageError),
    Unspecific(Option<String>),
}

use std::convert::From;
impl From<io::Error> for BiwiReadError {
    fn from(err: io::Error) -> Self {
        BiwiReadError::IoError(err)
    }
}

impl From<ParseFloatError> for BiwiReadError {
    fn from(err: ParseFloatError) -> Self {
        BiwiReadError::ParseFloatError(err)
    }
}
impl From<image::ImageError> for BiwiReadError {
    fn from(err: image::ImageError) -> Self {
        BiwiReadError::ImageError(err)
    }
}

impl fmt::Display for BiwiReadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            BiwiReadError::DirectoryExpected(Some(ref x)) => {
                write!(f, "Directory not found. Dirname: {}", x)
            }
            BiwiReadError::DirectoryExpected(None) => {
                write!(f, "Directory not found. Dirname: (cannot interpret dirame)")
            }
            BiwiReadError::IoError(ref io) => io.fmt(f),
            BiwiReadError::ImageError(ref err) => err.fmt(f),
            BiwiReadError::ParseFloatError(ref err) => err.fmt(f),
            BiwiReadError::InvalidDirStructure(Some(ref x)) => {
                write!(f, "Unexpected Dir-Structure for directory {}", x)
            }
            BiwiReadError::InvalidDirStructure(None) => {
                write!(f,
                       "Unexpected Dir-Structure for directory (not interpretable)")
            }
            BiwiReadError::Unspecific(ref x) => {
                write!(f, "Unspecific: {}", x.as_ref().map(|x| &**x).unwrap_or("-"))
            }

        }
    }
}

impl Error for BiwiReadError {
    fn description(&self) -> &str {
        match *self {
            BiwiReadError::DirectoryExpected(_) => "Directory not found", //TODO: Return dir-path
            BiwiReadError::InvalidDirStructure(_) => "Unexpected Dir-Structure", //TODO: Return dir-path
            BiwiReadError::Unspecific(_) => "Unspecific error", //TODO: Return dir-path
            BiwiReadError::IoError(ref err) => err.description(),
            BiwiReadError::ParseFloatError(ref err) => err.description(),
            BiwiReadError::ImageError(ref err) => err.description(),
        }
    }
    fn cause(&self) -> Option<&Error> {
        match *self {
            BiwiReadError::IoError(ref err) => Some(err),
            BiwiReadError::ImageError(ref err) => Some(err),
            BiwiReadError::ParseFloatError(ref err) => Some(err),
            BiwiReadError::DirectoryExpected(_) |
            BiwiReadError::Unspecific(_) |
            BiwiReadError::InvalidDirStructure(_) => None,
        }
    }
}

// Define Reader

/// A reader for the BIWI Database.
pub struct BiwiReader {
    mask_dir: String,
    depth_dir: String,
    truth_dir: String,
    pers_nr: Cell<Option<usize>>,
}


impl BiwiReader {
    /// Creates a new reader
    /// # Arguments
    /// * `mask_dir` - directory of the mask files ("Masks used to select positive patches" item on the website(
    /// * `depth_dir` - directoy of the depth files ("Data" item on the website)
    /// * `truth_dir` - directoy of the truth files ("Binary ground truth files" item on the website)
    pub fn new(mask_dir: String, depth_dir: String, truth_dir: String) -> BiwiReader {
        BiwiReader {
            mask_dir: mask_dir,
            depth_dir: depth_dir,
            truth_dir: truth_dir,
            pers_nr: Cell::new(None),
        }
    }

    /// Gets subdirectory in the given path for the person with the given number.
    /// Please note that the database starts counting at one.
    fn person_subdir<P: AsRef<Path>>(path: P, nr: usize) -> PathBuf {
        let path = path.as_ref();
        let subdir = format!("{:02}", nr);
        path.join(subdir)
    }

    /// Returns a list of all files that have the given extension in the given path
    fn list_subdir<P: AsRef<Path>>(path: P, ext: &str) -> IOResult<Vec<DirEntry>> {
        let path = path.as_ref();
        let mut dirs: Vec<_> = read_dir(path)?
            .filter_map(|x| x.ok())
            .filter(|x| match x.path().extension() {
                Some(x) => x == ext,
                _ => false,
            })
            .collect();
        dirs.sort_by_key(|x| x.file_name());
        Ok(dirs)
    }

    /// Checks if the directory structure of the path is  correct
    /// and returns the number of persons available
    fn check_common<P: AsRef<Path>>(path: P) -> BResult<usize> {
        let path = path.as_ref();
        if !path.is_dir() {
            return Err(BiwiReadError::DirectoryExpected(path.to_str().map(|x| x.to_string())));
        }
        let dirs: Vec<_> = read_dir(path)?
            .filter_map(|x| x.ok())
            .filter(|x| x.path().is_dir())
            .map(|x| x.file_name())
            .collect();
        let len = dirs.len();
        for i in 1..(len + 1) {
            let needed = format!("{:02}", i);
            let needed = OsString::from(needed);
            if !dirs.iter().any(|x| *x == needed) {
                return Err(BiwiReadError::InvalidDirStructure(path.to_str()
                    .map(|x| x.to_string())));
            }
        }
        Ok(len)
    }
}

impl DepthReader for BiwiReader {
    type Err = BiwiReadError;
    type Iter = Box<Iterator<Item = BResult<DepthTrue>>>;

    // person number starts with 1
    fn person(&self, nr: usize) -> BResult<Self::Iter> {
        lazy_static! {
            static ref BIWI_NAME: Regex = Regex::new(r"(frame_\d+)_.+").unwrap();
        }
        let depth_subdir = BiwiReader::person_subdir(&*self.depth_dir, nr);
        let depth = BiwiReader::list_subdir(&depth_subdir, "bin")?;
        let calib = depth_subdir.join("depth.cal");
        let calib_mat = read_cal(BufReader::new(File::open(calib)?))?;

        let mask_subdir = BiwiReader::person_subdir(&*self.mask_dir, nr);
        let truth_subdir = BiwiReader::person_subdir(&*self.truth_dir, nr);
        let res = depth.into_iter().filter_map(move |depth| {
            let depth_file_name = depth.file_name().into_string().ok();
            let name_prefix = match depth_file_name.as_ref().and_then(|x| BIWI_NAME.captures(&*x)).and_then(|x| x.get(1)){
                Some(x) => x.as_str(), None => return Some(Err(BiwiReadError::Unspecific(Some("Invalid filename found".to_string()))))
                // Some(x) => x, None => return None
            };
            
            let mask:PathBuf = mask_subdir.join(format!("{}_depth_mask.png",name_prefix));
            let truth = truth_subdir.join(format!("{}_pose.bin",name_prefix));

            // Skip if the corresponding file does not exist.
            if !mask.exists()  || !truth.exists(){
                // return Err(BiwiReadError::Unspecific(Some(format!("Needed Truth or Depth does not exists for depth {}",depth.file_name().into_string().unwrap_or("Unparsable".to_string())))));
                return None

            }
            // t!(..) returns in the case of an error
            macro_rules! t {
                ($x: expr) => (
                    match $x { Ok(x) => x, Err(x) => return Some(Err(BiwiReadError::from(x)))}
                )
            }
            let truth_file = t!(File::open(truth));
            let gt = t!(read_gt(truth_file,&calib_mat));
            let depth_file = t!(File::open(depth.path()));
            let d = t!(read_depth(depth_file));
            let m = t!(image::open(mask));
            let mut m = m.to_luma();
            for p in m.pixels_mut(){
                if p.data[0] > 0 { p.data[0] = 1 }
            }
            Some(Ok(DepthTrue{
                trans: gt,
                depth: d,
                mask: m,
                intrinsic: calib_mat.clone(),
            }))
            });

        Ok(Box::new(res))
    }

    fn is_valid(&self) -> BResult<bool> {
        let n = BiwiReader::check_common(&*self.mask_dir)?;
        let n2 = BiwiReader::check_common(&*self.depth_dir)?;
        let n3 = BiwiReader::check_common(&*self.truth_dir)?;
        let allright = n == n2 && n2 == n3;
        if allright {
            self.pers_nr.set(Some(n));
        }
        Ok(allright)
    }

    fn person_count(&self) -> BResult<usize> {
        // Lazy looking
        if let Some(n) = self.pers_nr.get() {
            return Ok(n);
        }

        if !self.is_valid()? {
            return Ok(0);
        }

        Ok(self.pers_nr.get().unwrap_or(0))
    }
}
