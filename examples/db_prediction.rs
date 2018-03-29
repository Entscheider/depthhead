/// Predict the head pose using image from the biwi database.
/// Use your right key to navigate between the frames.
/// Use you up and down key to increase/decrease the sigma value used
/// for mean shifting.

extern crate depthhead;
extern crate depthhead_example_utils;
extern crate image;
extern crate clap;
#[macro_use]
extern crate error_chain;

use depthhead_example_utils::{headwin, imgwin, VirtualKeyCode};


extern crate serde_json;
#[macro_use]
extern crate log;
extern crate env_logger;


use depthhead::hough::prediction::HoughPrediction;
use depthhead::types::{DepthImage, IntrinsicMatrix};
use depthhead::db_reader::biwi::{BiwiReader, BResult, BiwiReadError};
use depthhead::db_reader::reader::DepthTrue;

use clap::{Arg, App};
use std::time::Duration;

macro_rules! eprintln {
    ($e: expr, $($args: expr)*) => {
        use std::io::{stderr, Write};
        let mut out = stderr();
        let mut out = out.lock();
        out.write_fmt(format_args!($e, $($args,)*)).unwrap();
    };
}

error_chain!{
    foreign_links {
        Biwi(BiwiReadError);
        IO(std::io::Error);
        Serde(serde_json::Error);
    }
}

pub fn main(){
    use std::process::exit;
    if let Err(e) = main_(){
        eprintln!("Error: {}",e);
        exit(-1);
    }
}

fn main_() -> Result<()> {
    use std::str::FromStr;
    use depthhead::db_reader::reader::DepthReader;
    env_logger::init();

    let args = App::new("DB Prediction")
        .arg(Arg::with_name("trained")
            .short("t")
            .long("trained")
            .required(true)
            .takes_value(true)
            .help("Filename of learned tree"))
        .arg(Arg::with_name("db")
            .short("d")
            .long("data")
            .required(true)
            .takes_value(true)
            .help("Path to the database"))
        .arg(Arg::with_name("person_nr")
            .short("n")
            .long("person_nr")
            .required(true)
            .takes_value(true)
            .help("Number of person in database to show"))
        .get_matches();


    let rootpath = args.value_of("db").ok_or("Invalid parameter for data path")?;
    let learned_tree = args.value_of("trained").ok_or(
                             "Invalid parameter for learned forest")?;
    let nr_of_person = args.value_of("person_nr").and_then(|x| usize::from_str(*&x).ok()).ok_or(
                             "Invalid argument for the number of person")?;

    let reader = BiwiReader::new(format!("{}/head_pose_masks", rootpath),
                                 format!("{}/hpdb", rootpath),
                                 format!("{}/db_annotations", rootpath));
    if !reader.is_valid()? {
        bail!("Invalid Data");
    }

    let personcount = reader.person_count()?;
    info!("Found {} persons", personcount);

    let tree = load_tree(learned_tree)?;


    let mut visualizer = Visualizer::new(reader.person(nr_of_person)?, tree);

    loop {
        let _ = imgwin::FixWaitTimer::new(Duration::from_millis(1000 / 25));
        if visualizer.event()? {
            break;
        }
        visualizer.redraw();
    }
    Ok(())
}

struct Visualizer<I: Iterator<Item = BResult<DepthTrue>>> {
    win: headwin::HeadWin,
    reader: I,
    forest: HoughPrediction,
    current: Option<DepthTrue>,
}

impl<I: Iterator<Item = BResult<DepthTrue>>> Visualizer<I> {
    fn new(reader: I, forest: HoughPrediction) -> Visualizer<I> {
        Visualizer {
            win: headwin::HeadWin::new("Prediction", (IntrinsicMatrix::default_kinect_intrinsic().0).0),
            reader: reader,
            forest: forest,
            current: None,
        }
    }

    fn update(&mut self) -> BResult<()> {
        use std::sync::Arc;
        let depth = match self.current {
            Some(ref x) => x,
            None => return Ok(()),
        };
        let ref intrinsic = depth.intrinsic;
        self.win.set_intrinsic((intrinsic.0).0);
        let depthp = Arc::new(depth.depth.clone());
        self.win.update_image(depth_to_rgba(&*depthp));
        let prediction = self.forest.predict_parameter(depthp.clone(), &intrinsic, None, None);
        let mid = prediction.mid_point;
        let rot = prediction.rotation;
        let rot = [rot[0] as f32, rot[1] as f32, rot[2] as f32];
        let trans = depth.trans;
        let truth_mid = trans.pos3d;
        let truth_rot = trans.rot;
        info!("Guess: Midpoint {:?}, Rotation {:?}; Truth: Midpoint {:?}, Rotation {:?}",
              mid,
              radian_to_degree(rot),
              truth_mid,
              truth_rot);
        self.win.update_transformation(rot.into(), mid.into(), 2.0);
        //self.win.update_transformation(degree_to_radian(truth_rot).into(), truth_mid.into(), 2.0);
        let prediction = self.forest.predict_parameter_from2dhough(depthp, &intrinsic);
        let mid = prediction.mid_point;
        info!("2D-Hough-Guess: Midpoint {:?} - Truth {:?}",
              mid,
              trans.pos3d);
        Ok(())
    }

    fn redraw(&mut self) {
        self.win.redraw();
    }

    fn event(&mut self) -> BResult<bool> {
        let mut handler = EventHandler::new();
        self.win.check_for_event(&mut handler);
        if handler.next {
            self.current = match self.reader.next() {
                Some(Ok(x)) => Some(x), 
                Some(Err(x)) => return Err(x),
                _ => return Ok(false),
            }
        }
        if handler.sigma_add != 0.0 {
            let sigma = self.forest.sigma();
            self.forest.update_sigma(sigma + handler.sigma_add);
            info!("Updated Gaussian-Sigma to {}", self.forest.sigma());
        }
        if handler.update {
            self.update()?;
        }
        Ok(handler.close)
    }
}

struct EventHandler {
    close: bool,
    update: bool,
    next: bool,
    sigma_add: f32,
}

impl EventHandler {
    fn new() -> EventHandler {
        EventHandler {
            close: false,
            update: false,
            next: false,
            sigma_add: 0.0,
        }
    }
}

impl imgwin::EventHandler for EventHandler {
    fn close_event(&mut self) {
        self.close = true
    }

    fn key_event(&mut self, inp: Option<VirtualKeyCode>) {
        if let Some(code) = inp {
            match code{
               VirtualKeyCode::Right => {
                   self.update = true;
                   self.next = true
               },
               //VirtualKeyCode::Left => {
                   //self.n -= if self.n == 0 {0} else {1};
               //},
               VirtualKeyCode::Up => {
                   self.sigma_add += 0.1;
               },
               VirtualKeyCode::Down => {
                   self.sigma_add -= 0.1;
               }
               VirtualKeyCode::Escape | VirtualKeyCode::Q => {
                   self.close = true;
                   self.update = false;
               }
               _ => {
                   self.update = false;
               }
           }
        }
    }
}

fn depth_to_rgba(depth_img: &DepthImage) -> image::RgbaImage {
    image::RgbaImage::from_fn(depth_img.width(), depth_img.height(), |x, y| {
        let val = depth_img[(x, y)];
        let val = val.data[0];
        let val8 = if val > 600 { val - 600 } else { 0 };
        let val8 = val8 / 2;
        let val8 = if val8 > 255 { 255 } else { val8 };
        let val8 = val8 as u8;
        image::Rgba([val8, val8, val8, 255])
    })
}

fn load_tree(path: &str) -> Result<HoughPrediction> {
    use std::fs::File;
    use std::io::Read;
    let mut json = String::new();
    let mut jsonfile = File::open(path)?;
    jsonfile.read_to_string(&mut json)?;
    Ok(serde_json::from_str(&json)?)
}

fn radian_to_degree(rad: [f32; 3]) -> [f32; 3] {
    macro_rules! r2d {
        ($x: expr) => ($x * 180.0 / 3.14159)
    }
    [r2d!(rad[0]), r2d!(rad[1]), r2d!(rad[2])]
}

fn degree_to_radian(rad: [f32; 3]) -> [f32; 3] {
    macro_rules! r2d {
        ($x: expr) => ($x * 3.14159 / 180.0)
    }
    [r2d!(rad[0]), r2d!(rad[1]), r2d!(rad[2])]
}