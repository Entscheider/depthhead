/// Show the hough forest image (using 2d hough forest variant) of a depth image

extern crate depthhead;
extern crate clap;
#[macro_use]
extern crate log;
extern crate env_logger;
extern crate depthhead_example_utils;
extern crate image;
#[macro_use]
extern crate lazy_static;
extern crate regex;

extern crate serde_json;

#[macro_use]
extern crate error_chain;


use std::fs::File;
use std::io::Read;
use std::sync::Arc;

use clap::{Arg, App};

use depthhead::db_reader::biwi::read_depth;
use depthhead::hough::prediction::HoughPrediction;

use depthhead_example_utils::imgwin::{ImgWindow, FixWaitTimer};
use std::time::Duration;
use std::path::Path;
use std::cell::Cell;

use depthhead::types::IntrinsicMatrix;

error_chain!{
    foreign_links {
        Serde(serde_json::Error);
        IO(std::io::Error);
    }
}

macro_rules! eprintln {
    ($e: expr, $($args: expr)*) => {
        use std::io::{stderr, Write};
        let mut out = stderr();
        let mut out = out.lock();
        out.write_fmt(format_args!($e, $($args,)*)).unwrap();
    };
}

pub fn main(){
    use std::process::exit;
    if let Err(e) = main_(){
        eprintln!("Error: {}",e);
        exit(-1);
    }
}


fn load_related_rgb<P: AsRef<Path>>(path: P) -> Option<image::DynamicImage> {
    use regex::Regex;
    lazy_static! {
        static ref BIWI_NAME: Regex = Regex::new(r"(frame_\d+)_.+").unwrap();
    }
    let path = path.as_ref();
    let parent = match path.parent() {
        Some(x) => x,
        None => return None,
    };
    let file = match path.file_name()
        .and_then(|x| x.to_str())
        .as_ref()
        .and_then(|x| BIWI_NAME.captures(&*x))
        .and_then(|x| x.get(1))
        .map(|x| parent.join(format!("{}_rgb.png", x.as_str()))) {
        Some(x) => x,
        None => return None,
    };
    image::open(file).ok()
}

pub fn main_() -> Result<()> {
    env_logger::init();
    let args = App::new("Hough-Generator")
        .arg(Arg::with_name("learnedforest")
            .short("t")
            .long("trained")
            .required(true)
            .takes_value(true)
            .help("Filename of learned forest"))
        .arg(Arg::with_name("depth")
            .help("Path to depth image")
            .required(true)
            .index(1))
        .arg(Arg::with_name("output")
            .short("o")
            .long("output")
            .takes_value(true)
            .help("Outputpath to save hough image"))
        .get_matches();
    let forest_path = args.value_of("learnedforest").ok_or(
                            "Invalid parameter for learned forest")?;
    let depth_path = args.value_of("depth").ok_or(
                           "Invalid parameter for depth image path")?;

    info!("Loading forest ...");
    let mut json = String::new();
    let mut jsonfile = File::open(forest_path)?;
    jsonfile.read_to_string(&mut json)?;
    let forest: HoughPrediction = serde_json::from_str(&json)?;

    info!("Loading image ...");
    let img_file = File::open(depth_path)?;
    let depth = Arc::new(read_depth(img_file)?);
    let rgb = load_related_rgb(&*depth_path);
    info!("Predicting ...");
    let intrin = IntrinsicMatrix::default_kinect_intrinsic();
    let hough = forest.build_hough_image(depth, &intrin);
    let max = hough.pixels().map(|x| x.data[0]).max().unwrap();

    let mut hough_win = ImgWindow::new("Hough".to_string());
    let mut rgb_win = if rgb.is_some() {
        Some(ImgWindow::new("RGB"))
    } else {
        None
    };
    let best_pos = Cell::new([0u32; 2]);
    let hough = match max {
        0 => {
            warn!("Got Zero-Hough");
            image::RgbaImage::new(hough.width(), hough.height())
        }
        _ => {
            image::RgbaImage::from_fn(hough.width(), hough.height(), |x, y| {
                let val = hough[(x, y)].data[0];
                let val = 255 * val / max;
                let val = val as u8;
                if val == 255 {
                    best_pos.set([x, y]);
                }
                // let val = match val { x if x > 255 => 255, x => x as u8 };
                match val {
                    val if val > 250 => image::Rgba([255u8, 0, 255, 255]),
                    val => image::Rgba([val, val, val, 255]),
                }
            })
        }
    };

    if let Some(output) = args.value_of("output") {
        info!("Saved file to {}", output);
        hough.save(output)?;
    }

    hough_win.set_img(hough);
    hough_win.redraw();
    if let Some(rgb) = rgb {
        // Paint the predicted middle points into the rgb image
        let mut img = rgb.to_rgba();
        let x = best_pos.get()[0] as i32;
        let y = best_pos.get()[1] as i32;
        for j in 0..10 {
            for i in 0..10 {
                let j = (j as i32) - 5;
                let i = (i as i32) - 5;
                if x + i < 0 || x + i > img.width() as i32 {
                    continue;
                }
                if y + j < 0 || y + j > img.height() as i32 {
                    continue;
                }
                let x = (x + i) as u32;
                let y = (y + j) as u32;
                img[(x, y)] = image::Rgba([255, 0, 0, 255]);
            }
        }
        rgb_win.as_mut().unwrap().set_img(img);
    }


    loop {
        let _ = FixWaitTimer::new(Duration::from_millis(1000 / 25));
        if hough_win.check_for_close() {
            break;
        }
        if let Some(true) = rgb_win.as_ref().map(|x| x.check_for_close()) {
            break;
        }
        hough_win.redraw();
        if let Some(ref rgb_win) = rgb_win {
            rgb_win.redraw();
        }
    }
    Ok(())
}
