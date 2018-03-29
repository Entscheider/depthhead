/// Train a hough forest

extern crate depthhead;
extern crate clap;
#[macro_use]
extern crate log;
extern crate env_logger;
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate error_chain;

use depthhead::hough::prediction;
use depthhead::db_reader as db;
use depthhead::db_reader::reader::DepthReader;
use clap::{Arg, App};
use std::u8;
use std::fs::File;
use std::io::Write;
use std::str::FromStr;
use std::process::exit;
use std::io::Result as IOResult;

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
        Biwi(db::biwi::BiwiReadError);
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


macro_rules! try_or_exit {
    ($x: expr) => (
        match $x{
            Ok(x) => x,
            Err(r) => {warn!("Error: {}", r); exit(-1)}
        }
        )
}

#[inline]
fn tojson<T>(obj: &T, path: &str) -> IOResult<()>
    where T: serde::Serialize
{
    let json = serde_json::to_string(obj).unwrap();
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}


macro_rules! MAXDEPTH_DEFAULT {() =>   (15u8) }
macro_rules! TREE_SUBSET_DEFAULT { () => (5200u32)}
macro_rules! NODE_FEATURES_DEFAULT { () =>  (2000u32)}
macro_rules! STEEPNESS_DEFAULT { () => (5.0f64) }
macro_rules! STOPSIZE_DEFAULT {() => (20u32)}
macro_rules! TREE_COUNT_DEFAULT { () => (20u8) }

pub fn main_() -> Result<()> {
    env_logger::init();
    let args = App::new("Houghtree learner")
        .arg(Arg::with_name("BIWI_Data_Dir")
            .short("d")
            .long("datadir")
            .required(true)
            .takes_value(true)
            .help("Path to BIWI Trainingsdata"))
        .arg(Arg::with_name("personcount")
            .short("n")
            .long("personcount")
            .takes_value(true)
            .help("Number of person for training"))
        .arg(Arg::with_name("out_filename")
            .short("o")
            .long("out")
            .takes_value(true)
            .required(true)
            .help("Filename for trained tree"))
        .arg(Arg::with_name("maxdepth")
            .long("maxdepth")
            .takes_value(true)
        .help(concat!("Max depth a tree may be grown - Default ",MAXDEPTH_DEFAULT!())))
        .arg(Arg::with_name("subset")
            .long("subset")
            .takes_value(true)
        .help(concat!("Size of subset of the trainset used to train one tree in forest - Default ",TREE_SUBSET_DEFAULT!())))
        .arg(Arg::with_name("feature")
            .long("feature")
            .takes_value(true)
            .help(concat!("Number of feature to generate for each leave. Best of this feature will be \
        taken for this leave - Default ", NODE_FEATURES_DEFAULT!())))
        .arg(Arg::with_name("steepness")
            .long("steepness")
            .takes_value(true)
        .help(concat!("Steepness of the weighting function - Default ", STEEPNESS_DEFAULT!())))
        .arg(Arg::with_name("stopsize")
            .long("stopsize")
            .takes_value(true)
        .help(concat!("Max size of splitted subset to stop growing a tree - Default ",STOPSIZE_DEFAULT!())))
        .arg(Arg::with_name("trees")
            .long("trees")
            .takes_value(true)
        .help(concat!("Number of trees for the forest - Default ",TREE_COUNT_DEFAULT!())))
        .get_matches();

    // Load trainings data
    let pers_for_training =
        args.value_of("personcount").map(|x| try_or_exit!(u8::from_str(&*x))).unwrap_or(12u8);
    let rootpath = args.value_of("BIWI_Data_Dir").ok_or(
                         "BIWI Datadir parameter is invalid")?;
    let filename = args.value_of("out_filename").ok_or("No valid output filename")?;
    info!("Reading truth");
    let reader = db::biwi::BiwiReader::new(format!("{}/head_pose_masks", rootpath),
                                           format!("{}/hpdb", rootpath),
                                           format!("{}/db_annotations", rootpath));

    if !reader.is_valid()? {
        bail!("Error: Invalid Data");
    }

    let person_count = reader.person_count()?;
    if person_count == 0 {
        bail!("Found 0 persons");
    }
    let mut it = reader.person(1)?;
    for j in 1..(pers_for_training as usize) {
        it = Box::new(it.chain(reader.person(j + 1)?));
    }

    // let gaussian_sigma = 12.0;
    let gaussian_sigma = 8.0;

    let maxdepth = args.value_of("maxdepth").map(|x| try_or_exit!(u8::from_str(x))).unwrap_or(MAXDEPTH_DEFAULT!());
    let tree_subset = args.value_of("subset").map(|x| try_or_exit!(u32::from_str(x))).unwrap_or(TREE_SUBSET_DEFAULT!());
    let node_features = args.value_of("feature").map(|x| try_or_exit!(u32::from_str(x))).unwrap_or(NODE_FEATURES_DEFAULT!());
    let steepness = args.value_of("steepness").map(|x| try_or_exit!(f64::from_str(x))).unwrap_or(STEEPNESS_DEFAULT!());
    let subset_to_stop = args.value_of("stopsize").map(|x| try_or_exit!(u32::from_str(x))).unwrap_or(STOPSIZE_DEFAULT!());
    let trees_count = args.value_of("trees").map(|x| try_or_exit!(u8::from_str(x))).unwrap_or(TREE_COUNT_DEFAULT!());
    info!("Starting Learning");
    let learner = prediction::HoughLearning::new(// Trained on 3000 Images., 18 Subjects
                                                       10, // stepwidth (10 Paper for realtime)
                                                       80, // subimg-width (80 Paper)
                                                       80, // subimg-height (80 Paper )
                                                       maxdepth as usize, // max-depth (15 Paper) - default 15
                                                       trees_count as usize, // number of trees (10 oder 20 Paper) - default 20
                                                       tree_subset as usize, // subset size per tree  - default 5200
                                                       0.3, // subrect feature scale
                                                       node_features as usize, /* feature number per node (2K-20K Paper) - default 2000 */
                                                       subset_to_stop as usize, // subset size to stop (Paper) - default 20
                                                       steepness /* steepness der weighting function (Paper) - default 5.0 */)
                                        .ok_or("Bad parameter for learning Houghforest")?;
    let learn_param_filename = format!("{}_param.json", filename);
    tojson(&learner, &*learn_param_filename)?;
    let it = it.map(|x| {
        match x {
            Ok(x) => x,
            Err(r) => {
                warn!("Error: {}", r);
                exit(20)
            }
        }
    });
    let tree = learner.learn(gaussian_sigma, it).ok_or("Cannot train tree")?;
    info!("Learned sucessfull");
    tojson(&tree, &*filename)?;
    Ok(())
}
