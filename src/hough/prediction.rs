/// Functions and structs to use a hough forest to predict the head
/// position and rotation within a depth image and util functions/structs to
/// train them using the data of a database.

use super::houghforest;
use super::houghforest::HoughForestParameter;
use stamm::randforest::RandomForest;
use image::{ImageBuffer, Luma};
use db_reader::reader::DepthTrue;
use imageproc::filter::gaussian_blur_f32;
use std::iter::Iterator;
use types::*;
use meancov_estimation::{Vec2, Vec3, estimate_mean_cov, MatrixFuncSimple};
use rand;
use std::sync::Arc;
use meanshift::*;
use std::cell::{RefCell, Ref};

macro_rules! max {
    ($x: expr, $y: expr) => (if $x > $y {$x} else {$y})
}

macro_rules! min {
    ($x: expr, $y: expr) => (if $x < $y {$x} else {$y})
}

/*
macro_rules! abs {
    ($x: expr) => (if $x < 0 {-$x} else {$x})
}
*/

pub type HoughForest = RandomForest<houghforest::LeafParam,
                                    houghforest::HoughTreeFunctions<InMutSubImage>>;


/// Helps to learn a hough forest
#[derive(Serialize)]
pub struct HoughLearning {
    /// Stepwidth for the sliding window used
    /// to extract the subimage for the hough forest
    pub stepwidth: u32,
    /// paramters for train a hough forest.
    pub learn_params: HoughForestParameter<InMutSubImage>,
}


impl HoughLearning {

    /// Sets the width of a subimage used as input for this
    /// hough forest
    pub fn subimg_width(mut self, width: u32) -> Self {
        self.learn_params.tree_param.input_size =
            self.learn_params.tree_param.input_size.with_width(width);
        self
    }

    /// Sets the height of a subimage used as input for this
    /// hough forest
    pub fn subimg_height(mut self, width: u32) -> Self {
        self.learn_params.tree_param.input_size =
            self.learn_params.tree_param.input_size.with_height(width);
        self
    }


    /// Returns the width of the size of a subimage
    /// used for this hough forest.
    fn get_subimg_width(&self) -> u32 {
        self.learn_params.tree_param.input_size.width()
    }

    /// Returns the height of the size of a subimage
    /// used for this hough forest.
    fn get_subimg_height(&self) -> u32 {
        self.learn_params.tree_param.input_size.height()
    }


    /// sets the minimal size of a subset to stop further training
    /// a subtree
    pub fn min_subset_size_to_stop(mut self, val: f64) -> Self {
        self.learn_params.tree_param.min_subrect_factor = val;
        self.learn_params.tree_param.max_subrect_factor = val;
        self
    }

    /// Creates a new HoughLearning.
    /// Returns None if some parameters are invalid.
    ///
    /// # Arguments
    /// * `stepwidth` - stepwidth use for sliding window
    /// * `subimg_width` - width of a subimage used as input for the hough forest
    /// * `subimg_height` - height of a subimage used as input for the hough forest
    /// * `max_depth` - maximum depth a tree may have
    /// * `num_of_trees` - number of trees this forest consists
    /// * `subset_size_per_tree` - size of the subset of the training set for train a tree
    /// * `feature_number_per_node` - number of feature to generate for training a node
    /// * `min_subset_size_to_stop` - stop training a subtree if the size of set to train it is lesser than this value
    /// * `steepness_weighting` - value used in the weighting function (see Paper [^1])
    ///
    /// [^1]: Multiview Facial Landmark Localization in RGB-D Images via Hierarchical Regression With Binary Patterns (Zhang et al.)
    pub fn new(stepwidth: u32,
               subimg_width: u32,
               subimg_height: u32,
               max_depth: usize,
               num_of_trees: usize,
               subset_size_per_tree: usize,
               subrect_feature_scale: f64,
               feature_number_per_node: usize,
               min_subset_size_to_stop: usize,
               steepness_weighting: f64)
               -> Option<HoughLearning> {
        let mut tree_func = match houghforest::HoughTreeFunctions::new(Rect::new(0,
                                                                                 0,
                                                                                 subimg_width,
                                                                                 subimg_height),
                                                                       subrect_feature_scale,
                                                                       subrect_feature_scale,
                                                                       feature_number_per_node,
                                                                       steepness_weighting) {
            Some(x) => x,
            None => return None,
        };
        tree_func.max_depth = max_depth;
        tree_func.min_subset_size = min_subset_size_to_stop;

        Some(HoughLearning {
            stepwidth: stepwidth,
            learn_params: houghforest::HoughForestParameter {
                tree_param: tree_func,
                number_of_trees: num_of_trees,
                size_of_subset_per_training: subset_size_per_tree,
            },
        })
    }

    /// Train a hough forest from data of a database.
    /// Therefore a Iterator of type `I` is needed which iterates over
    /// the ground truth set.
    ///
    /// # Arguments
    /// * `gaussian_sigma` - Used for the prediction later. This value gives the sigma used for meanshifting.
    /// * `data` - Iterator which iterate over the ground truth data
    pub fn learn<'a, I: Iterator<Item = DepthTrue>>(&self,
                                                    gaussian_sigma: f32,
                                                    data: I)
                                                    -> Option<HoughPrediction> {
        use super::houghforest::HoughForestImage;
        let mut rng = rand::thread_rng();
        let mut train_data: Vec<(InMutSubImage, houghforest::Truth)> = vec![];
        let mut i = 0;
        let len = data.size_hint().1.unwrap_or(0);
        let subimage_width = self.get_subimg_width();
        let subimage_height = self.get_subimg_height();
        train_data.reserve_exact(len * 40);

        // For every ground truth image we extract randomly
        // 20 subimages and collect them across the entire
        // training-set to train the hough forest.
        for truth in data {
            info!("Loading img {} of approx. {}", i + 1, len);
            let mut img_positives = vec![];
            let mut img_negatives = vec![];
            i += 1;
            let depth = Arc::new(truth.depth);
            let mask = truth.mask;
            let intrinsic = truth.intrinsic;
            let mid = truth.trans.pos3d;
            let rot = truth.trans.rot;
            let rot = [rot[0] as f64, rot[1] as f64, rot[2] as f64];

            // Sliding window to extract every subimage with the ground truth values
            iterate_subimage(depth.clone(),
                             subimage_width,
                             subimage_height,
                             self.stepwidth,
                             self.stepwidth,
                             |subimg, x, y| {
                // Check if the current subimage is not just background
                if subimg.average_value_in_rect(Rect::new(0, 0, subimage_width, subimage_height)) >
                   0.0 {
                    let truth_flag = mask[(x, y)];
                    let z = depth[(x, y)].data[0] as f32;
                    let truth = match truth_flag.data[0] {
                        0 => houghforest::Truth::NoObject,
                        _ => {
                            let vec2 = Vec2::new([x as f32, y as f32]);
                            let vec3 = intrinsic.img_to_space_coord(vec2, z);
                            let offset = vec3 - mid.into();
                            houghforest::Truth::Object {
                                offset: offset,
                                rotation: Vec3::new(rot),
                            }
                        }
                    };
                    match truth {
                        x @ houghforest::Truth::NoObject => img_negatives.push((subimg, x)),
                        x @ houghforest::Truth::Object { offset: _, rotation: _ } => {
                            img_positives.push((subimg, x))
                        }
                    }
                }
            });

            rand_perm(&mut img_positives, &mut rng);
            rand_perm(&mut img_negatives, &mut rng);

            // Collect only 20 positive and 20 negative values for every picture.
            // Thanks to `to_cropped_subimage` we can also delete a big part of the image
            // form the memory.
            train_data.extend(
                img_negatives.into_iter().take(20).map(|(x, y)| (x.to_cropped_subimage(), y)));
            train_data.extend(
                img_positives.into_iter().take(20).map(|(x, y)| (x.to_cropped_subimage(), y)));
        }
        let mut train_ref: Vec<_> = train_data.iter().map(|&(ref a, ref b)| (a, b)).collect();
        rand_perm(&mut train_ref, &mut rng);
        info!("Start training with {} samples", train_ref.len());

        let forest = match self.learn_params.train_tree_parallel(&train_ref[..]) {
            Some(x) => x,
            None => return None,
        };
        Some(HoughPrediction {
            stepwidth: self.stepwidth,
            subimage_width: subimage_width,
            subimage_height: subimage_height,
            gaussian_sigma: gaussian_sigma,
            forest: forest,
            kernel3d: RefCell::new(None),
            meanshift_iterations: 20,
        })
    }
}

/// This struct is able to estimate the head position and rotation
/// from a depth image given by the kinect.
#[derive(Serialize, Deserialize)]
pub struct HoughPrediction {
    /// stepwidth use for sliding window
    pub stepwidth: u32,
    /// width of a subimage used for the hough forest
    subimage_width: u32,
    /// height of a subimage used for the hough forest
    subimage_height: u32,
    /// sigma value used for meanshifting
    gaussian_sigma: f32,
    /// hough forest used to vote for the head position and rotation of an subimage
    forest: HoughForest,
    /// used for meanshifting (for the rotatopn)
    #[serde(skip_serializing, skip_deserializing)]
    kernel3d: RefCell<Option<FullArray3D<f32>>>,
    /// number of iteration used for meanshifting
    pub meanshift_iterations: u32,
}

/// Result of a prediction
pub struct PredictionResult {
    /// 3d point prediction of the center of the head
    pub mid_point: [f32; 3],
    /// rotation guess of the head
    pub rotation: [f64; 3],
    /// Bounding box of the head. Note that the current implementation
    /// does not predict this.
    pub bounding_box: Rect,
}

/// value to normalize distance relative to the x and y values (1 = no normalization)
const ZSCALEFACTOR: u32 = 1;


/// Gives a size of an rough search used for get a good initialization
/// for meanshifting (here for head position)
const GUESS_GRID_PARTS: usize = 20;

/// Gives a size of an rough search used for get a good initialization
/// for meanshifting (here for head rotation)
const ROT_GRID_PARTS: usize = 120;

/// Largest value  the trace of an covariance matrix may have
/// to let it vote (here for head rotation)
const MAX_VARIANCE_ROT: f64 = 400.0;
/// Largest value  the trace of an covariance matrix may have
/// to let it vote (here for head position)
const MAX_VARIANCE_OFFSET: f32 = 5200.0;

// // Serialize a SparseArray3D (built in houghcube) to a ply file
// fn serialize(cube: &SparseArray3D<u32>, filename: &str){
//     use std::fs::File;
//     use std::io::Write;
//     let mut f = File::create(filename).unwrap();
//     f.write_fmt(format_args!("ply\nformat ascii 1.0\nelement vertex {}\n", cube.iter().len())).unwrap();
//     f.write_fmt(format_args!("property float x\nproperty float y\nproperty float z\n")).unwrap();
//     f.write_fmt(format_args!("property uchar red\nproperty uchar green\nproperty uchar blue\n")).unwrap();
//     f.write_fmt(format_args!("end_header\n")).unwrap();
//     let max = cube.iter().map(|(_,y)| y).max().unwrap();
//     for (&(x,y,z), val) in cube.iter(){
//         let color = val * 255 / max;
//         f.write_fmt(format_args!("{} {} {} {} {} {}\n", x-180, y-180, z-180, color, color, color )).unwrap();
//     }
// }

impl HoughPrediction {

    /// Computes the kernel for the sigma-value (given in self)
    /// used for mean shifting.
    /// Note for simplicity the return value is a `Ref<Option<...>>`
    /// but you can always unwrap the Option.
    fn get_or_build_kernel(&self) -> Ref<Option<FullArray3D<f32>>> {
        if self.kernel3d.borrow().is_some() {
            return self.kernel3d.borrow();
        }
        let kernel = FullArray3D::build_kernel(20, self.gaussian_sigma); //TODO: neighbour_size variable machen
        *self.kernel3d.borrow_mut() = Some(kernel);
        self.kernel3d.borrow()
    }

    /// Update sigma value used for mean shifting
    pub fn update_sigma(&mut self, val: f32) {
        if val == self.gaussian_sigma || val <= 0.0 {
            return;
        }
        self.gaussian_sigma = val;
        *self.kernel3d.borrow_mut() = None;
    }

    /// Returns the sigma value used for mean shifting
    pub fn sigma(&self) -> f32 {
        self.gaussian_sigma
    }

    /// A much simpler variant to predict the head parameter.
    /// Here we operate on a 2d image to predict the head center in 2d
    /// and map the result to the nearest 3d point using the depth image.
    ///
    /// Note that this variant does not predict the rotation nor the bounding box. This value
    /// will always be 0,0,0.
    ///
    /// # Arguments
    /// - `img` - the depth image used to predict the head pose
    /// - `intrinsic` - the intrinsic matrix of the camera used to generate the depth image
    pub fn predict_parameter_from2dhough(&self,
                                         img: Arc<DepthImage>,
                                         intrinsic: &IntrinsicMatrix)
                                         -> PredictionResult {

        let hough = self.build_hough_image(img.clone(), intrinsic);
        let w = hough.width();
        let h = hough.height();
        let best = (0..w * h)
            .max_by_key(|i| {
                let x = i % w;
                let y = i / w;
                hough[(x, y)].data[0]
            })
            .unwrap();
        let x = best % w;
        let y = best / w;
        let z = img[(x, y)].data[0]; // TODO: Mittels Hough-Forest Predicten
        let p = intrinsic.img_to_space_coord([x as f32, y as f32], z as f32);
        PredictionResult {
            mid_point: p.into(),
            rotation: [0.0, 0.0, 0.0], // TODO: Implement this
            bounding_box: Rect::new(0, 0, 0, 0),
        }
    }

    /// Predict the head pose using a single core.
    ///
    /// # Arguments
    /// * `img` - the depth image for which the prediction is being done
    /// * `intrinsic` - the intrinsic matrix of the camera used to get the depth image
    /// * `midp_guess` - an optional guess of the midpoint used to initialize the mean shift
    /// * `rot_guess` - an optional guess of the rotation used to initialize the mean shift
    pub fn predict_parameter(&self,
                             img: Arc<DepthImage>,
                             intrinsic: &IntrinsicMatrix,
                             midp_guess: Option<[f32; 3]>,
                             rot_guess: Option<[f64; 3]>)
                             -> PredictionResult {
        self.predict_parameter_generic(img,
                                       intrinsic,
                                       midp_guess,
                                       rot_guess,
                                       |forest, subimg| forest.forest_predictions(subimg))

    }

    /// Predict the head pose using a multiple cores.
    ///
    /// # Arguments
    /// * `img` - the depth image for which the prediction is being done
    /// * `intrinsic` - the intrinsic matrix of the camera used to get the depth image
    /// * `midp_guess` - an optional guess of the midpoint used to initialize the mean shift
    /// * `rot_guess` - an optional guess of the rotation used to initialize the mean shift
    pub fn predict_parameter_parallel(&self,
                                      img: Arc<DepthImage>,
                                      intrinsic: &IntrinsicMatrix,
                                      midp_guess: Option<[f32; 3]>,
                                      rot_guess: Option<[f64; 3]>)
                                      -> PredictionResult {
        self.predict_parameter_generic(img,
                                       intrinsic,
                                       midp_guess,
                                       rot_guess,
                                       |forest, subimg| forest.forest_predictions_parallel(subimg))

    }

    /// Predict the head pose using a `pred_func`. This function
    /// can be used specify if the prediction will be computed
    /// using multiple cores or just one.
    ///
    /// # Arguments
    /// * `img` - the depth image for which the prediction is being done
    /// * `intrinsic` - the intrinsic matrix of the camera used to get the depth image
    /// * `midp_guess` - an optional guess of the midpoint used to initialize the mean shift
    /// * `rot_guess` - an optional guess of the rotation used to initialize the mean shift
    /// * `pred_func` - Function which returns the prediction of every in the hough forest
    fn predict_parameter_generic<'a, F>(&'a self,
                                        img: Arc<DepthImage>,
                                        intrinsic: &IntrinsicMatrix,
                                        midp_guess: Option<[f32; 3]>,
                                        rot_guess: Option<[f64; 3]>,
                                        pred_func: F)
                                        -> PredictionResult
        where F: Fn(&'a HoughForest, &InMutSubImage) -> Vec<&'a houghforest::LeafParam>
    {
        // Use the forest to vote on parameters.
        // The rest of this function wants to extract the best parameters
        // using mean shift and this votes.
        let (houghmid, guessmid, houghrot, guessrot) =
            self.build_hough_cube_generic(img.clone(), intrinsic, pred_func);

        // if we have a better guess for the central head position lets use it.
        let guessmid = if let Some(guess) = midp_guess {
            (guess[0] as i32, guess[1] as i32, guess[2] as i32 / ZSCALEFACTOR as i32)
        } else {
            guessmid
        };

        // if we have a better guess for the head rotation lets use it.
        let guessrot = if let Some(guess) = rot_guess {
            // Convert guessrot from radian to degree.
            // We also translate it, so there is no problem using mean shift
            // because negative values are on the other end of the mean shift array.
            ((guess[0] * 180.0 / 3.14159 + 180.0),
             (guess[1] * 180.0 / 3.14159 + 180.0),
             (guess[2] * 180.0 / 3.14159 + 180.0))
        } else {
            guessrot
        };

        // Convert rotation guess into coordinate for the mean shift grid
        // ( we use a grid because there are so many values between 0.0-360.0
        //  and we do not want to try them all)
        let guessrot = ((guessrot.0 * ROT_GRID_PARTS as f64 / 360.0) as i32,
                        (guessrot.1 * ROT_GRID_PARTS as f64 / 360.0) as i32,
                        (guessrot.2 * ROT_GRID_PARTS as f64 / 360.0) as i32) ;

        // serialize(&houghrot);

        // Get the mean shift kernel.
        let kernel = self.get_or_build_kernel();
        let kernel = kernel.as_ref().unwrap();

        // using mean shift to get the head position
        let res_mid = houghmid.meanshift(guessmid, kernel, self.meanshift_iterations);

        // using mean shift to get the head rotation
        let (r1, r2, r3) = houghrot.meanshift(guessrot, kernel, self.meanshift_iterations);
        // info!("Meanshift: rotation - {}°,{}°,{}°", r1, r2, r3);

        // Convert the grid coordinate to rotation coordinate (in radian).
        // In particular we want to undo the traslation by 180°.
        let r1 = (r1 as f64 - ROT_GRID_PARTS as f64 / 2.0) / (ROT_GRID_PARTS / 2 )as f64  *
                 3.14159;
        let r2 = (r2 as f64 - ROT_GRID_PARTS as f64 / 2.0) / (ROT_GRID_PARTS / 2) as f64 *
                 3.14159;
        let r3 = (r3 as f64 - ROT_GRID_PARTS as f64 / 2.0) / (ROT_GRID_PARTS / 2) as f64  *
                 3.14159;

        PredictionResult {
            // Normalsierung von z Rückgängig machen
            mid_point: [res_mid.0 as f32,
                        res_mid.1 as f32,
                        (res_mid.2 * ZSCALEFACTOR as i32) as f32]
                .into(),
            rotation: [r1, r2, r3],
            bounding_box: Rect::new(0, 0, 0, 0),
        }
    }

    /// Returns two 3d arrays which contains the voting information for the head position and
    /// rotation. To get this information this function use a hough forest.
    ///
    /// The return tuple contains:
    ///
    /// 1. the voting information for the position
    /// 2. a guess for starting the mean shift for position
    /// 3. the voting information for the rotation
    /// 4. a guess for starting the mean shift for ration
    ///
    /// # Arguments:
    /// * `img` - the depth image for which we want to predict the head pose
    /// * `intrinsic` - the intrinsic matrix of the camera which produced the depth image
    /// * `pred_func` - a function which returns the result of every tree of the hough forest.
    fn build_hough_cube_generic<'a, F>
        (&'a self,
         img: Arc<DepthImage>,
         intrinsic: &IntrinsicMatrix,
         pred_func: F)
         -> (SparseArray3D<u32>, (i32, i32, i32), SparseArray3D<u32>, (f64, f64, f64))
        where F: Fn(&'a HoughForest, &InMutSubImage) -> Vec<&'a houghforest::LeafParam>
    {
        use hough::houghforest::HoughForestImage;
        let (h, w) = (img.height(), img.width());
        // We need a good initialize guess for using mean shift, otherwise this procedure
        // is not able to find the good value.
        // We want to initialize the search using a rough guess of some point within a good cluster.
        // Because of noise, we cannot simple find the maximum value to make a good guess.
        // So we have to blur the arrays into some bigger areas where we can a maximum value quickly
        // without being worried about the noise.
        // To do that we create a little 2d arrays for position (2d is sufficient for the position)
        // and a rough 3d array for the rotation.
        // For voting we vote into a finer array and into this rougher ones.
        // So we can use the latter to get a good guess for initialize mean shifting for the first one.
        let mut guess_pos_grid = [0u32; GUESS_GRID_PARTS * GUESS_GRID_PARTS];
        let mut guess_rot_grid = FullArray3D::from_value(GUESS_GRID_PARTS as u32,
                                                         GUESS_GRID_PARTS as u32,
                                                         GUESS_GRID_PARTS as u32,
                                                         0u32);

        let left_w = self.subimage_width / 2;
        let right_w = self.subimage_width - left_w;
        let left_h = self.subimage_height / 2;
        let right_h = self.subimage_height - left_h;

        // For voting the finer values.
        let mut mid = SparseArray3D::with_default_value(0u32);
        let mut rot = SparseArray3D::with_default_value(0u32);

        let mut y = left_h;
        // Iterating over the image (sliding windows)
        while y < h - right_h {
            let mut x = left_w;
            while x < w - right_w {

                // z position of the central point within the subimage
                let z = img[(x, y)].data[0];

                // Compute the 3d position using this z coordinate
                let p3 = intrinsic.img_to_space_coord([x as f32, y as f32], z as f32);

                // Compute the vote for every tree of the forst.
                let leafs = {

                    // Build the current subimage
                    let subimg = InMutSubImage::new(x - left_w,
                                                    y - left_h,
                                                    self.subimage_width,
                                                    self.subimage_height,
                                                    img.clone())
                        .unwrap();
                    // Only do something if this is not background
                    if subimg.average_value_in_rect(Rect::new(0,
                                                              0,
                                                              self.subimage_width,
                                                              self.subimage_height)) >
                       0.0 {
                        Some(pred_func(&self.forest, &subimg))
                    } else {
                        None
                    }
                };

                // Let's add these votes to the voting array if the probability is high enough.
                if let Some(leafs) = leafs {

                    // Compute the probability of being a head using the average probability over every tree.
                    let prob = leafs.iter().map(|x| x.prob).sum::<f64>() / leafs.len() as f64;
                    // If the probablity is high enough
                    if prob > 0.7 { // TODO: Is this the best value?
                        // Let's add the vote of every tree
                        for leaf in leafs.iter() {
                            // We may wish to use only trees whose proability guess is high enough
                            // if leaf.prob > 0.95 {
                            // but for now we don't do that
                            if leaf.prob > 0.0{

                                // We increase the vote value within the vote array using the probability
                                // so we have to convert the probability into some integer value.
                                let valtoadd = (1000.0 * leaf.prob) as usize / leaf.offsets.len();
                                let valtoadd = valtoadd as u32;

                                // # Rotation voting
                                // Estimate the trace of the covariance matrix.
                                // This has to be low enough to allow any influence.
                                if estimate_mean_cov(&leaf.rotations[..]).unwrap().1.trace() <= MAX_VARIANCE_ROT /*&& prob > 0.8*/{
                                    for rot_vote in leaf.rotations.iter() {
                                        // Rotation data is in degree (with negative values).
                                        // They must be converted into radian and moved, so
                                        // the negative values don't make any trouble anymore.
                                        let r1 =
                                            (rot_vote.0[0] * ROT_GRID_PARTS as f64 / 360.0) as i32 +
                                            ROT_GRID_PARTS as i32 / 2;
                                        let r2 =
                                            (rot_vote.0[1] * ROT_GRID_PARTS as f64 / 360.0) as i32 +
                                            ROT_GRID_PARTS as i32 / 2;
                                        let r3 =
                                            (rot_vote.0[2] * ROT_GRID_PARTS as f64 / 360.0) as i32 +
                                            ROT_GRID_PARTS as i32 / 2;

                                        // inBetweenMod(x,y) forces x to fulfilled 0<= x <= y
                                        macro_rules! inBetweenMod {
                                            ($val: expr, $max: expr) => (
                                            match $val {
                                                x if x >= $max => x - $max,
                                                x if x < 0 => $max + x ,
                                                x => x
                                            }
                                            )
                                        }
                                        let r1 = inBetweenMod!(r1, ROT_GRID_PARTS as i32) as u32;
                                        let r2 = inBetweenMod!(r2, ROT_GRID_PARTS as i32) as u32;
                                        let r3 = inBetweenMod!(r3, ROT_GRID_PARTS as i32) as u32;

                                        // Convert the prediction into coordinates for the rough voting
                                        let rough_r1 = r1 * GUESS_GRID_PARTS as u32 / ROT_GRID_PARTS as u32;
                                        let rough_r2 = r2 * GUESS_GRID_PARTS as u32 / ROT_GRID_PARTS as u32;
                                        let rough_r3 = r3 * GUESS_GRID_PARTS as u32 / ROT_GRID_PARTS as u32;

                                        // Add the voting into the arrays
                                        rot[(r1 as i32, r2 as i32, r3 as i32)] += valtoadd as u32;
                                        guess_rot_grid[(rough_r1, rough_r2, rough_r3)] += valtoadd as u32;
                                    }
                                }

                                // # Head position voting
                                // Estimate the trace of the covariance matrix.
                                // This has to be low enough to allow any influence.
                                if estimate_mean_cov(&leaf.offsets[..]).unwrap().1.trace() <= MAX_VARIANCE_OFFSET{
                                    for offs in leaf.offsets.iter() {
                                        // Aus aktuellen Punkt und Offset den Voting-Punkt berechnen
                                        // Offset is relative but we want an absolut value
                                        let np = p3 - *offs;

                                        // Does not make sense => ignore
                                        if np.0[2] < 0.0 {
                                            continue;
                                        }

                                        let x3d = np.0[0];
                                        let y3d = np.0[1];
                                        let z3d = np.0[2];

                                        // Convert the vote into rough voting information.
                                        // Therefore we map the coordinate into 2d to
                                        // add this coordinate into the rough voting array.
                                        let point2d = intrinsic.space_to_img_coord([x3d, y3d, z3d]).0;
                                        let x2d = min!(max!(point2d[0], 0.0), (w - 1) as f32);
                                        let y2d = min!(max!(point2d[1], 0.0), (h - 1) as f32);

                                        // Normalize z value
                                        let z3d = z3d / ZSCALEFACTOR as f32;
                                        mid[(x3d as i32, y3d as i32, z3d as i32)] += valtoadd as u32;


                                        // Add the vote
                                        let guess_idx_x = x2d as usize * GUESS_GRID_PARTS /
                                                        w as usize;
                                        let guess_idx_y = y2d as usize * GUESS_GRID_PARTS /
                                                        h as usize;
                                        guess_pos_grid[guess_idx_y * GUESS_GRID_PARTS + guess_idx_x] +=
                                        valtoadd as u32;
                                    }
                                }
                            }
                        }
                    }
                }

                x += self.stepwidth;
            }
            y += self.stepwidth;
        }

        // Now we want to use the rough voting array to find a good initialize value
        // for mean shifting. So we look for the maximal value (= most votes)
        // and convert its coordinate into a guess for the fine voting array.

        // best 2d grid position
        let (_, best_idx) = guess_pos_grid.into_iter()
            .enumerate()
            .fold((0u32, 0usize), |(prev_max, prev_idx), (idx, el)| {
                if *el > prev_max {
                    (*el, idx)
                } else {
                    (prev_max, prev_idx)
                }
            });

        // But now we need a good z coordinate
        // => use mean z coordinate of the grid part
        let grid_part_size_w = w as usize / GUESS_GRID_PARTS;
        let grid_part_size_h = h as usize / GUESS_GRID_PARTS;
        let (max_x_grid, max_y_grid) = (best_idx % GUESS_GRID_PARTS,
                                        best_idx / GUESS_GRID_PARTS);

        let grid_img = InMutSubImage::new((grid_part_size_w * max_x_grid) as u32,
                                            (grid_part_size_h * max_y_grid) as u32,
                                            grid_part_size_w as u32,
                                            grid_part_size_h as u32,
                                            img)
            .expect("Unable to creates immutable subimage");
        let zrel =
            grid_img.iter().filter(|i| i.data[0] > 0).fold((0u64, 0usize), |(sum, size), el| {
                (sum + el.data[0] as u64, size + 1)
            });
        let meanz = if zrel.1 > 0 {
            (zrel.0 as f64 / zrel.1 as f64) as f32
        } else {
            0.0
        };
        // Use the central position of this grid and then convert the coordinate to 3d
        let (max_x, max_y) = ((max_x_grid as f32 + 0.5) * grid_part_size_w as f32,
                              (max_y_grid as f32 + 0.5) * grid_part_size_h as f32);
        let max3d = intrinsic.img_to_space_coord([max_x, max_y], meanz as f32);


        // Find best rotation guess
        let (rx, ry, rz, _) = guess_rot_grid.iter()
            .filter(|&(_, _, _, &count)| count > 0)
            .fold((0, 0, 0, 0),
                  |(oldx, oldy, oldz, oldc), (x, y, z, &count)| {
                if count > oldc {
                    (x, y, z, count)
                } else {
                    (oldx, oldy, oldz, oldc)
                }
            });

        // convert the coordinate to degree and move them
        let rx = (rx as f64 * 360.0 + 180.0) / GUESS_GRID_PARTS as f64;
        let ry = (ry as f64 * 360.0 + 180.0) / GUESS_GRID_PARTS as f64;
        let rz = (rz as f64 * 360.0 + 180.0) / GUESS_GRID_PARTS as f64;

        (mid,
         (max3d[0] as i32, max3d[1] as i32, max3d[2] as i32 / ZSCALEFACTOR as i32),
         rot,
         (rx, ry, rz))
    }

    /// Use a 2d image for compute quickly a voting image.
    ///
    /// # Arguments
    /// * `img` - the depth image
    /// * `intrinsic` - intrinsic matrix of the camera used to get the depth image
    pub fn build_hough_image(&self,
                             img: Arc<DepthImage>,
                             intrinsic: &IntrinsicMatrix)
                             -> ImageBuffer<Luma<u16>, Vec<u16>> {
        use super::houghforest::HoughForestImage;
        let (h, w) = (img.height(), img.width());
        let mut hough_img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(w, h);
        let left_w = self.subimage_width / 2;
        let right_w = self.subimage_width - left_w;
        let left_h = self.subimage_height / 2;
        let right_h = self.subimage_height - left_h;
        let mut y = left_h;
        // Sliding Window
        while y < h - right_h {
            let mut x = left_w;
            while x < w - right_w {

                let z = img[(x, y)].data[0]; // get z coordinate
                // get 3d point
                let p3 = intrinsic.img_to_space_coord([x as f32, y as f32], z as f32);

                // get prediction
                let leafs = {
                    let subimg = InMutSubImage::new(x - left_w,
                                                    y - left_h,
                                                    self.subimage_width,
                                                    self.subimage_height,
                                                    img.clone())
                        .unwrap();
                    // Is Background?
                    if subimg.average_value_in_rect(Rect::new(0,
                                                              0,
                                                              self.subimage_width,
                                                              self.subimage_height)) >
                       0.0 {
                        Some(self.forest.forest_predictions(&subimg))
                    } else {
                        None
                    }
                };

                if let Some(leafs) = leafs {
                    // Use the trees to vote
                    for leaf in leafs.iter() {
                        // vote only if the current tree thinks it is realy a head image
                        if leaf.prob >= 0.95 {
                            // value to increase the voting position
                            let valtoadd = (255.0 * leaf.prob) as usize / leaf.offsets.len();
                            let valtoadd = valtoadd as u16;

                            // info!("Prob >=0.95 {:?}", leaf);

                            // Use every offset to vote on the position
                            for offs in leaf.offsets.iter() {
                                let np = p3 - *offs;
                                let p2 = intrinsic.space_to_img_coord(np);
                                let (nx, ny) = (p2.0[0] as i32, p2.0[1] as i32);
                                // force valid coordinates
                                if nx < 0 {
                                    continue;
                                }
                                let nx = nx as u32;
                                if nx >= w {
                                    continue;
                                }
                                if ny < 0 {
                                    continue;
                                }
                                let ny = ny as u32;
                                if ny >= h {
                                    continue;
                                }
                                hough_img[(nx, ny)].data[0] += valtoadd;
                            }
                        }
                        // }
                    }
                }
                x += self.stepwidth;
            }
            y += self.stepwidth;
        }
        // Use gaussian blur to avoid mean shifting and just search for a maximum value
        // see "Class-Specific Hough Forests for Object Detection" (Gall and Lempitsky)
        gaussian_blur_f32(&hough_img, self.gaussian_sigma)
    }

    /// Predict which pixel belongs to the head and which does not.
    /// # Arguments
    /// *  `img` - the depth image for which we want to predict
    pub fn predict_mask(&self, img: Arc<DepthImage>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        use super::houghforest::HoughForestImage;
        let (h, w) = (img.height(), img.width());
        let mut mask: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(w, h);
        let left_w = self.subimage_width / 2;
        let right_w = self.subimage_width - left_w;
        let left_h = self.subimage_height / 2;
        let right_h = self.subimage_height - left_h;
        let mut y = left_h;
        while y < h - right_h {
            let mut x = left_w;
            while x < w - right_w {
                let leafs = {
                    let subimg = InMutSubImage::new(x - left_w,
                                                    y - left_h,
                                                    self.subimage_width,
                                                    self.subimage_height,
                                                    img.clone())
                        .unwrap();
                    // Ignore background
                    if subimg.average_value_in_rect(Rect::new(0,
                                                              0,
                                                              self.subimage_width,
                                                              self.subimage_height)) >
                       0.0 {
                        Some(self.forest.forest_predictions(&subimg))
                    } else {
                        None
                    }
                };
                if let Some(leafs) = leafs {
                    let prob: f64 = leafs.iter().map(|x| x.prob).sum::<f64>() /
                                    (leafs.len() as f64);
                    let prob = (prob * 255.0) as u8;
                    for i in 0..self.stepwidth {
                        for j in 0..self.stepwidth {
                            if x + i < self.stepwidth / 2 || y + j < self.stepwidth / 2 {
                                continue;
                            }
                            if x + i - self.stepwidth / 2 >= w || y + j - self.stepwidth / 2 >= h {
                                continue;
                            }
                            mask[(x + i -
                                  self.stepwidth / 2,
                                  y + j -
                                  self.stepwidth / 2)]
                                .data[0] = prob;
                        }
                    }
                }
                x += self.stepwidth;
            }
            y += self.stepwidth;
        }
        mask
    }
}
