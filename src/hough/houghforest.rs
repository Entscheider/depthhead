/// Functions and structs for learning a hough forest

use meancov_estimation::*;
use stamm::tree::*;
use stamm::randforest::*;
use std::marker::PhantomData;
use std::vec::IntoIter;
use types::Rect;

/// For using a hough forest we need to get the average value
/// of an image within a rectangle.
/// This trait abstract this requirement.
pub trait HoughForestImage {
    fn average_value_in_rect(&self, rect: Rect) -> f64;
}

/// ln!(x) = ln x if x != 0 and 0 otherwise
macro_rules! ln {
     ($x: expr) => {if $x == 0f64 {0f64} else {$x.ln()} }
}

/// rel!(a,b) = a / b where a and b are forced to be a f64
macro_rules! rel {
    ($x: expr, $y: expr) => {($x as f64) / ($y as f64)}
}

/// We want to classify if a image part contains a head or not.
/// In the first case we also want to get votes for
/// the central point of the head and the rotation
#[derive(Debug)]
pub enum Truth {
    NoObject,
    Object {
        /// Vote for the central point.
        /// This offset is relative to the current
        /// center of the viewed depth image.
        offset: Vec3<f32>,
        /// Vote for the rotation of the head.
        rotation: Vec3<f64>,
    },
}

impl Truth {
    fn is_object(&self) -> bool {
        match *self {
            Truth::NoObject => false,
            _ => true,
        }
    }

    fn as_option(&self) -> Option<(Vec3<f32>, Vec3<f64>)> {
        match *self {
            Truth::NoObject => None,
            Truth::Object { offset, rotation } => Some((offset, rotation)),
        }
    }
}

/// In a hough forest a node is made up of two rectangles and a threshold.
/// If the difference of the average value of both rectangels
/// is lower than the threshold, another child will be used than
/// if it is larger.
#[derive(Serialize, Deserialize, Debug)]
pub struct NodeParam {
    r1: Rect,
    r2: Rect,
    threshold: f64,
}


/// A leaf is made up of the probability to be a head and
/// of the relative coordinates (=offset) of the head and its rotation.
#[derive(Serialize, Deserialize, Debug)]
pub struct LeafParam {
    pub prob: f64,
    pub offsets: Vec<Vec3<f32>>,
    pub rotations: Vec<Vec3<f64>>,
}

impl FromGetProbability for LeafParam {
    fn probability(&self) -> f64 {
        self.prob
    }
}

/// Parameters for training a tree of the hough forest.
/// `I` should be a HoughForestImage so that the forest can
/// be trained on these.
#[derive(Serialize, Deserialize)]
pub struct HoughTreeFunctions<I> {
    /// Size of the input image.
    /// Note that a depth image will be divided into
    /// images of this size (=> Sliding Windows)
    pub input_size: Rect,
    /// The hough forest used the average of two rectangle
    /// within the input image as an feature.
    /// The size of this rectangles is taken randomly for training.
    /// This parameter gives the minimal factor (relative to the size
    /// of the input image) such rectangles may have.
    pub min_subrect_factor: f64,
    /// The hough forest used the average of two rectangle
    /// within the input image as an feature.
    /// The size of this rectangles is taken randomly for training.
    /// This parameter gives the maximum factor (relative to the size
    /// of the input image) such rectangles may have.
    pub max_subrect_factor: f64,
    /// The number of randomly generated features used for a node
    /// to train a tree.
    pub number_of_gen_features: usize,
    /// A parameter for the weighting function (see paper)
    pub steepness: f64,
    /// The maximum depth a tree may have.
    pub max_depth: usize,
    /// For training a tree a set of trainings data is partitioned
    /// using features. So every node is trained use some subset.
    /// If the size of this subset is lower than this value,
    /// the subset won't be differentiated further (stop training for this subtree)
    pub min_subset_size: usize,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    phantom: PhantomData<I>,
}

impl<I> HoughTreeFunctions<I> {

    /// Creates a HoughTreeFunction for an Image (HoughForestImage)
    /// of type `I`.
    /// Returns None if invalid parameters are used.
    /// # Parameters
    /// - `input_size` - Size of a image used as input of this tree
    /// - `min_subrect_factor` - minimal size a rectangle used in features may have
    /// - `max_subrect_factor` - maximum size a rectangle used in features may have
    /// - `number_of_gen_features` - number of randomly generated features per node used for training
    /// - `steepness` - a parameter of the weighting function (see Paper[^1])
    ///
    /// [^1]: Multiview Facial Landmark Localization in RGB-D Images via Hierarchical Regression With Binary Patterns (Zhang et al.)
    pub fn new(input_size: Rect,
               min_subrect_factor: f64,
               max_subrect_factor: f64,
               number_of_gen_features: usize,
               steppness: f64)
               -> Option<HoughTreeFunctions<I>> {
        if min_subrect_factor > 1f64 || min_subrect_factor < 0f64 || max_subrect_factor < 0f64 ||
           max_subrect_factor > 1f64 ||
           number_of_gen_features == 0 || steppness <= 0f64 {
            return None;
        }
        Some(HoughTreeFunctions {
            input_size: input_size,
            min_subrect_factor: min_subrect_factor,
            max_subrect_factor: max_subrect_factor,
            number_of_gen_features: number_of_gen_features,
            steepness: steppness,
            max_depth: 15,
            min_subset_size: 20,
            phantom: PhantomData {},
        })
    }
}

impl<I> Clone for HoughTreeFunctions<I> {
    fn clone(&self) -> Self {
        HoughTreeFunctions {
            input_size: self.input_size.clone(),
            min_subrect_factor: self.min_subrect_factor,
            max_subrect_factor: self.max_subrect_factor,
            number_of_gen_features: self.number_of_gen_features,
            steepness: self.steepness,
            max_depth: self.max_depth,
            min_subset_size: self.min_subset_size,
            phantom: PhantomData {},
        }
    }
}

impl<I> Copy for HoughTreeFunctions<I> {}

impl<I> TreeFunction for HoughTreeFunctions<I>
    where I: HoughForestImage
{
    type Data = I;
    type Param = NodeParam;

    // Used difference of the average values within the two parts of the input image
    fn binarize(&self, param: &NodeParam, element: &I) -> Binar {
        let avg1 = element.average_value_in_rect(param.r1);
        let avg2 = element.average_value_in_rect(param.r2);
        if avg1 - avg2 > param.threshold {
            Binar::One
        } else {
            Binar::Zero
        }
    }
}

impl<I> TreeLearnFunctions for HoughTreeFunctions<I>
    where I: HoughForestImage
{
    type LeafParam = LeafParam;
    type Truth = Truth;
    type ParamIter = IntoIter<NodeParam>;
    type PredictFunction = Self;

    fn comp_leaf_data(&self, set: &[(&I, &Truth)]) -> LeafParam {
        let mut offset_list: Vec<Vec3<f32>> = vec![];
        let mut rot_list: Vec<Vec3<f64>> = vec![];
        // extract offsets and rotations
        for &(_, truth) in set.iter() {
            match *truth {
                Truth::NoObject => (),
                Truth::Object { offset, rotation } => {
                    offset_list.push(offset);
                    rot_list.push(rotation);
                }
            }
        }

        let prob = (offset_list.len() as f64) / (set.len() as f64);

        LeafParam {
            prob: prob,
            offsets: offset_list,
            rotations: rot_list,
        }
    }

    fn param_set(&self) -> Self::ParamIter {
        // Generate rectangle and thresholds randomly
        // for using them as a feature.
        self.input_size
            .random_subrect_iterator(self.min_subrect_factor,
                                     self.max_subrect_factor,
                                     -256.0,
                                     256.0)
            .unwrap()
            .take(self.number_of_gen_features as usize)
            .map(|(a, b, c)| {
                NodeParam {
                    r1: a,
                    r2: b,
                    threshold: c,
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    // This impurity is a combination of the entropy impurity and regression uncertainty.
    // See Zhang et al.
    fn impurity(&self,
                _: &NodeParam,
                set_l: &[(&I, &Truth)],
                set_r: &[(&I, &Truth)],
                depth: usize)
                -> f64 {
        use std::f64;
        fn entropy<I>(set: &[(&I, &Truth)]) -> f64 {
            let positives =
                set.iter().fold(0,
                                |sum, &(_, ref t)| if t.is_object() { sum + 1 } else { sum });
            let prob = rel!(positives, set.len());
            prob * ln!(prob) + (1f64 - prob) * ln!(1f64 - prob)
        }
        fn regression_log<I>(set: &[(&I, &Truth)]) -> f64 {
            // good is a subset of data within the face
            let good: Vec<_> = set.iter().filter_map(|&(_, x)| x.as_option()).collect();
            let offsets: Vec<_> =
                good.iter().map(|x| Vec3::new([x.0[0] as f64, x.0[1] as f64, x.0[2] as f64])).collect();
            let rotations: Vec<_> = good.iter().map(|x| x.1).collect();
            if offsets.is_empty() {
                // not data => seems to be a good distribution
                return 0.0;
            }
            // Estimation of the covariance
            let (_, cov_of) = estimate_mean_cov(&offsets[..]).unwrap();
            let (_, cov_rot) = estimate_mean_cov(&rotations[..]).unwrap();
            match cov_of.det() + cov_rot.det() {
                x if x > 0f64 => x.ln(),
                x if x < -0.001f64 => unreachable!(), // Covariance-Matrix has to be positive
                _ => 0f64,
            }
        }
        let count = set_l.len() + set_r.len();
        let left_factor = rel!(set_l.len(), count);
        let right_factor = rel!(set_r.len(), count);
        let impurity = -(left_factor * entropy(set_l) + right_factor * entropy(set_r));
        let regression_uncert = left_factor * regression_log(set_l) +
                                right_factor * regression_log(set_r);

        let e_factor = -rel!(depth, self.steepness);
        let f = e_factor.exp();
        let res = impurity  + (1f64 - f) * regression_uncert;
        assert!(res.is_finite());
        res
    }

    fn as_predict_learn_func(self) -> Self {
        self
    }

    // Stop training if size of elements is too low or the depth is too high.
    fn early_stop(&self, depth: usize, elements: &[(&Self::Data, &Self::Truth)]) -> bool {
        if elements.iter().all(|x| !x.1.is_object()) {
            return true; // No element represents a head => no further  partitioning is needed
        }
        if depth >= self.max_depth || elements.len() < self.min_subset_size {
            return true;
        }
        false
    }
}

#[derive(Serialize, Deserialize)]
pub struct HoughForestParameter<I> {
    pub tree_param: HoughTreeFunctions<I>,
    pub number_of_trees: usize,
    pub size_of_subset_per_training: usize,
}

impl<I> HoughForestParameter<I>
    where I: HoughForestImage
{

    /// Train a Hough-Forest using the given training set
    pub fn train_tree(&self,
                      train_set: &[(&I, &Truth)])
                      -> Option<RandomForest<LeafParam, HoughTreeFunctions<I>>>
        where I: HoughForestImage
    {
        let param = RandomForestLearnParam::new(self.number_of_trees,
                                                self.size_of_subset_per_training,
                                                self.tree_param);
        param.train_forest(train_set)
    }
}

impl<I> HoughForestParameter<I>
    where I: HoughForestImage + Sync + Send,
          HoughTreeFunctions<I>: Sync + Send
{
    /// Train a Hough-Forest using the given trainin set.
    /// Multiple trees will be trained parallel.
    pub fn train_tree_parallel(&self,
                               train_set: &[(&I, &Truth)])
                               -> Option<RandomForest<LeafParam, HoughTreeFunctions<I>>> {
        let param = RandomForestLearnParam::new(self.number_of_trees,
                                                self.size_of_subset_per_training,
                                                self.tree_param);
        param.train_forest_parallel(train_set)
    }
}
