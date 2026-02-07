2
2
0
2

t
c
O
2
1

]

V
C
.
s
c
[

2
v
5
6
6
0
0
.
6
0
2
2
:
v
i
X
r
a

MonoSDF: Exploring Monocular Geometric Cues
for Neural Implicit Surface Reconstruction

Zehao Yu1 Songyou Peng2,3 Michael Niemeyer1,3 Torsten Sattler4 Andreas Geiger1,3

1University of Tübingen

2ETH Zurich

3MPI for Intelligent Systems, Tübingen

4Czech Technical University in Prague

https://niujinshuchong.github.io/monosdf

Abstract

In recent years, neural implicit surface reconstruction methods have become popu-
lar for multi-view 3D reconstruction. In contrast to traditional multi-view stereo
methods, these approaches tend to produce smoother and more complete recon-
structions due to the inductive smoothness bias of neural networks. State-of-the-art
neural implicit methods allow for high-quality reconstructions of simple scenes
from many input views. Yet, their performance drops signiﬁcantly for larger and
more complex scenes and scenes captured from sparse viewpoints. This is caused
primarily by the inherent ambiguity in the RGB reconstruction loss that does not
provide enough constraints, in particular in less-observed and textureless areas.
Motivated by recent advances in the area of monocular geometry prediction, we
systematically explore the utility these cues provide for improving neural implicit
surface reconstruction. We demonstrate that depth and normal cues, predicted by
general-purpose monocular estimators, signiﬁcantly improve reconstruction quality
and optimization time. Further, we analyse and investigate multiple design choices
for representing neural implicit surfaces, ranging from monolithic MLP models
over single-grid to multi-resolution grid representations. We observe that geometric
monocular priors improve performance both for small-scale single-object as well
as large-scale multi-object scenes, independent of the choice of representation.

1

Introduction

3D reconstruction from multiple RGB images is a fundamental problem in computer vision with
various applications in robotics, graphics, animation, virtual reality, and more. Recently, coordinate-
based neural networks have emerged as a powerful tool for representing 3D geometry and appearance.
The key idea is to use compact, memory efﬁcient multi-layer perceptrons (MLPs) to parameterize
implicit shape representations such as occupancy or signed distance ﬁelds. While early works [9, 42,
50] relied on 3D supervision, several recent works [47, 66, 82] use differentiable surface rendering to
reconstruct scenes from multi-view images. At the same time, neural radiance ﬁelds (NeRFs) [44]
achieved impressive novel view synthesis results with volume rendering techniques.
[49, 76, 81]
combine surface and volume rendering for the task of 3D reconstruction by expressing volume density
as a function of the underlying 3D surface, which in turn improves scene geometry.

Current neural implicit-based surface reconstruction approaches achieve impressive reconstruction
results for simple scenes with dense viewpoint sampling. Yet, as shown in the ﬁrst row of Fig. 1,
they struggle in the presence of limited input views (DTU with 3 views) or for scenes that contain
large textureless regions (walls in ScanNet or Tanks & Temples). A key reason for this behavior is
that these model are optimized using a per-pixel RGB reconstruction loss. Using only RGB images
as input leads to an underconstrained problem as there exist an inﬁnite number of photo-consistent

36th Conference on Neural Information Processing Systems (NeurIPS 2022).

 
 
 
 
 
 
]
1
8
[
F
D
S
l
o
V

s
e
u
C

r
a
l
u
c
o
n
o
M
+

DTU (3 views)

ScanNet (464 views)

Tanks & Temples (298 views)

Figure 1: MonoSDF. Top: State-of-the-art neural implicit surface reconstruction methods fail in
the presence of limited input views or when applied to complex multi-object scenes. Bottom: We
demonstrate that incorporating geometric cues from general-purpose monocular predictors enables
scaling to larger scenes while yielding more accurate reconstructions and speeding up optimization.
An image resolution of 384 × 384 pixels was used for all results shown above.

explanations [5, 90]. Previous works address this problem by incorporating priors on the structure of
the scene into the optimization process, e.g., depth smoothness [46], surface smoothness [49, 91],
semantic similarity [30], or Manhattan world assumptions [21]. In this paper, we explore monocular
geometric priors as they are readily available and efﬁcient to compute. We show that using such
priors signiﬁcantly improves 3D reconstruction quality in challenging scenarios (see second row of
Fig. 1).

Estimating geometric cues such as depth and normals from a single image has been an active
research area for decades. The seminal work by Eigen et al. [18, 19] showed that learned models
based on deep convolutional neural networks (CNNs) signiﬁcantly improved over early work in this
area [24–27, 59–61]. Recent work [17, 55, 56], in particular Omnidata [17], has made signiﬁcant
headway in terms of prediction quality and generalization to new scenes using very large datasets for
training. These strong results on individual images, and the fact that monocular geometric cues can
be computed efﬁciently, naturally lead to the question whether such models are able to provide the
additional constraints required by implicit neural surface reconstruction approaches to handle more
challenging settings.

This paper describes a framework, called MonoSDF, for integrating monocular geometric priors into
neural implicit surface reconstruction methods: given multi-view images, we infer depth and surface
normals for each image, and use them as additional supervision signals during optimization together
with the RGB image reconstruction loss. We observe that these priors lead to signiﬁcant gains in
reconstruction quality, especially in textureless and less-observed areas as shown in Fig. 1. This is
due to the fact that the photometric consistency cues used by surface reconstruction methods and the
recognition cues used by monocular networks are complementary: while photometric consistency
fails in texturless regions such as walls, surface normals can be predicted reliably in these areas due
to the structured 3D scene layout. Conversely, photoconsistency cues allow for establishing globally
accurate 3D geometry in textured regions, while normal and (relative) depth cues only provide local
geometric information.

Apart from incorporating monocular geometric cues, we provide a systematic study and analysis
of state-of-the-art design choices for coordinate-based neural representations in the context of
implicit surface reconstruction. More speciﬁcally, we investigate the following architectures: a
single, large MLP [49, 76, 81, 82], a dense SDF grid [31], a single feature grid [28, 38, 53, 54] and
multi-resolution feature grids [10, 22, 45, 69, 92]. We observe that MLPs act globally and exhibit
an inductive smoothness bias while being computationally expensive to optimize and evaluate. In
contrast, grid-based representations beneﬁt from locality during optimization and evaluation, hence

2

they are computationally more efﬁcient. However, reconstructions are noisier for sparse views or
less-observed areas. Including monocular geometric priors improves neural implicit reconstruction
results across different settings with faster convergence times and independent of the underlying
representation.

In summary, we make the following contributions:

• We introduce MonoSDF, a novel framework which exploits monocular geometric cues to improve
multi-view 3D reconstruction quality, efﬁciency, and scalability for neural implicit surface models.

• We provide a systematic comparison and detailed analysis of design choices of neural implicit

surface representations, including vanilla MLP and grid-based approaches.

• We conduct extensive experiments on multiple challenging datasets, ranging from object-level
reconstruction on the DTU dataset [1], over room-level reconstruction on Replica [67] and Scan-
Net [13], to large-scale indoor scene reconstruction on Tanks and Temples [34].

2 Related Work

Architectures for Neural Implicit Scene Representations. Neural implicit scene representations
or neural ﬁelds [78] have recently gained popularity for representing 3D geometry due to their
expressiveness and low memory footprint. Seminal works [9, 42, 50] use a single MLP as the scene
representation and show impressive object-level reconstruction quality, but they do not scale to more
complicated or large-scale scenes due to the limited model capacity. Follow-up works [10, 22, 41,
45, 54, 69, 92] combine an MLP decoder with one or multi-level voxel grids of low-dimensional
features. Such hybrid representations are able to better represent ﬁne geometric details and can be
evaluated fast. However, they lead to a larger memory footprint with increasing scene size. In this
paper we provide a systematic comparison of four architectural design choices for implicit surface
reconstruction.
3D Reconstruction from Multi-view Images. Reconstructing the underlying 3D geometry from
multi-view images is a long-standing goal of computer vision. Classic multi-view stereo (MVS)
methods [2, 6–8, 35, 35, 62, 64, 65] consider either feature matching for depth estimation [6, 62]
or represent shapes with voxels [2, 7, 8, 35, 51, 64, 72, 73]. Learning-based MVS methods usually
replace some parts of the classic MVS pipeline, e.g., feature matching [23, 36, 40, 74, 88], depth
fusion [16, 57], or inferring depth from multi-view images [29, 79, 80, 85]. In contrast to the explicit
scene representations used by classic MVS algorithms, recent neural approaches [39, 48, 82] represent
surfaces via a single MLP with continuous outputs. Learned purely from posed 2D images, they
show appealing reconstruction results and do not suffer from discretization. However, accurate
object masks are required. Inspired by the density-based volume rendering in NeRF [44], which
demonstrated impressive view synthesis without object masks, several works [49, 76, 81] use volume
rendering for neural implicit surface reconstruction without masks. However, these methods lead to
poor results in large-scale scenes with textureless regions. In this work, we show that incorporating
monocular priors allows these approaches to obtain signiﬁcantly more detailed reconstructions and to
scale to larger and more challenging scenes.
Incorporating Priors into Neural Scene Representations. Several researchers proposed to in-
corporate priors such as depth smoothness [46], semantic similarity [30], or sparse MVS point
clouds [58] for the task of novel view synthesis from sparse inputs. In contrast, in this work, our
focus is on implicit 3D surface reconstruction. Concurrently, Manhattan-SDF [21] uses dense MVS
depth maps from COLMAP [63] as supervision and adopts Manhattan world priors [11] to handle
low-textured planar regions corresponding to walls, ﬂoors, etc. Our approach is based on the observa-
tion that data-driven monocular depth and normal predictions [17] provide high-quality priors for
the full scene. Incorporating these priors into the optimization of neural implicit surfaces not only
removes the Manhattan world assumption [11] but also results in improved reconstruction quality
and a simpler pipeline.1 Compared to NeuRIS [75], a concurrent work that proposes to use normal
priors for indoor scene reconstruction, we integrate monocular depth cues and further demonstrate
the effectiveness of monocular cues on various neural scene representations, ranging from MLP to
multi-resolution feature grids.

1Manhattan-SDF [21] requires semantic segmentation to determine where to enforce the assumption.

3

Figure 2: Overview. In this work we use monocular geometric cues predicted by a general-purpose
pretrained network to guide the optimization of neural implicit surface models. More speciﬁcally, for
a batch of rays, we volume render predicted RGB colors, depth, and normals, and optimize wrt. the
input RGB images and monocular geometric cues. Further, we investigate different design choices
for neural implicit architectures and provide an in-depth analysis. For clarity, we only show the SDF
and not the color prediction branch above.

3 Method

Our goal is to recover the underlying scene geometry from multiple posed images while utilizing
monocular geometric cues to guide the optimization process. To this end, we ﬁrst review neural
implicit scene representations and various design choices in Section 3.1 and discuss how to perform
volume rendering of these representations in Section 3.2. Next, we introduce the monocular geometric
cues we investigate in our study in Section 3.3 and discuss loss functions and the overall optimization
process in Section 3.4. An overview of our framework is provided in Fig. 2.

3.1

Implicit Scene Representations

We represent scene geometry as a signed distance function (SDF). A signed distance function is a
continuous function f that, for a given 3D point, returns the point’s distance to the closest surface:

f : R3 → R

x (cid:55)→ s = SDF(x) .

(1)

Here, x is the 3D point and s denotes the corresponding SDF value. In this work, we parameterize
the SDF function with learnable parameters θ and investigate several different design choices for
representing the function: explicit as a dense grid of learnable SDF values, implicit as a single MLP,
or hybrid using an MLP in combination with single- or multi-resolution feature grids.
Dense SDF Grid. The most straightforward way of parameterizing an SDF is to directly store SDF
values in each cell of a discretized volume Gθ with resolution of RH × RW × RD [31]. To query the
SDF value ˆs for an arbitrary point x from the dense SDF grid, we can use any interpolation operation:

In our experiments, we implement interp as trilinear interpolation.
Single MLP. The SDF function can also be parameterized by a single MLP [50] fθ:

ˆs = interp(x, Gθ) .

ˆs = fθ(γ(x)) ,

(2)

(3)

where ˆs is the predicted SDF value and γ corresponds to a ﬁxed positional encoding [44, 70] mapping
x to a higher dimensional space. After their introduction to novel view synthesis [44], positional

4

encoding functions are now widely used for neural implicit surface reconstruction [49, 76, 81, 82] as
they increase the expressiveness of coordinate-based networks [70].
Single-Resolution Feature Grid with MLP Decoder. We can also combine both parameterizations
and use a feature-conditioned MLP fθ together with a feature grid Φθ with a resolution of R3, where
each cell of the grid stores a feature vector [28, 38, 54, 69] instead of directly storing SDF values:

ˆs = fθ(γ(x), interp(x, Φθ)) .

(4)

Note that the MLP fθ is conditioned on the interpolated local feature vector from the feature grid Φθ.
Multi-Resolution Feature Grids with MLP Decoder. Instead of using a single feature grid Φθ, one
can also employ multi-resolution feature grids {Φl
l=1 with resolutions Rl [10, 22, 45, 69, 92]. The
resolutions are sampled in geometric space [45] to combine features at different frequencies:

θ}L

Rl := (cid:98)Rminbl(cid:99)

b := exp

(cid:18) ln Rmax − ln Rmin
L − 1

(cid:19)

,

(5)

where Rmin, Rmax are the coarsest and ﬁnest resolution, respectively. Similarly, we extract the
interpolated features at each level and concatenate them together:

ˆs = fθ(γ(x), {interp(x, Φl

θ)}l)) .

(6)

As the total number of grid cells grows cubically, we use a ﬁxed number of parameters to store
the feature grids and use a spatial hash function to index the feature vector at ﬁner levels [45] (see
supplementary for details).
Color Prediction. In addition to the 3D geometry, we also predict color values such that our model
can be optimized with a reconstruction loss. Following [82], we therefore deﬁne a second function
cθ

ˆc = cθ(x, v, ˆn, ˆz)
that predicts a RGB color value ˆc for a 3D point x and a viewing direction v. The 3D unit normal ˆn
is the analytical gradient of our SDF function. The feature vector ˆz is the output of a second linear
head of the SDF network as in [82]. We parameterize cθ with a two-layer MLP with network weights
θ. In case of the dense grid SDF parameterization, we similarly optimize a dense feature grid and
obtain the feature vector ˆz via the interpolation function interp.

(7)

3.2 Volume Rendering of Implicit Surfaces

Following recent work [49,76,81,82], we optimize the implicit representations described in Section 3.1
via an image-based reconstruction loss using differentiable volume rendering. More speciﬁcally, to
render a pixel, we cast a ray r from the camera center o through the pixel along its view direction v.
We sample M points xi
r. We
follow [81] to transform the SDF values ˆsi

rv along the ray and predict their SDF ˆsi
r to density values σi

r for volume rendering:

r and color values ˆci

r = o + ti

σβ(s) =






(cid:17)

(cid:16) s
β
1 − 1
2 exp

1
2β exp
(cid:16)
1
β

(cid:17)(cid:17)

(cid:16)

− s
β

if s ≤ 0

if s > 0

,

(8)

where β is a learnable parameter. Following NeRF [44], the color ˆC(r) for the current ray r is
computed via numerical integration:

ˆC(r) =

M
(cid:88)

i=1

r αi
T i

r ˆci
r

T i
r =

i−1
(cid:89)

j=1

(cid:0)1 − αj

r

(cid:1)

r = 1 − exp (cid:0)−σi
αi

rδi
r

(cid:1) ,

(9)

r and αi

r denote the transmittance and alpha value of sample point i along ray r, respectively,
r is the distance between neighboring sample points. Similarly, we compute the depth ˆD(r) and

where T i
and δi
normal ˆN (r) of the surface intersecting the current ray as:

ˆD(r) =

M
(cid:88)

i=1

r αi
T i

r ti
r

ˆN (r) =

M
(cid:88)

i=1

5

r αi
T i

r ˆni

r .

(10)

3.3 Exploiting Monocular Geometric Cues

Unifying volume rendering with implicit surfaces leads to impressive 3D reconstruction results. Yet,
this approach struggles with more complex scenes especially in textureless and sparsely covered
regions. To overcome this limitation, we use readily available, efﬁcient-to-compute monocular
geometric priors thereby improving neural implicit surface methods.
Monocular Depth Cues. One common monocular geometric cue is a monocular depth map, which
can be easily obtained via an off-the-shelf monocular depth predictor. More speciﬁcally, we use a
pretrained Omnidata model [17] to predict a depth map ¯D for each input RGB image. Note that the
absolute scale is difﬁcult to estimate in general scenes, so ¯D must be considered as a relative cue.
However, this relative depth information is provided also over larger distances in the image.
Monocular Normal Cues. Another geometric cue we use is the surface normal. Similar to the
depth cues, we apply the same pretrained Omnidata model to acquire a normal map ¯N for each RGB
image. Unlike depth cues that provide semi-local relative information, normal cues are local and
capture geometric detail. We hence expect that surface normals and depth are complementary to each
other.

3.4 Optimization

Reconstruction Loss. Eq. (9) provides a linkage from the 3D scene representation to 2D observa-
tions. We can therefore optimize the scene representation with a simple RGB reconstruction loss:

Lrgb =

(cid:88)

r∈R

(cid:107) ˆC(r) − C(r)(cid:107)1 .

(11)

Here R denotes the set of pixels/rays in the minibatch and C(r) is the observed pixel color.
Eikonal Loss. Following common practice, we also add an Eikonal term [20] on the sampled points
to regularize SDF values in 3D space

Leikonal =

(cid:88)

x∈X

((cid:107)∇fθ(x)(cid:107)2 − 1)2 ,

(12)

where X are a set of uniformly sampled points together with near-surface points [81].
Depth Consistency Loss. Besides Lrgb and Leikonal, we also enforce consistency between our
rendered expected depth ˆD and the monocular depth ¯D:

Ldepth =

(cid:88)

r∈R

(cid:107)(w ˆD(r) + q) − ¯D(r)(cid:107)

2

,

(13)

where w and q are the scale and shift used to align ˆD and ¯D since ¯D is deﬁned only up to scale.
Note that these factors have to be estimated individually per batch as the depth maps predicted for
different batches can differ in scale and shift. Speciﬁcally, we solve for w and q with a least-squares
criterion [19, 56] which has a closed-form solution (see supplementary for details).
Normal Consistency Loss. Similarly, we impose consistency on the volume-rendered normal ˆN
and the predicted monocular normals ¯N transformed to the same coordinate system with angular and
L1 losses [17]:

Lnormal =

(cid:88)

(cid:107) ˆN (r) − ¯N (r)(cid:107)1 + (cid:107)1 − ˆN (r)(cid:62) ¯N (r)(cid:107)1 .

(14)

The overall loss we use to optimize our implicit surfaces jointly with the appearance network is:

r∈R

L = Lrgb + λ1Leikonal + λ2Ldepth + λ3Lnormal .

(15)

Implementation Details. We implement our method in PyTorch [52] and use the Adam opti-
mizer [33] with a learning rate of 5e-4 for neural networks and 1e-2 for feature grids and dense SDF
grids. We set λ1, λ2, λ3 to 0.1, 0.1, 0.05, respectively. We sample 1024 rays per iteration and apply
the error-bounded sampling strategy introduced by [81] to sample points along each ray. For MLPs
and feature grids, we adapt the architecture and initialization scheme from [81] and [45], respectively.
For obtaining monocular cues, we ﬁrst resize each image and center crop it to 384 × 384, which we
then feed as input to the pretrained Omnidata model [17]. See supplementary for more details.

6

Dense SDF Grid

MLP

Single-Res.
Fea. Grid

Multi-Res.
Fea. Grids

Ground Truth

Figure 3: Architectural Ablation Study. Comparing different design choices for neural implicit
surface representations, we observe that a dense SDF grid leads to noisy reconstructions due to a
missing smoothness bias. The MLP and the Single-Res. Fea. Grid improve results, but geometry
tends to be overly smooth with missing details. The best results are obtained using Multi-Res. Fea.
Grids.

4 Experiments

We ﬁrst analyze different architectural design choices and perform ablation studies wrt. monocular
cues and optimization time on a room-level dataset (Replica) with perfect ground truth. Next, we
provide qualitative and quantitative comparisons against state-of-the-art baselines on real-world
indoor scenes. Finally, we evaluate our method on object-level reconstruction for both sparse input
and dense input scenarios.
Datasets. While previous neural implicit-based reconstruction methods mainly focused on single-
object scenes with many input views, in this work, we investigate the importance of monocular
geometric cues for scaling to more complex scenes. Thus we consider: a) Real-world indoor scans:
Replica [67] and ScanNet [13]; b) Real-world large-scale indoor scenes: Tanks and Temples [34]
advanced scenes; c) Object-level scenes: DTU [1] in the sparse 3-view setting from [46, 84].
Baselines. We compare against a) state-of-the-art neural implicit surfaces methods: UNISURF [49],
VolSDF [81], NeuS [76], and Manhattan-SDF [21]. b) Classic MVS methods: COLMAP [62]
and a state-of-the-art commercial software (RealityCapture2). c) TSDF-Fusion [12] with predicted
monocular depth cues, where GT depth maps are used to recover the scale and shift values (cf.
Eq. (13)). This baseline shows the reconstruction quality if only monocular depth cues and no
implicit surface model is used.
Evaluation Metrics. For DTU, we follow the ofﬁcial evaluation protocol and report the Chamfer
distance. For Replica and ScanNet, following [21, 42, 53, 54, 68, 92], we report the Chamfer Distance,
the F-score with a threshold of 5cm, as well as a Normal Consistency measure.

4.1 Ablation Study

Normal C.↑ Chamfer-L1 ↓ F-score ↑

MLP [81]
Dense SDF Grid
Single-res. Fea. Grid
Multi-res. Fea. Grids

We ﬁrst analyze different scene representation choices on the Replica dataset. Next, we ablate the
impact of our geometric cues on reconstruction quality and convergence time.
Architecture Choices for Scene Representations.
We compare the four different scene geometry repre-
sentations introduced in Section 3.1 and report metrics
averaged over the Replica dataset in Table 1. Note that
no monocular geometric cues are used here. We ﬁrst
observe that using a single MLP as the scene geometry
representation leads to decent results, but the recon-
struction tends to be over-smooth (see Table 1 and Fig. 3). For grid-based representations, optimizing
a dense SDF grid leads to a signiﬁcantly worse performance compared to all other neural implicit
scene representations, even with careful parameter tuning. The reason is the lack of a smoothness bias:
The SDF values in grid cells are all stored and optimized independently of each other, hence there is
no local or global smoothness bias. In contrast, the Single-Res. Fea. Grid replaces the SDF value
in each grid cell with a low-dimensional latent code, and uses a shallow MLP conditioned on these
features to read out SDF values of arbitrary 3D points. This modiﬁcation leads to a notable boost
in reconstruction quality over the dense grid, performing similarly well as the single MLP. Using a
Multi-Res. Fea. Grids as in [45] further increases performance. We observe that the Multi-Res. Fea.
Grids is the best-performing grid-based model, and from now on we report results for the single MLP

Table 1: Architectural Ablation on Replica.

66.88
15.50
64.22
78.38

86.48
57.30
86.41
87.95

6.75
26.68
6.28
5.03

2https://www.capturingreality.com/

7

No Cue

+ Depth

+ Normal

+ Both

Ground Truth

Figure 4: Ablation of Monocular Geometric Cues. Monocular geometric cues signiﬁcantly im-
prove reconstruction quality for both architectures (we show our MLP variant). With monocular
depth cues, the recovered geometry contains more details and a better overall structure. With normal
cues, missing details are added and the results become smoother. Using both cues leads to the best
performance.

MLP

No Cues
Only Depth
Only Normal
Both Cues
No Cues

Multi-Res. Only Depth

Grids Only Normal

Both Cues

Normal C.↑ Chamfer-L1 ↓ F-score ↑
6.75
4.26
3.19
2.94
5.03
3.75
3.61
3.23

66.88
76.42
85.84
86.18
78.38
80.32
81.28
85.91

86.48
90.56
91.35
92.11
87.95
90.87
89.90
90.93

(a) Different Cues

(b) Optimization Time

Table 2: Ablation of Monocular Geometric Cues. a.) We report reconstruction results on Replica
for MLP and Multi-Res. Grids with and without the monocular geometric cues. We observe that
monocular cues improve reconstruction quality for both architectures, and using both cues in combi-
nation leads to the best performance. b.) The optimization speed becomes signiﬁcantly faster when
incorporating monocular cues. Comparing the two architectures, we observe that the grid approach
yields faster convergences while the MLP with both cues leads to the best results.

and the Multi-Res. Feature Grids. For simplicity, we will refer to the multi-resolution feature grids as
Multi-Res. Grids or Grids in the following.
Ablation of Different Cues. We now investigate the effectiveness of different monocular geometric
cues for the two chosen representations. Table 2 (a) and Fig. 4 show that, for both representations,
using either one or both monocular cues signiﬁcantly boosts reconstruction quality. We also ﬁnd
both cues to be complementary, with the best performance being achieved when using both. Similar
behavior can be observed for the other two representations (cf. supplementary material). It is worth
noting that the differences between the two representations become negligible when using monocular
cues, indicating that those serve as a general drop-in to improve reconstruction quality.
Optimization Time. Table 2 (b) shows optimization time for the two scene representations with and
without cues. We see that the Multi-Res. Grids converge faster than the single MLP model. Further,
adding the monocular cues signiﬁcantly speeds up the convergence process. After only 10K iterations,
both representations perform better than the converged models without monocular cues. Note that the
overhead required for incorporating the monocular cues into the optimization process is small and
can be neglected. An extended version of Table 2 (b) can be found in the supplementary materials.

4.2 Real-world Large-scale Scene Reconstruction

To show the effectiveness of our method for large-scale scene reconstruction, we compare against
various baselines on two challenging large-scale indoor datasets.
ScanNet. On ScanNet, we use the test split from [21] and also follow their evaluation protocol in
which depth maps are rendered from input camera poses and then re-fused using TSDF Fusion [12] to
evaluate only observed areas. We observe in Table 3 that our MLP variant outperforms all baselines
achieving smoother reconstructions with more ﬁne details. Note that we outperform concurrent
work [75]. Further, we ﬁnd that the MLP variant performs signiﬁcantly better than using Multi-Res.
Grids. ScanNet’s RGB images contain motion blur and the camera poses are also noisy. This can be
harmful to the local geometry updates in grid-based representations, while MLPs are more robust to
this noise due to their smoothness bias.

8

5204060Iterations (×103)0.20.40.60.81F-scoreMLPMLP (w/ Cues)GridsGrids (w/ Cues)COLMAP [62]

VolSDF [44]

Manhattan-SDF [21]

Ours (MLP)

Ground Truth

COLMAP [62] UNISURF [49] NeuS [76] VolSDF [81] M-SDF [21] NeuRIS [75] Ours (Grids) Ours (MLP)

Chamfer-L1 ↓
F-score↑

0.141
0.537

0.359
0.267

0.194
0.291

0.267
0.364

0.070
0.602

0.050
0.692

0.064
0.626

0.042
0.733

Table 3: Scene-level Reconstruction on ScanNet. Colmap and VolSDF do not lead to competitive
reconstructions. Manhatten-SDF achieves compelling results, but less-observed areas are noisier and
details are missing. In contrast, our approaches reconstruct smooth and details surfaces, achieving
the best results. Further, MLPs are more robust to the motion blur and noise in camera poses.

Tanks & Temples. To further investigate the scalability of our method to larger-scale scenes, we
conduct experiments on the Tanks and Temples advanced sets. The qualitative results in Fig. 1 show
that the monocular cues signiﬁcantly boost the performance of VolSDF [81], making MonoSDF the
ﬁrst neural implicit model achieving reasonable results on such a large-scale indoor scene. See the
supplementary material for more visual comparisons and discussions.

4.3 Object-level Reconstruction from Sparse Views

We now evaluate our method on another challenging task: reconstructing single objects from sparse
input views. We adopt the test split from [81,82] on DTU and choose three input views following [46].

We ﬁrst observe in Table 4 and Fig. 1 that without the usage of the
monocular geometric cues, neither the MLP (VolSDF [81]) nor the
Multi-Res. Grids work well with only 3 input views. When incorpo-
rating the cues, the results for both representations are signiﬁcantly
improved. Interestingly, the grid-based representations perform
inferior to a single MLP as they are updated locally and do not
beneﬁt from the inductive bias of a monolithic MLP representation.

Chamfer-L1 ↓

TSDF-Fusion [12]
COLMAP [62]
RealityCapture
Grids
Grids w/ cues
MLP [81]
MLP w/ cues

4.80
2.56
2.84
6.47
3.68
4.21
1.86

Comparing against TSDF Fusion [12] that fused predicted depth
cues from all views into a TSDF volume without any optimiza-
tion, we observe that this baseline has difﬁculties in reconstructing
meaningful details due to inconsistencies in the monocular depth
cues. Note that this baseline uses the GT depth maps from [16] to
compute scale and shift for the depth cues. Classic MVS methods
perform well quantitatively, but they heavily rely on dense matching, and in case of three input
images, this inevitably leads to incomplete reconstructions (see supplementary material). In contrast,
our approach combines neural implicit surface representations with the beneﬁts from monocular
geometric cues that are more robust to less-observed regions.

Table 4: Reconstruction on
DTU (3 Views). We report the av-
erage over the test split from [81]
(see supplementary for per-object
results).

4.4 Object-level Reconstruction from Dense Views

To further investigate the effectiveness and ﬂexibility of our method, we evaluate our approach on
the DTU dataset with all input views, which is a common setting in recent work [49, 77, 81]. In this
experiment, we simply resize the low-resolution monocular cues to full resolution (from 384 × 384
to 1200 × 1200 pixels) while keeping the image ratio. As the original image is of size 1200 × 1600,
the monocular cues are missing in the left and right part of the image. Therefore, we only use the
monocular cues where they are available.

As shown in Table 5, our approach with MLP architecture achieves reconstruction quality similar
to state-of-the-art methods [49, 77, 81]. This is reasonable as the dense input views provide enough
constraints and the prior information from monocular cues is negligible. However, our method with
multi-resolution feature grid architecture outperforms previous work by a large margin. We attribute
this to the expressiveness of multi-resolution feature grids where monocular cues are still effective to

9

NeuS [77]

VolSDF [81]

Ours (MLP)

Ours (Grids)

Ground Truth View

Scan

24

37

40

55

63

65

69

83

97 105 106 110 114 118 122 Mean

COLMAP 0.81 2.05 0.73 1.22 1.79 1.58 1.02 3.05 1.40 2.05 1.00 1.32 0.49 0.78 1.17 1.36
NeRF [44] 1.90 1.60 1.85 0.58 2.28 1.27 1.47 1.67 2.05 1.07 0.88 2.53 1.06 1.15 0.96 1.49
UniSurf [49] 1.32 1.36 1.72 0.44 1.35 0.79 0.80 1.49 1.37 0.89 0.59 1.47 0.46 0.59 0.62 1.02
NeuS [77] 1.00 1.37 0.93 0.43 1.10 0.65 0.57 1.48 1.09 0.83 0.52 1.20 0.35 0.49 0.54 0.84
VolSDF [81] 1.14 1.26 0.81 0.49 1.25 0.70 0.72 1.29 1.18 0.70 0.66 1.08 0.42 0.61 0.55 0.86

Ours (MLP) 0.83 1.61 0.65 0.47 0.92 0.87 0.87 1.30 1.25 0.68 0.65 0.96 0.41 0.62 0.58 0.84
Ours(Grids) 0.66 0.88 0.43 0.40 0.87 0.78 0.81 1.23 1.18 0.66 0.66 0.96 0.41 0.57 0.51 0.73

Table 5: Object-level Reconstruction on DTU Dataset will All Input Views. We compare Chamfer
distance with state-of-the-art methods. Our approach with MLP achieves similar results to previous
methods, while our method with multi-resolution feature grids leads to more detailed surfaces and
outperforms previous work by a large margin.

suppress noise and therefore can reconstruct smooth and detailed surfaces. We kindly refer the reader
to the supplementary material for additional visual comparisons.

5 Conclusion

We have presented MonoSDF, a novel framework that systematically explores how monocular
geometric cues can be incorporated into the optimization of neural implicit surfaces from multi-
view images. We show that such easy-to-obtain monocular cues can signiﬁcantly improve 3D
reconstruction quality, efﬁciency, and scalability for a variety of neural implicit representations.
When using monocular cues, a simple MLP architecture performs best overall, demonstrating that
MLPs in principle are able to represent complex scenes, albeit being slower to converge compared to
grid-based representations. Multi-resolution feature grids in general can converge fast and capture
details, but are less robust to noise and ambiguities in the input images.
Limitations.
The performance of our model depends on the quality of the monocular cues.
Filtering strategies to handle failures of the monocular predictor are thus a promising direction to
further improve reconstruction quality. We kindly refer the reader to the supplementary material
for additional analysis. While we demonstrated that integrating depth and normal cues signiﬁcantly
improves reconstruction, exploring other cues such as occlusion edges, plane, or curvature [17, 87]
is an interesting future direction. We are currently limited by the low-resolution (384 × 384 pixels)
output of the Omnidata model [17] and plan to explore different ways of using higher-resolution
cues. We provide some preliminary results of using high-resolution cues in the supplementary. Joint
optimization of scene representations and camera parameters [4, 92] is another interesting direction,
especially for multi-resolution grids, in order to better handle noisy camera poses.

Acknowledgments and Disclosure of Funding

This work was supported by an NVIDIA research gift. We thank the Max Planck ETH Center for
Learning Systems (CLS) for supporting SP and the International Max Planck Research School for
Intelligent Systems (IMPRS-IS) for supporting MN. ZY is supported by BMWi in the project KI
Delta Learning (project number 19A19013O). AG is supported by the ERC Starting Grant LEGO-
3D (850533) and DFG EXC number 2064/1 - project number 390727645. TS is supported by
the EU Horizon 2020 project RICAIP (grant agreeement No.857306), and the European Regional
Development Fund under project IMPACT (No. CZ.02.1.01/0.0/0.0/15_003/0000468). We thank
the authors of Manhattan-SDF and NeuRIS for sharing results on ScanNet. We also thank Christian
Reiser and Zijian Dong for proofreading.

10

References

[1] H. Aanæs, R. R. Jensen, G. Vogiatzis, E. Tola, and A. B. Dahl. Large-scale data for multiple-
view stereopsis. International Journal of Computer Vision (IJCV), 120(2):153–168, 2016. 3, 7,
18

[2] M. Agrawal and L. S. Davis. A probabilistic framework for surface reconstruction from multiple
images. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2001. 3
[3] M. Atzmon and Y. Lipman. SAL: Sign agnostic learning of shapes from raw data. In Proc.

IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2020. 17

[4] D. Azinovi´c, R. Martin-Brualla, D. B. Goldman, M. Nießner, and J. Thies. Neural rgb-d surface
reconstruction. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR),
2022. 10

[5] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman. Mip-nerf 360: Un-
bounded anti-aliased neural radiance ﬁelds. IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2022. 2

[6] M. Bleyer, C. Rhemann, and C. Rother. Patchmatch stereo - stereo matching with slanted

support windows. In Proc. of the British Machine Vision Conf. (BMVC), 2011. 3

[7] J. D. Bonet and P. Viola. Poxels: Probabilistic voxelized volume reconstruction. In Proc. of the

IEEE International Conf. on Computer Vision (ICCV), 1999. 3

[8] A. Broadhurst, T. W. Drummond, and R. Cipolla. A probabilistic framework for space carving.

In Proc. of the IEEE International Conf. on Computer Vision (ICCV), 2001. 3

[9] Z. Chen and H. Zhang. Learning implicit ﬁelds for generative shape modeling. Proceedings of

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 1, 3

[10] J. Chibane, T. Alldieck, and G. Pons-Moll. Implicit functions in feature space for 3d shape re-
construction and completion. In IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), 2020. 2, 3, 5

[11] J. M. Coughlan and A. L. Yuille. Manhattan world: Compass direction from a single image by
bayesian inference. In Proc. of the IEEE International Conf. on Computer Vision (ICCV), 1999.
3

[12] B. Curless and M. Levoy. A volumetric method for building complex models from range images.

In ACM Trans. on Graphics, 1996. 7, 8, 9, 23

[13] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Niessner. Scannet: Richly-
annotated 3d reconstructions of indoor scenes. In Proc. IEEE Conf. on Computer Vision and
Pattern Recognition (CVPR), 2017. 3, 7, 18

[14] K. Deng, A. Liu, J.-Y. Zhu, and D. Ramanan. Depth-supervised nerf: Fewer views and faster
training for free. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR),
2022. 23

[15] T. Do, K. Vuong, S. I. Roumeliotis, and H. S. Park. Surface normal estimation of tilted
images via spatial rectiﬁer. In Proc. of the European Conference on Computer Vision, Virtual
Conference, August 23–28 2020. 21

[16] S. Donne and A. Geiger. Learning non-volumetric depth fusion using successive reprojections.

In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2019. 3, 9

[17] A. Eftekhar, A. Sax, J. Malik, and A. Zamir. Omnidata: A scalable pipeline for making
multi-task mid-level vision datasets from 3d scans. In Proc. of the IEEE International Conf. on
Computer Vision (ICCV), 2021. 2, 3, 6, 10, 17, 19, 21

[18] D. Eigen and R. Fergus. Predicting depth, surface normals and semantic labels with a common
multi-scale convolutional architecture. In Proc. of the IEEE International Conf. on Computer
Vision (ICCV), 2015. 2

[19] D. Eigen, C. Puhrsch, and R. Fergus. Depth map prediction from a single image using a
multi-scale deep network. In Advances in Neural Information Processing Systems (NIPS), 2014.
2, 6, 16

[20] A. Gropp, L. Yariv, N. Haim, M. Atzmon, and Y. Lipman. Implicit geometric regularization for
learning shapes. In Proc. of the International Conf. on Machine learning (ICML), 2020. 6
[21] H. Guo, S. Peng, H. Lin, Q. Wang, G. Zhang, H. Bao, and X. Zhou. Neural 3d scene recon-
struction with the manhattan-world assumption. In Proc. IEEE Conf. on Computer Vision and
Pattern Recognition (CVPR), 2022. 2, 3, 7, 8, 9, 18, 22, 26

[22] S. Hadadan, S. Chen, and M. Zwicker. Neural radiosity. ACM Trans. Graph., 2021. 2, 3, 5

11

[23] W. Hartmann, S. Galliani, M. Havlena, L. Van Gool, and K. Schindler. Learned multi-patch
similarity. In Proc. of the IEEE International Conf. on Computer Vision (ICCV), 2017. 3
[24] D. Hoiem, A. Efros, and M. Hebert. Putting objects in perspective. International Journal of

Computer Vision (IJCV), 80:3–15, 2008. 2

[25] D. Hoiem, A. A. Efros, and M. Hebert. Automatic photo pop-up. ACM Trans. on Graphics,

2005. 2

[26] D. Hoiem, A. A. Efros, and M. Hebert. Geometric context from a single image. In Proc. of the

IEEE International Conf. on Computer Vision (ICCV), 2005. 2

[27] D. Hoiem, A. A. Efros, and M. Hebert. Recovering surface layout from an image. International

Journal of Computer Vision (IJCV), 75(1):151–172, October 2007. 2

[28] J. Huang, S.-S. Huang, H. Song, and S.-M. Hu. Di-fusion: Online implicit 3d reconstruction
with deep priors. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR),
2021. 2, 5

[29] P. Huang, K. Matzen, J. Kopf, N. Ahuja, and J. Huang. Deepmvs: Learning multi-view
stereopsis. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 3
[30] A. Jain, M. Tancik, and P. Abbeel. Putting nerf on a diet: Semantically consistent few-shot view
synthesis. In Proc. of the IEEE International Conf. on Computer Vision (ICCV), 2021. 2, 3
[31] Y. Jiang, D. Ji, Z. Han, and M. Zwicker. Sdfdiff: Differentiable rendering of signed dis-
tance ﬁelds for 3d shape optimization. In Proc. IEEE Conf. on Computer Vision and Pattern
Recognition (CVPR), 2020. 2, 4

[32] M. M. Kazhdan and H. Hoppe. Screened poisson surface reconstruction. ACM Trans. on

Graphics, 32(3):29, 2013. 22

[33] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization.

In Proc. of the

International Conf. on Machine learning (ICML), 2015. 6

[34] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun. Tanks and temples: Benchmarking large-scale

scene reconstruction. ACM Trans. on Graphics, 36(4), 2017. 3, 7, 18

[35] K. N. Kutulakos and S. M. Seitz. A theory of shape by space carving. International Journal of

Computer Vision (IJCV), 38(3):199–218, 2000. 3

[36] V. Leroy, J. Franco, and E. Boyer. Shape reconstruction using volume sweeping and learned
photoconsistency. In Proc. of the European Conf. on Computer Vision (ECCV), 2018. 3
[37] B. Li, Y. Huang, Z. Liu, D. Zou, and W. Yu. Structdepth: Leveraging the structural regular-
ities for self-supervised indoor depth estimation. In Proceedings of the IEEE International
Conference on Computer Vision, 2021. 21

[38] L. Liu, J. Gu, K. Z. Lin, T. Chua, and C. Theobalt. Neural sparse voxel ﬁelds. In Advances in

Neural Information Processing Systems (NeurIPS), 2020. 2, 5

[39] S. Liu, Y. Zhang, S. Peng, B. Shi, M. Pollefeys, and Z. Cui. DIST: Rendering deep implicit
signed distance function with differentiable sphere tracing. In Proc. IEEE Conf. on Computer
Vision and Pattern Recognition (CVPR), 2020. 3

[40] W. Luo, A. Schwing, and R. Urtasun. Efﬁcient deep learning for stereo matching. In Proc.

IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2016. 3

[41] J. N. Martel, D. B. Lindell, C. Z. Lin, E. R. Chan, M. Monteiro, and G. Wetzstein. Acorn:
Adaptive coordinate networks for neural scene representation. In ACM Trans. on Graphics,
2021. 3

[42] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger. Occupancy networks:
Learning 3d reconstruction in function space. In Proc. IEEE Conf. on Computer Vision and
Pattern Recognition (CVPR), 2019. 1, 3, 7, 18

[43] S. M. H. Miangoleh, S. Dille, L. Mai, S. Paris, and Y. Aksoy. Boosting monocular depth
estimation models to high-resolution via content-adaptive multi-resolution merging. In CVPR,
2021. 23

[44] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. NeRF:
Representing scenes as neural radiance ﬁelds for view synthesis. In Proc. of the European Conf.
on Computer Vision (ECCV), 2020. 1, 3, 4, 5, 9, 10, 26

[45] T. Müller, A. Evans, C. Schied, and A. Keller.

Instant neural graphics primitives with a

multiresolution hash encoding. ACM Trans. on Graphics, 2022. 2, 3, 5, 6, 7, 16

[46] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. Sajjadi, A. Geiger, and N. Radwan. Regnerf:
Regularizing neural radiance ﬁelds for view synthesis from sparse inputs. In Proc. IEEE Conf.
on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3, 7, 9

12

[47] M. Niemeyer, L. Mescheder, M. Oechsle, and A. Geiger. Occupancy ﬂow: 4d reconstruction
by learning particle dynamics. In Proc. of the IEEE International Conf. on Computer Vision
(ICCV), 2019. 1

[48] M. Niemeyer, L. Mescheder, M. Oechsle, and A. Geiger. Differentiable volumetric rendering:
Learning implicit 3d representations without 3d supervision. In Proc. IEEE Conf. on Computer
Vision and Pattern Recognition (CVPR), 2020. 3

[49] M. Oechsle, S. Peng, and A. Geiger. Unisurf: Unifying neural implicit surfaces and radiance
ﬁelds for multi-view reconstruction. In Proc. of the IEEE International Conf. on Computer
Vision (ICCV), 2021. 1, 2, 3, 5, 7, 9, 10, 22

[50] J. J. Park, P. Florence, J. Straub, R. A. Newcombe, and S. Lovegrove. Deepsdf: Learning
continuous signed distance functions for shape representation. In Proc. IEEE Conf. on Computer
Vision and Pattern Recognition (CVPR), 2019. 1, 3, 4

[51] D. Paschalidou, A. O. Ulusoy, C. Schmitt, L. van Gool, and A. Geiger. Raynet: Learning
volumetric 3d reconstruction with ray potentials. In Proc. IEEE Conf. on Computer Vision and
Pattern Recognition (CVPR), 2018. 3

[52] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin,
N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani,
S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala. Pytorch: An imperative style,
high-performance deep learning library. In Advances in Neural Information Processing Systems
(NIPS), 2019. 6

[53] S. Peng, C. M. Jiang, Y. Liao, M. Niemeyer, M. Pollefeys, and A. Geiger. Shape as points: A
differentiable poisson solver. In Advances in Neural Information Processing Systems (NeurIPS),
2021. 2, 7, 18

[54] S. Peng, M. Niemeyer, L. Mescheder, M. Pollefeys, and A. Geiger. Convolutional occupancy
networks. In Proc. of the European Conf. on Computer Vision (ECCV), 2020. 2, 3, 5, 7, 18
[55] R. Ranftl, A. Bochkovskiy, and V. Koltun. Vision transformers for dense prediction. ArXiv

preprint, 2021. 2

[56] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun. Towards robust monocular
depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Trans. on Pattern
Analysis and Machine Intelligence (PAMI), 2020. 2, 6, 16, 21

[57] G. Riegler, A. O. Ulusoy, H. Bischof, and A. Geiger. OctNetFusion: Learning depth fusion

from data. In Proc. of the International Conf. on 3D Vision (3DV), 2017. 3

[58] B. Roessle, J. T. Barron, B. Mildenhall, P. P. Srinivasan, and M. Nießner. Dense depth priors for
neural radiance ﬁelds from sparse input views. In Proc. IEEE Conf. on Computer Vision and
Pattern Recognition (CVPR), 2021. 3

[59] A. Saxena, S. H. Chung, and A. Y. Ng. Learning depth from single monocular images. In

Advances in Neural Information Processing Systems (NIPS), 2006. 2

[60] A. Saxena, S. H. Chung, and A. Y. Ng. 3-D depth reconstruction from a single still image.

International Journal of Computer Vision (IJCV), 76:53–69, 2008. 2

[61] A. Saxena, M. Sun, and A. Y. Ng. Make3D: learning 3D scene structure from a single still
image. IEEE Trans. on Pattern Analysis and Machine Intelligence (PAMI), 31:824–840, 2009. 2
[62] J. L. Schönberger, E. Zheng, M. Pollefeys, and J.-M. Frahm. Pixelwise view selection for
unstructured multi-view stereo. In Proc. of the European Conf. on Computer Vision (ECCV),
2016. 3, 7, 9, 22, 23, 26

[63] J. L. Schönberger and J.-M. Frahm. Structure-from-motion revisited. In Proc. IEEE Conf. on

Computer Vision and Pattern Recognition (CVPR), 2016. 3

[64] S. Seitz and C. Dyer. Photorealistic scene reconstruction by voxel coloring. In Proc. IEEE Conf.

on Computer Vision and Pattern Recognition (CVPR), 1997. 3

[65] S. M. Seitz, B. Curless, J. Diebel, D. Scharstein, and R. Szeliski. A comparison and evaluation
of multi-view stereo reconstruction algorithms. In Proc. IEEE Conf. on Computer Vision and
Pattern Recognition (CVPR), 2006. 3

[66] V. Sitzmann, M. Zollhöfer, and G. Wetzstein. Scene representation networks: Continuous
3d-structure-aware neural scene representations. In Advances in Neural Information Processing
Systems (NIPS), 2019. 1

[67] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren,
S. Verma, A. Clarkson, M. Yan, B. Budge, Y. Yan, X. Pan, J. Yon, Y. Zou, K. Leon, N. Carter,
J. Briales, T. Gillingham, E. Mueggler, L. Pesqueira, M. Savva, D. Batra, H. M. Strasdat, R. D.

13

Nardi, M. Goesele, S. Lovegrove, and R. Newcombe. The Replica dataset: A digital replica of
indoor spaces. arXiv.org, 1906.05797, 2019. 3, 7, 18

[68] E. Sucar, S. Liu, J. Ortiz, and A. Davison. iMAP: Implicit mapping and positioning in real-time.

In Proc. of the IEEE International Conf. on Computer Vision (ICCV), 2021. 7, 18

[69] T. Takikawa, J. Litalien, K. Yin, K. Kreis, C. Loop, D. Nowrouzezahrai, A. Jacobson,
M. McGuire, and S. Fidler. Neural geometric level of detail: Real-time rendering with implicit
3D shapes. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2021. 2,
3, 5

[70] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ra-
mamoorthi, J. Barron, and R. Ng. Fourier features let networks learn high frequency functions
in low dimensional domains. In Advances in Neural Information Processing Systems (NeurIPS),
2020. 4, 5

[71] M. Teschner, B. Heidelberger, M. Müller, D. Pomeranets, and M. Gross. Optimized spatial
hashing for collision detection of deformable objects. In Proceedings of VMV’03, Munich,
Germany, 2003. 16

[72] S. Tulsiani, T. Zhou, A. A. Efros, and J. Malik. Multi-view supervision for single-view
reconstruction via differentiable ray consistency. In Proc. IEEE Conf. on Computer Vision and
Pattern Recognition (CVPR), 2017. 3

[73] A. O. Ulusoy, A. Geiger, and M. J. Black. Towards probabilistic volumetric reconstruction

using ray potentials. In Proc. of the International Conf. on 3D Vision (3DV), 2015. 3

[74] B. Ummenhofer, H. Zhou, J. Uhrig, N. Mayer, E. Ilg, A. Dosovitskiy, and T. Brox. Demon:
Depth and motion network for learning monocular stereo. In Proc. IEEE Conf. on Computer
Vision and Pattern Recognition (CVPR), 2017. 3

[75] J. Wang, P. Wang, X. Long, C. Theobalt, T. Komura, L. Liu, and W. Wang. Neuris: Neural

reconstruction of indoor scenes using normal priors. In ECCV, 2022. 3, 8, 9, 21, 22

[76] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang. Neus: Learning neural implicit
surfaces by volume rendering for multi-view reconstruction. In Advances in Neural Information
Processing Systems (NeurIPS), 2021. 1, 2, 3, 5, 7, 9, 22

[77] S. Wang, M. Mihajlovic, Q. Ma, A. Geiger, and S. Tang. Metaavatar: Learning animatable
clothed human models from few depth images. In Advances in Neural Information Processing
Systems (NeurIPS), 2021. 9, 10

[78] Y. Xie, T. Takikawa, S. Saito, O. Litany, S. Yan, N. Khan, F. Tombari, J. Tompkin, V. Sitzmann,

and S. Sridhar. Neural ﬁelds in visual computing and beyond. In EUROGRAPHICS, 2022. 3
[79] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan. Mvsnet: Depth inference for unstructured

multi-view stereo. Proc. of the European Conf. on Computer Vision (ECCV), 2018. 3

[80] Y. Yao, Z. Luo, S. Li, T. Shen, T. Fang, and L. Quan. Recurrent mvsnet for high-resolution multi-
view stereo depth inference. Proc. IEEE Conf. on Computer Vision and Pattern Recognition
(CVPR), 2019. 3

[81] L. Yariv, J. Gu, Y. Kasten, and Y. Lipman. Volume rendering of neural implicit surfaces. In
Advances in Neural Information Processing Systems (NeurIPS), 2021. 1, 2, 3, 5, 6, 7, 9, 10, 22,
23, 27, 28, 34, 35

[82] L. Yariv, Y. Kasten, D. Moran, M. Galun, M. Atzmon, B. Ronen, and Y. Lipman. Multiview
neural surface reconstruction by disentangling geometry and appearance. In Advances in Neural
Information Processing Systems (NIPS), 2020. 1, 2, 3, 5, 9

[83] W. Yin, J. Zhang, O. Wang, S. Niklaus, L. Mai, S. Chen, and C. Shen. Learning to recover 3d
scene shape from a single image. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (CVPR), 2021.
21

[84] A. Yu, V. Ye, M. Tancik, and A. Kanazawa. pixelNeRF: Neural radiance ﬁelds from one or few

images. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2021. 7

[85] Z. Yu and S. Gao. Fast-mvsnet: Sparse-to-dense multi-view stereo with learned propagation

and gauss-newton reﬁnement. In CVPR, 2020. 3

[86] Z. Yu, L. Jin, and S. Gao. P2net: Patch-match and plane-regularization for unsupervised indoor

depth estimation. In ECCV, 2020. 21

[87] Z. Yu, J. Zheng, D. Lian, Z. Zhou, and S. Gao. Single-image piece-wise planar 3d reconstruction

via associative embedding. In CVPR, pages 1029–1037, 2019. 10

[88] S. Zagoruyko and N. Komodakis. Learning to compare image patches via convolutional neural
networks. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2015. 3

14

[89] J. Zhang, Y. Yao, S. Li, Z. Luo, and T. Fang. Visibility-aware multi-view stereo network. In

Proc. of the British Machine Vision Conf. (BMVC), 2020. 22, 31

[90] K. Zhang, G. Riegler, N. Snavely, and V. Koltun. Nerf++: Analyzing and improving neural

radiance ﬁelds. arXiv:2010.07492, 2020. 2

[91] X. Zhang, P. P. Srinivasan, B. Deng, P. Debevec, W. T. Freeman, and J. T. Barron. NeRFactor:
Neural Factorization of Shape and Reﬂectance Under an Unknown Illumination. arXiv preprint
arXiv:2106.01970, 2021. 2

[92] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys. Nice-slam:
Neural implicit scalable encoding for slam. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3, 5, 7, 10, 18

Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes]
(c) Did you discuss any potential negative societal impacts of your work? [Yes] We discuss

potential negative societal impacts in our supplementary material.

(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [N/A]
(b) Did you include complete proofs of all theoretical results? [N/A]

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experi-
mental results (either in the supplemental material or as a URL)? [Yes] Code and data
are released.

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they

were chosen)? [Yes]

(c) Did you report error bars (e.g., with respect to the random seed after running experi-

ments multiple times)? [No]

(d) Did you include the total amount of compute and the type of resources used (e.g.,
type of GPUs, internal cluster, or cloud provider)? [Yes] We describe details of our
computational resources in supplementary material.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]
(b) Did you mention the license of the assets? [N/A]
(c) Did you include any new assets either in the supplemental material or as a URL? [N/A]

(d) Did you discuss whether and how consent was obtained from people whose data you’re

using/curating? [N/A]

(e) Did you discuss whether the data you are using/curating contains personally identiﬁable

information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review

Board (IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount

spent on participant compensation? [N/A]

15

Supplementary Material for
MonoSDF: Exploring Monocular Geometric Cues
for Neural Implicit Surface Reconstruction

In this supplementary document, we ﬁrst discuss architectural and implementation details in Sec-
tion A. Next, we provide additional ablation studies of our monocular geometric cues for four
different scene representations in Section B and report additional quantitative and qualitative results
in Section C. Finally, we discuss potential negative impact of this work in Section D.

A Implementation Details

In this section, we ﬁrst present an overview of 4 different architectures for neural implicit scene
representations and details of Multi-Res. Grids in Section A.1 and provide details of the depth loss
computation in Section A.2. Next, we describe additional details regarding our parameterizations and
optimization in Section A.3 and discuss evaluation metrics in Section A.4.

A.1 Architectures

In the main paper, we investigate four different architectures as our scene representation: Dense SDF
Grid, Single MLP, Single-Res. Grid, and Multi-Res. Grids . See Fig. 5 for an overview over the
architectures. In the following, we provide details for Multi-Res. Feature Grids.
Multi-Res. Grids. Following Instant-NGP [45], we use L levels of feature grids with resolutions
sampled in geometric space to combine features at different frequencies:
(cid:18) ln Rmax − ln Rmin
L − 1

Rl := (cid:98)Rminbl(cid:99)

b := exp

(16)

(cid:19)

,

where Rmin, Rmax are the coarsest and ﬁnest resolutions, respectively. As the total number of grid
cells grows cubically, we use a ﬁxed number of parameters to store the feature grids and use a spatial
hash function to index the feature vector at ﬁner levels. More speciﬁcally, each grid contains up
to T feature vectors with dimensionality F . At the coarse level where R3
l ≤ T , the feature grid is
stored densely. At the ﬁner level where R3
l > T , a spatial hash function [71] is used to index the
corresponding feature vector:

h(x) =

(cid:33)

xiπi

mod T ,

(cid:32) 3

(cid:77)

i=1

(17)

where (cid:76) is the bit-wise XOR operation and πi are unique, large prime numbers. We use the default
values Rmin = 16, Rmax = 2048, L = 16, F = 2, and T = 219 similar to [45] in all experiments.

A.2 Depth Consistency Loss

We enforce consistency between our rendered expected depth ˆD and the monocular depth ¯D with a
scale invariant loss function:

Ldepth =

(cid:88)

r∈R

(cid:107)(w ˆD(r) + q) − ¯D(r)(cid:107)

2

,

(18)

where w and q are the scale and shift used to align ˆD and ¯D since ¯D is given only up to scale.
Speciﬁcally, we solve w and q with a least-squares criterion [19, 56]:

(w, q) = arg min

(cid:88)

(cid:16)

w ˆD(r) + q − ¯D(r)

(cid:17)2

.

(19)

w,q

r∈R

16

Figure 5: Architectures. We show an overview over four different scene representations considered
in this paper.

w and q can be efﬁciently computed as follows: Let h = (w, q)T and dr = ( ˆD(r), 1)T , then Eq. (19)
can be rewrite as:

hopt = arg min

h

(cid:88)

r∈R

(cid:0)dT

r h − ¯D(r)(cid:1)2

.

which has the closed-form solution:

h =

(cid:32)

(cid:88)

r

drdT
r

(cid:33)−1 (cid:32)

(cid:33)

dr ¯D(r)

.

(cid:88)

r

(20)

(21)

Note that we estimate w and q individually at each iteration for a batch of randomly sampled rays
within a single image because depth maps predicted by the monocular depth predictor can differ in
scale and shift and the underlying scene geometry changes at each iteration.

A.3 Additional Details

For our single MLP architecture, we use an 8-layer MLP with hidden dimension 256. We use a
two-layer MLP with hidden dimension 256 for the SDF prediction for both, Single-Res. Grid and
Multi-Res. Grids. We implement the color network with a two-layer MLP with hidden dimension
256 and use it for all architectures. We use Softplus activation for geometric network and use ReLU
activation for the color network. We explicitly initialize the SDF grid with a sphere and use the
geometric initialization from [3] for other architectures. For obtaining monocular cues, we ﬁrst resize
each image and center crop it to 384 × 384, which we then feed as input to the pretrained Omnidata
model [17]. The output depth and normal maps have the same resolution of 384 × 384. As a result,
we use the same resolution for RGB images, depth cues and normal cues and adjust camera intrinsics
accordingly for all experiments. We optimize our model for 200k iterations which takes about 6 hours
and 11 hours for our Multi-Res. Grids and MLP, respectively, on a single NVIDIA RTX3090 GPU.

17

A.4 Evaluation Metrics

For the DTU dataset [1], we follow the ofﬁcial evaluation protocol and report the reconstruction
quality with: Accuracy, Completeness and Chamfer Distance. Accuracy measures how close the
reconstructed points are to the ground truth and is deﬁned as the mean distance of the reconstructed
points to the ground truth. Completeness measures to what extent the ground truth points are recovered
and is deﬁned as the mean distance of the ground truth points to the reconstructed points. Chamfer
Distance is the mean of Accuracy and Completeness. It measures the overall reconstruction quality.
For efﬁciency, we use the Python script3 to compute these evaluation metrics.

Metric

Acc

Comp

Chamfer

Precision

Recall

F-score

Deﬁnition

(cid:18)

mean
p∈P

min
p∗∈P ∗
(cid:18)

(cid:19)

||p − p∗||1

(cid:19)

mean
p∗∈P ∗

min
p∈P

||p − p∗||1

(cid:18)

mean
p∈P

min
p∗∈P ∗
(cid:18)

Acc+Comp
2

||p − p∗||1 < 0.05

(cid:19)

(cid:19)

mean
p∗∈P ∗

min
p∈P

||p − p∗||1 < 0.05

Normal-Acc

(cid:0)nT

p np∗

mean
p∈P

Normal-Comp

mean
p∗∈P ∗

(cid:0)nT

p np∗

2·Precision·Recall
Precision+Recall
(cid:1) s.t. p∗ = argmin
p∗∈P ∗
(cid:1) s.t. p = argmin

p∈P

||p − p∗||1

||p − p∗||1

Normal-Consistency

Normal-Acc+Normal-Comp
2

Table 6: Evaluation Metrics. We show the evaluation metrics with their deﬁnitions that we use to
measure reconstruction quality. P and P ∗ are the point clouds sampled from the predicted and the
ground truth mesh. np is the normal vector at point p.

For Replica [67] and ScanNet [13], we report Accuracy, Completeness, Chamfer Distance, Precision,
Recall, and F-score with a threshold of 5cm following [21, 68, 92]. We further report Normal
Consistency for the Replica dataset following [21, 42, 53, 54, 68, 92] as near-perfect ground truth is
available. These metrics are deﬁned in Table 6.

For the Tanks and Temples dataset [34], we submit our reconstruction results to the ofﬁcial evaluation
server4 and report the provided F-score.

B Ablation

In this section, we ﬁrst conduct several ablation studies to verify the effectiveness of our method,
including using geometric cues with different scene representations in Section B.1, different archi-
tecture conﬁgurations in Section B.2, different number of input views in Section B.3, different cues
predictors in Section B.4. Next, we analyze the optimization time of our framework in Section B.5.

18

Test Split

Train Split

Normal C.↑ Chamfer-L1 ↓ F-score ↑ Normal C.↑ Chamfer-L1 ↓ F-score ↑

No Cues

Dense SDF Only Depth

Grid

MLP

Only Normal
Both Cues
No Cues
Only Depth
Only Normal
Both Cues
No Cues

Single-Res. Only Depth

Grids

Only Normal
Both Cues
No Cues

Multi-Res. Only Depth

Grids

Only Normal
Both Cues

57.30
71.81
73.95
76.47
86.48
90.56
91.35
92.11
86.41
90.50
89.60
90.59
87.95
90.87
89.90
90.93

26.68
12.60
13.62
11.39
6.75
4.26
3.19
2.94
6.28
3.94
4.07
3.56
5.03
3.75
3.61
3.23

15.50
30.09
33.34
37.27
66.88
76.42
85.84
86.18
64.22
78.42
76.47
83.34
78.38
80.32
81.28
85.91

60.86
73.15
77.80
80.05
86.69
91.80
92.85
93.86
86.54
91.3
91.87
91.87
87.15
91.25
91.11
91.41

17.34
13.09
11.30
10.09
7.48
3.59
4.23
2.63
6.63
3.29
3.13
2.98
5.83
3.41
3.59
3.14

26.34
30.30
42.45
41.57
63.24
85.67
85.58
92.12
67.26
86.34
85.96
88.23
72.13
87.04
84.02
86.87

Table 7: Ablation of Monocular Geometric Cues on Replica. Our monocular geometric cues
signiﬁcantly improve reconstruction quality across all architectures.

10 Views

20 Views

30 Views

40 Views

GT Views

28.5

44.6

61.6

61.4

P
L
M

s
e
u
C

/

w

75.4

89.8

90.9

92.8

Figure 6: Ablation of Different Number of Input Views on the Replica Dataset. We show F-score
under each image. We observe that using more input views for training improves reconstruction
quality. Further, adding monocular geometric cues improves reconstruction quality. When using only
10 input views, the MLP fails to reconstruct reasonable results while using monocular geometric cues
signiﬁcantly improves results.

B.1 Ablation of Different Cues

To evaluate the effectiveness of our monocular geometric cues for different scene representations, we
conduct ablation studies on the Replica dataset with our four different scene representations. Note
that as the Replica dataset is part of the training set of Omnidata (making up 0.46% of the entire
training data) [17], we split the evaluation into the train/test split of Omnidata [17].

As shown in Table 2 and Fig. 8, our geometric cues improve reconstruction quality signiﬁcantly
independent of the underlying scene representations. We observe that using both, depth cues and
normal cues, leads to the best results, indicating the complementary nature of the different cues. We
further observe that the reconstruction quality as well as the improvements from adding geometric
cues are similar for the train and test split of Omnidata, showing that the monocular predictor did not
overﬁt to the training data.

19

Figure 7: Optimization Processes Using Different Architecture Conﬁgurations. Using monocu-
lar geometric cues improves reconstruction quality and convergence speed independent of the network
conﬁgurations.

Model conﬁguration

Num. Params

MLP (2 layers)
MLP (4 layers)
MLP (8 layers)
MLP (12 layers)
Multi-res. Feature Grids (hash table size 213)
Multi-res. Feature Grids (hash table size 215)
Multi-res. Feature Grids (hash table size 217)
Multi-res. Feature Grids (hash table size 219)

0.15M
0.26M
0.53M
0.8M

0.41M
1.11M
3.67M
12.67M

Table 8: Number of Learnable Parameters Using Different Architecture Conﬁgurations.

B.2 Ablation of Different Architecture Conﬁgurations

In order to evaluate the performance with different model capacities, we consider MLPs with a
different number of layers and Multi-res. Feature Grids with different sizes of the hash table. We list
the number of learnable parameters using different architecture conﬁgurations in the Table 8, and
show their performance over the optimization processes in Fig. 7. Our experiments show that using
monocular geometric cues improves reconstruction quality and convergence speed independent of
the network conﬁguration.

B.3 Ablation of Different Numbers of Input Views

We ran experiments with a different number of input images and monocular geometric cues. As
shown in Fig. 6, adding the monocular geometric cues leads to consistent improvements across
different numbers of input views.

20

52040Iterations (×103)0.20.40.60.81F-scoreMLP 2 layerMLP 4 layerMLP 8 layerMLP 12 layerMLP 2 layer (w/ Cues)MLP 4 layer (w/ Cues)MLP 8 layer (w/ Cues)MLP 12 layer (w/ Cues)Grids 2^13Grids 2^15Grids 2^17Grids 2^19Grids 2^13 (w/ Cues)Grids 2^15 (w/ Cues)Grids 2^17 (w/ Cues)Grids 2^19 (w/ Cues)Method

F-score

MLP
w/ MiDaS [56]
w/ LeReS [83]
w/ Omnidata [17]

64.2
68.6
72.6
86.7

Method

F-score

Method

F-score

MLP
w/ Tilted [15]
w/ Omnidata [17]

64.2
45.6
92.2

MLP
w/ Self-supervised [37, 86]
w/ Omnidata [17]

64.2
45.6
86.7

(a) Different Depth

(b) Different Normal

(c) Self-supervised Depth

Table 9: Ablation of Different Monocular Cues Predictors. a.) Adding monocular depth improves
performance over a single MLP without cues. Unsurprisingly, better depth predictors lead to better
performance, with the state-of-the-art Omnidata model giving the best results. b.) Adding monocular
normal improve the results. Similarly, using normals predicted by the state-of-the-art Omnidata
model leads to the best performance. c.) Using self-supervised depth estimator degrades performance.
We hypothesize that this is due to the weaker performance of the self-supervised model which is
also trained with an RGB loss and hence suffers from the under-constrained problem of recovering
geometry from multi-view images.

B.4 Ablation of Different Monocular Cues Predictors

To further analyze the robustness of our approach to monocular geometric cues of different levels
of quality, we further tested our model with different supervised depth predictors [56, 83], normal
predictors [15], and self-supervised depth predictors [37, 86]. The result is shown in Table 9. We
found that using the state-of-the-art Omnidata model leads to the best results, indicating that the
development of better geometric cues will further improve the performance of our approach.

B.5 Optimization Time

Adding monocular geometric cues to the optimization introduces a small overhead to our overall
optimization pipeline. First, predicting these cues with a pretrained Omnidata model is very efﬁcient
(36 FPS with an NVIDIA RTX3090 GPU). For example, it takes less than 26 seconds to predict both
depth maps and normal maps for 464 images for one of the ScanNet scene. Note that this only needs
to be done once and that we measure FPS with a batch size of one; using a larger batch size will result
in a speed up. Second, we volume render depth and normals during optimization in order to apply a
loss against these monocular cues. This overhead is also small and can be neglected since the most
expensive part wrt. compute is the inference of the network. For our MLP variant, the additional ﬂops
for volume rendering depth and normal is only 0.0002% of the MLP inference time. While adding
monocular geometric cues introduce a small overhead, the improvements in terms of reconstruction
quality and converge speed are signiﬁcant. As shown in Table 2 (b) in the main paper, with only 5k
iterations, our Multi-Res. Grids representation with cues performs better than the converged models
without geometric cues, which implies a 40× speed up (5k vs. 200k).

C Additional Results

In this section, we provide more qualitative and quantitative results for three datasets: ScanNet
( Section C.1), Tanks and Temples ( Section C.2), and DTU ( Section C.4).

C.1 ScanNet

We report quantitative results with all metrics for ScanNet in Table 10 and show more visualizations
in Fig. 9. Compared to state-of-the-art methods, our approach with MLP architecture produces
signiﬁcantly better reconstructions both visually as well as quantitatively. It’s worth noting that we
perform better than concurrent work [75] even though they have some ﬁltering mechanism.

21

Acc↓

Comp↓ Chamfer-L1 ↓

Prec↑ Recall↑

F-score↑

COLMAP [62]
UNISURF [49]
NeuS [76]
VolSDF [81]
Manhattan-SDF [21]
NeuRIS [75]

Ours (Multi-Res. Grids)
Ours (MLP)

0.047
0.554
0.179
0.414
0.072
0.050

0.072
0.035

0.235
0.164
0.208
0.120
0.068
0.049

0.057
0.048

0.141
0.359
0.194
0.267
0.070
0.050

0.064
0.042

0.711
0.212
0.313
0.321
0.621
0.717

0.660
0.799

0.441
0.362
0.275
0.394
0.586
0.669

0.601
0.681

0.537
0.267
0.291
0.346
0.602
0.692

0.626
0.733

Table 10: Scene-level 3D Reconstruction on ScanNet. We report reconstruction results for our
methods and baselines on ScanNet (baselines from [21]). We ﬁnd that our approaches outperform
previous state-of-the-art, highlighting the effectiveness of the use of monocular geometric priors. As
ScanNet’s RGB images contain motion blur and the camera poses are partially noisy, we further
observe that the MLP architecture is more robust to this noise and achieves the best results. It’s
worth noting that we perform better than concurrent work [75] even though they have some ﬁltering
mechanism.

Grid Grid w/ cues MLP [81] MLP w/ cues

Auditorium 1.36
Ballroom
2.67
Courtroom 7.84
4.12
Museum

mean

4.00

3.17
3.70
13.75
5.68

6.58

1.60
2.04
8.03
2.96

3.66

3.09
2.47
10.00
5.10

5.165

Table 11: Evaluation Results on the Tanks and Temples Dataset Advanced Set. We evaluate the
reconstructed meshes using the ofﬁcial server and report the F-score with 10mm. Our monocular
geometric cues improve the reconstruction quality for all scenes.

C.2 Tanks and Temples

We show quantitative results for Tanks and Temples in Table 11. Qualitative comparisons of with or
without monocular cues of our MLP variant are shown in Fig. 10 and Fig. 11. Fig. 12 and Fig. 13
show qualitative comparison of our Mulit-Res. Grids. Our monocular geometric cues signiﬁcantly
improve the reconstruction quality.

We further show an additional comparison against state-of-the-art MVS methods in Fig. 14. We use a
pretrained Vis-MVSNet [89] to predict depth maps for the input images and fuse them to point clouds
follow the ofﬁcial code.5 Next, we use Meshlab’s screened Poisson reconstruction [32] to reconstruct
a mesh from point clouds with default parameters. We observe that our reconstructions are more
complete which is useful for many applications. Further, reconstructing a mesh from point clouds
involves lossy post-processing, leading to ﬂoating artifacts and bloated areas in less-observed areas.

C.3 Preliminary Results of Using High-resolution Monocular Cues

In the main paper, we center-crop each image and resize it to 384 × 384. Then, we use a pretrained
Omnidata model to predict depth maps and normal maps which are also of size 384 × 384. While
we have shown that training at a resolution of 384 × 384 produces impressive results, we believe
that exploring different ways to generate and integrate higher resolution cues could further improve
reconstruction quality. Here, we provide a proof-of-concept experiment for generating higher
resolution monocular cues and integrating them into our model. We use a divide-and-conquer
method for generating high-resolution cues. First, we partition a high-resolution image to multiple
overlapping sub-images, and we predict monocular depth and normal for each sub-image. Next, we
merge these predictions. We use Eq. 21 to align the depth maps and solve the rotation for the normal

3https://github.com/jzhangbs/DTUeval-python
4https://www.tanksandtemples.org/
5Available at https://github.com/jzhangbs/Vis-MVSNet

22

TSDF [12] COLMAP RealityCapture MLP [81] MLP Multi-Res. Multi-Res.

w/ cues Grids Grids w/ cues

scan24
scan37
scan40
scan55
scan63
scan65
scan69
scan83
scan97
scan105
scan106
scan110
scan114
scan118
scan122

mean

5.01
5.28
5.09
4.63
5.03
4.50
4.55
4.88
6.22
3.89
5.67
3.80
4.67
4.51
4.35

4.80

4.45
4.67
2.51
1.90
2.81
2.92
2.12
2.05
2.93
2.05
2.01
N/A
1.10
2.72
1.64

2.56

4.19
3.85
2.26
2.49
3.49
3.97
1.91
2.49
2.37
2.27
2.90
4.60
1.38
2.57
1.76

2.84

5.24
5.09
3.99
1.42
5.10
4.33
5.36
3.15
5.78
2.07
2.79
5.73
1.20
5.64
6.20

4.21

3.47
3.61
2.10
1.05
2.37
1.38
1.41
1.85
1.74
1.10
1.46
2.28
1.25
1.44
1.45

1.86

6.46
8.30
7.03
5.87
6.92
3.09
5.34
6.03
6.93
6.01
6.14
7.62
6.27
7.59
6.47

6.47

5.24
6.37
2.52
1.95
6.64
2.05
4.25
1.81
5.27
2.54
3.85
3.89
1.90
3.12
3.84

3.68

Table 12: Evaluation Results on the DTU Dataset with 3 Input Views. Note the COLMAP fails
on scan110 so we take the average over the remaining 14 scenes. We ﬁnd that without geometric
cues, neither Grids nor MLP works well with only 3 input views. When incorporating the monocular
geometric cues, the results for both representations are signiﬁcantly improved. Interestingly, the
grid-based representations perform inferior to a single MLP as they are updated only locally and do
not have an inductive smoothness bias compared to a monolithic MLP representation.

maps. An example of the resulting high-resolution monocular cues is shown in Fig. 15. We found
that our high-resolution cues contain more ﬁne details compared to low-resolution cues. Note that
using other methods for generating high-resolution depth maps is also possible, e.g., [43]. We then
use the high-resolution cues to train our model, and the results are shown in Fig. 16. We observe
signiﬁcant improvements when using high-resolution monocular cues.

C.4 DTU

Geometry. We show per-scene quantitative results on the DTU dataset with 3 input-views in Table 12
and more qualitative results in Fig. 17. We ﬁnd that without the monocular geometric cues, both
MLP and Multi-Res. Grids fail to produce satisfying reconstructions, while with our monocular cues,
both methods are improved and are able to reconstruct high-quality meshes. We further show more
visualizations on the DTU dataset using all input views in Fig. 19. Compared to state-of-the-art
methods, our approach with multi-resolution feature grids produces more accurate reconstructions.
Novel View Synthesis. We further compare our novel view synthesis
results on the DTU dataset with three input views. As shown in Table 13
and Fig. 18, using monocular geometric cues improves novel view synthesis
results signiﬁcantly.
Weight Annealing. As the monocular depth and normal predictor is not
perfect, we exponentially anneal the loss weight for the monocular depth
consistency and normal consistency loss, λ2 and λ3, to 0 during the ﬁrst
200 epochs of optimization. Qualitative comparison in Fig. 20 veriﬁes the
importance of weight annealing.
Failure cases. We show a failure case on DTU with 3 input views in Fig. 21. The reconstructed
mesh duplicates the object in front of each camera frustum. One reason is that the monocular depth
cues that we use are only up to scale so they do not guarantee multi-view consistency. Therefore,
the optimization is still underconstrained since the input RGB images and monocular cues can
be explained by individual objects in front of the image plane. One possible solution would be
incorporating explicit multi-view constraints such as using sparse point clouds from COLMAP [62]
as an additional supervision [14].

Table 13: Novel view
synthesis results on
DTU (3 Views).

17.65
MLP [81]
MLP w/ cues 23.64

PSNR

23

D Societal Impact

Our method can faithfully reconstruct a 3D scene which can be used for application ranging from
virtual reality to robotics. However, it can also have potential negative societal impact. First, our
method relies on a general purpose monocular geometric predictor that needs to be trained on large
amounts of data and with large computational resources, which potentially has a negative impact
on global climate change. Second, accurate reconstruction of a scene may raise privacy concerns
that need to be addressed carefully. Finally, accurate geometry reconstructed by our method can
potentially be used for malicious purposes.

24

Dense SDF Grid

+ Depth

+ Normal

+ Both

MLP

+ Depth

+ Normal

+ Both

Single-Res. Grid

+ Depth

+ Normal

+ Both

Multi-Res. Grids

+ Depth

+ Normal

+ Both

Figure 8: Ablation of Monocular Geometric Cues on the Replica Dataset. Monocular geometric
cues signiﬁcantly improve reconstruction quality for all architectures. With monocular depth cues, the
recovered geometry contains more details and a better overall structure. Similarly, with our normal
cues, missing details are added and the results become smoother. Using both cues leads to the best
performance. Zoom in for details.

25

COLMAP [62]

VolSDF [44]

Manhattan-SDF [21]

Ours (MLP)

Ground Truth

Figure 9: Qualitative Comparison on ScanNet. We show different views for each scene. Our
method leads to better results containing smooth surfaces and detailed reconstructions compared
against state-of-the-art neural implicit methods.

26

MLP [81]

MLP w/ cues

GT view

Figure 10: Qualitative Comparison on Tanks & Temples. We use a single MLP as the scene
geometry representation [81] and compare the reconstruction when using monocular cues or not on
Auditorium and Ballroom.

27

MLP [81]

MLP w/ cues

GT view

Figure 11: Qualitative Comparison on Tanks & Temples Dataset. We use a single MLP as the
scene geometry representation [81] and compare the reconstruction quality when using monocular
cues or not on Courtroom and Museum.

28

Multi-Res. Grids

Multi-Res. Grids w/ cues

GT view

Figure 12: Qualitative Comparison on Tanks & Temples. We use Multi-Res. Grids as the scene
geometry representation and compare the reconstruction when using monocular cues or not on
Auditorium and Ballroom.

29

Multi-Res. Grids

Multi-Res. Grids w/ cues

GT view

Figure 13: Qualitative Comparison on Tanks & Temples. We use Multi-Res. Grids as the scene
geometry representation and compare the reconstruction when using monocular cues or not on
Courtroom and Museum.

30

VisMVSNet [89]

Ours (MLP)

Ours (Multi-Res. Grids)

GT view

Figure 14: Qualitative Comparison on Tanks & Temples.

31

(a) RGB Image.

(b) Low Resolution Depth Map.

(c) High Resolution Depth Map.

(d) Low Resolution Normal Map.

(e) High Resolution Normal Map.

Figure 15: Visual Comparison of Different Resolution Monocular Cues.
32

Low Resolution Cues

High Resolution Cues

GT view

Figure 16: Qualitative Comparison of Low Resolution Cues and High Resolution cues on Tanks
& Temples. We use Multi-Res. Grids as the scene geometry representation and compare the
reconstruction when using different resolution of monocular cues.

33

TSDF-Fusion RealityCapture MLP [81]

MLP
w/ cues

Multi-Res.
Grids

Multi-Res.
Grids w/ cues

GT View

Figure 17: Qualitative Comparison on the DTU Dataset with 3 Input Views. Adding monocular
geometric cues improves 3D reconstruction quality for both MLP and Multi-Res. Grids. We show a
failure case on the last row.

34

MLP [81]

MLP w/cues

GT View

MLP [81]

MLP w/cues

GT View

Figure 18: Qualitative Comparison of Novel View Synthesis on the DTU Dataset with 3 Input
Views. Adding monocular geometric cues improves novel view synthesis quality.

35

Figure 19: Qualitative Comparison on DTU Dataset with all input views. Our approach with
MLP achieves similar results with previous method, while our method with Multi-Res. Fea. Grids
reconstruct more detailed surface.

36

Without Weight Annealing

With Weight Annealing

GT View

Figure 20: Ablation of Weight Annealing on the DTU Dataset with 3 Input Views. Using weight
schedule improves reconstruction quality.

Input View 1

Input View 2

Input View 3

Ours

Figure 21: Failure Case on DTU Dataset with 3 Input Views. The reconstructed mesh duplicate
the object in front of each camera frustum.

37

