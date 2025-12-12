# CHANGELOG



## v0.7.0 (2025-12-12)

### Feature

* feat: add export of best threshold for maximum F1 score for use in pseudo-labeling workflows ([`86887f6`](https://github.com/mbari-org/vitstrain/commit/86887f6c565f11dd155bfb74972e5a693f9998ea))


## v0.6.0 (2025-10-09)

### Feature

* feat: Merge pull request #12 from mbari-org/copilot/add-exclude-label-argument

[WIP] Add argument to exclude one or more labels ([`4abd200`](https://github.com/mbari-org/vitstrain/commit/4abd2005de57865f34e1400a609bc27d0b1e620a))

* feat: add --exclude-labels argument to exclude specific labels from training

Co-authored-by: danellecline &lt;1424813+danellecline@users.noreply.github.com&gt; ([`49a32f6`](https://github.com/mbari-org/vitstrain/commit/49a32f692773bbcae6c1f07cefececcf59018cef))

### Unknown

* Initial plan ([`c37b513`](https://github.com/mbari-org/vitstrain/commit/c37b5139efa529b7a64c399667ab0b3025a11d83))


## v0.5.0 (2025-10-01)

### Feature

* feat: merge pull request #10 from mbari-org/cropovlp add support for rare classes in dataset splitting

feat: add support for rare classes in dataset splitting ([`ccfb2d3`](https://github.com/mbari-org/vitstrain/commit/ccfb2d31ce6cff68d6a778816ddff304193732f6))

### Unknown

* Update src/data_utils.py

Co-authored-by: Copilot &lt;175728472+Copilot@users.noreply.github.com&gt; ([`f9e4f6e`](https://github.com/mbari-org/vitstrain/commit/f9e4f6e8dd7d7472375e43787e372a988da88c05))

* bump to 10 ([`6b6ba12`](https://github.com/mbari-org/vitstrain/commit/6b6ba1237da97e15573b50e9b3a8dd61ac2cb101))

* added crop augmentation ([`b6ab657`](https://github.com/mbari-org/vitstrain/commit/b6ab65742af8e9aa09a0b50675c53ffe3410ae66))


## v0.4.1 (2025-09-10)

### Fix

* fix: add missing data_utils.py ([`7984aa2`](https://github.com/mbari-org/vitstrain/commit/7984aa2d25219995446bb19161fca82e8fdc13ff))


## v0.4.0 (2025-09-05)

### Feature

* feat: merge pull request #6 from mbari-org/prcurve

Add multi-class PR curve generation ([`285edf3`](https://github.com/mbari-org/vitstrain/commit/285edf3f1d5a816f2e498595a98d80746bad1f88))

### Fix

* fix: add missing CHANGELOG.md ([`d84655e`](https://github.com/mbari-org/vitstrain/commit/d84655e3cc95bd06221721cfd03b3236e4c3ea47))

### Unknown

* Update README.md

Co-authored-by: Copilot &lt;175728472+Copilot@users.noreply.github.com&gt; ([`f906c08`](https://github.com/mbari-org/vitstrain/commit/f906c08ed900a64d179f846327ae37249d3947c0))

* correct path and arg for raw data ([`6cf19e4`](https://github.com/mbari-org/vitstrain/commit/6cf19e4982e5b31610ab48c10fc190c36af8ceda))

* added precommit and updated semantic release ([`24c363a`](https://github.com/mbari-org/vitstrain/commit/24c363a69c5a0e1bab81b512a259b16145859bb3))

* updated documentation to reflect dog/cat example, link to internal doc, and revised tar.gz to unpack correctly per docs ([`1b9cb78`](https://github.com/mbari-org/vitstrain/commit/1b9cb782549fe25b168ad59325516724374c9e79))

* working train/plot of pr ([`ea5e0c8`](https://github.com/mbari-org/vitstrain/commit/ea5e0c8d564d1fdb47be2a6bee91227e1d3e9826))

* Update src/utils.py

Co-authored-by: Copilot &lt;175728472+Copilot@users.noreply.github.com&gt; ([`72cad6d`](https://github.com/mbari-org/vitstrain/commit/72cad6d3e259ba997c9b9035d0a52465417a7132))

* Update src/utils.py

Co-authored-by: Copilot &lt;175728472+Copilot@users.noreply.github.com&gt; ([`216b7ff`](https://github.com/mbari-org/vitstrain/commit/216b7fffa2919deded67e44b4ebe9770b0ffd9ab))

* Use sklearn&#39;s train_test_split with stratify for balanced benchmark splits

Replaced HuggingFace’s train_test_split with sklearn’s version to enable stratification, ensuring splits are balanced by class. The &#39;stratify&#39; flag is always applied since there’s no case where unbalanced splits are desirable. ([`36bb05a`](https://github.com/mbari-org/vitstrain/commit/36bb05a65af25573a042f0fe01cf6783a919ecf5))


## v0.3.0 (2025-04-03)

### Feature

* feat: support more model preprocessors, e.g. DINO and VITS ([`e4941fc`](https://github.com/mbari-org/vitstrain/commit/e4941fce7e35b5bdad05905cbb405d51ac3779a9))


## v0.2.1 (2025-03-31)

### Performance

* perf: avoid keeping mean/std compute in memory ([`c5102e8`](https://github.com/mbari-org/vitstrain/commit/c5102e8c8e3066d6b581fe5d2242bab5d9b7c373))


## v0.2.0 (2025-03-29)

### Feature

* feat: add support for remapping classes through a simple json file, remove default raw data and print version ([`bb2e3bd`](https://github.com/mbari-org/vitstrain/commit/bb2e3bd5d6ea033d3294f70c5ac9a0b4a0cc8353))


## v0.1.0 (2025-03-29)

### Build

* build: added missing deps ([`41d6bd5`](https://github.com/mbari-org/vitstrain/commit/41d6bd5c9beb8471b78be785f7be1ec186a1a710))

* build: added accelerate ([`a2ad8ff`](https://github.com/mbari-org/vitstrain/commit/a2ad8ff8ee1823395f6aee2384805c163da1b7a1))

* build: remove pip cache ([`dc778fa`](https://github.com/mbari-org/vitstrain/commit/dc778faf9275e56aa0da037cdbdff35f103e0773))

* build: pin requirements and put pip cache in tmp ([`fef2ebd`](https://github.com/mbari-org/vitstrain/commit/fef2ebd3214a25dd9c40bfa3badad4fb656b8c7d))

* build: transformers need torch for acceleration ([`4c76041`](https://github.com/mbari-org/vitstrain/commit/4c760411256920ad6a7e3ecc0f793999fd4edd2d))

* build: pin requirements to CUDA 12.1 compatible ([`473693b`](https://github.com/mbari-org/vitstrain/commit/473693b6c2e4db7cbe81cd9a36cc5ef5db9dda34))

* build: added missing deps ([`495f124`](https://github.com/mbari-org/vitstrain/commit/495f124f5875fb2f5ea8a7c9bf32796822641210))

* build: added missing deps ([`ed6bb47`](https://github.com/mbari-org/vitstrain/commit/ed6bb47b78fd9ddc98ed156cba5499359c37031f))

### Documentation

* docs: updated example with explicit data versioning and some simplification for clarity ([`2df4d4c`](https://github.com/mbari-org/vitstrain/commit/2df4d4c38201aa9a503a4e0b85fdf43b02a3fb3f))

* docs: added tree output ([`ff11ec2`](https://github.com/mbari-org/vitstrain/commit/ff11ec2678b8aedb8f626952403ebe63a870e1ee))

* docs: adjust size of images ([`da299f4`](https://github.com/mbari-org/vitstrain/commit/da299f4759d842f487808aa7fa82406b369db9cb))

* docs: added example output images ([`d595b3d`](https://github.com/mbari-org/vitstrain/commit/d595b3deeae8f76b439262cea34203c751907e16))

### Feature

* feat: support png images ([`e67a9f5`](https://github.com/mbari-org/vitstrain/commit/e67a9f5e084c2407513f99d8f87b635e59edf3e2))

* feat: save deleted tail labels to deleted_labels.json ([`be99bbb`](https://github.com/mbari-org/vitstrain/commit/be99bbb2fd181bd189bfb03f8c6118e2c0d31abe))

* feat: correct stats if any errors which can happen from cleanvision ([`dcc9dea`](https://github.com/mbari-org/vitstrain/commit/dcc9deaae51927a83b6606361c4497cc50ec0b2b))

* feat: save stats.json to output of combined datasets ([`646c611`](https://github.com/mbari-org/vitstrain/commit/646c611157d81d8ebc59c974e09f465f7209f89c))

* feat: added --early-stopping-epochs with default 2 and other minor refactoring to avoid arg collisions ([`0fbd3b6`](https://github.com/mbari-org/vitstrain/commit/0fbd3b6896c1374610f0879942873698f65dc208))

* feat: added rotations arg ([`dcba283`](https://github.com/mbari-org/vitstrain/commit/dcba283cd6dcbe8a928a9aeae0a5e05b411d76e9))

* feat: added num epochs ([`5209e64`](https://github.com/mbari-org/vitstrain/commit/5209e64c0c01bb4f40b511c17077741aaeef84ce))

* feat: added command line ([`27b789a`](https://github.com/mbari-org/vitstrain/commit/27b789a1c76beb35c24385011ceb58c9a85f675a))

* feat: added loss curve persistence and plot ([`e9dd86c`](https://github.com/mbari-org/vitstrain/commit/e9dd86caf274c03529d9fa5199f5d9ac81e2e2e9))

* feat: support combining dataset ([`83e5dc1`](https://github.com/mbari-org/vitstrain/commit/83e5dc13aa68deebacbd6f401270af32a2334349))

* feat: resume from checkpoint ([`1da526b`](https://github.com/mbari-org/vitstrain/commit/1da526b56310d2342f9b220fb59c0738b3c69142))

* feat: load last checkpoint and save disk space by only caching last checkpoint ([`752f050`](https://github.com/mbari-org/vitstrain/commit/752f0500a4bd24971ceaead68c534e933ae850a0))

* feat: initial commit ([`aad01e1`](https://github.com/mbari-org/vitstrain/commit/aad01e177daf92e0b428fe2222ec0d1eff99d793))

### Fix

* fix: sort label keys ([`68700f8`](https://github.com/mbari-org/vitstrain/commit/68700f8d939ef9e14a793c0a81ae78872e19c9bf))

* fix: minor changes to roll back defaults on early stopping and put random resize crop back in which causes the default cats/dogs data to fail ([`c0525ee`](https://github.com/mbari-org/vitstrain/commit/c0525eec4c465e5a2bcea2badf42d34b5fc1b66b))

* fix: Focal loss with kwards ([`c3e950a`](https://github.com/mbari-org/vitstrain/commit/c3e950a4cf231ef1e699c813ee11cc1090c12eae))

* fix: set the mean and std to the training set in the process ([`4d8cb51`](https://github.com/mbari-org/vitstrain/commit/4d8cb5114c43e7e8795c0748b3161b5e85190acc))

* fix: correct handling of default raw-data arg ([`3a31287`](https://github.com/mbari-org/vitstrain/commit/3a31287203588fa67f0e46fcf54acc3c091067f7))

* fix: merge conflict ([`27de267`](https://github.com/mbari-org/vitstrain/commit/27de267a22d15db7be2845144c1222d2100686b8))

* fix: handle eval step gap ([`a61684e`](https://github.com/mbari-org/vitstrain/commit/a61684e30943f14c8cf2e14b72e4bf95f735ea0e))

* fix: override image mean in processor and ([`4bab074`](https://github.com/mbari-org/vitstrain/commit/4bab07432193f093b9208bb69ebe72e66f3dd69a))

* fix: correct labels on confusion matrix plot ([`4b0d6e8`](https://github.com/mbari-org/vitstrain/commit/4b0d6e8fb71b1f993deb7929b1a787479a65daec))

* fix: minor fix to stats combine ([`23dce9a`](https://github.com/mbari-org/vitstrain/commit/23dce9a8190acf13aa5deea251c7a14aed9128b7))

* fix: correct stat combine ([`9609a4c`](https://github.com/mbari-org/vitstrain/commit/9609a4cc6917cb59d078868c733689f1b22d2fbe))

### Performance

* perf: switch to accuracy ([`aad9bf1`](https://github.com/mbari-org/vitstrain/commit/aad9bf11d97d31518fff0b71bf686b1f0a6fcbdc))

* perf: better trainer arguments ([`f536ea1`](https://github.com/mbari-org/vitstrain/commit/f536ea10cd407031b7dd15ca6fe1633d999b93c4))

* perf: reduce early stopping patience to 2 ([`216f06c`](https://github.com/mbari-org/vitstrain/commit/216f06ce78b2c799d608c3ff2a6362a24248cedf))

* perf: add early stopping ([`1ab1278`](https://github.com/mbari-org/vitstrain/commit/1ab12788a3b584cb3bf934a69ef033b0ebc24a48))

* perf: change default alpha to 0.75 and rename loss variable ([`9c09b5a`](https://github.com/mbari-org/vitstrain/commit/9c09b5a69930e2f04c1ac25fb63450131ed6b799))

* perf: remove long tail &lt; 50 examples ([`59c263a`](https://github.com/mbari-org/vitstrain/commit/59c263afcbb9c0c161a7c32a34d9fc96ec337948))

* perf: use fast processor for speed-up ([`4808be7`](https://github.com/mbari-org/vitstrain/commit/4808be799be09401e028b8405db0a2ba2fa4940f))

* perf: normalize mean/std per expected AutoImageProcessor format ([`f8dd68f`](https://github.com/mbari-org/vitstrain/commit/f8dd68fe8b1c080e4e8f967d723bae5e55721f25))

* perf: added mean/std per the dataset ([`476e38d`](https://github.com/mbari-org/vitstrain/commit/476e38d5f82aaeb88cfd522933707cc1b4dc4b7b))

* perf: remove random downsample and measure balanced accuracy ([`b62ee12`](https://github.com/mbari-org/vitstrain/commit/b62ee12fd75369836e709698bf613a80b6f326b7))

* perf: more accurate val augmentations and longer training ([`63b9480`](https://github.com/mbari-org/vitstrain/commit/63b948034b4c24a1a536ba64b44b5bb682c7776f))

* perf: allow for retraining fine-tuned model and add simclr like augmentations ([`098ab1c`](https://github.com/mbari-org/vitstrain/commit/098ab1cf0489b7d91946015e049fc8b2cc145034))

* perf: added albumentations and accelerate ([`cba5293`](https://github.com/mbari-org/vitstrain/commit/cba52933129cf5927dd2f5c43b3b105a63c96079))

* perf: improved output of metrics with CM, accuracy, precision, recall and correct eval train argument ([`6440bcb`](https://github.com/mbari-org/vitstrain/commit/6440bcbc8e2e97e2c4bd67e88ab4672229b81d14))
