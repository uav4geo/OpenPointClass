# PCClassify

This is a work in progress. It currently generates only colored PLY outputs but it can be used for research already.

## Build

Dependencies:
 * PDAL
 * Intel TBB
 * Boost System and Serialization

```
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Run

```
./pctrain ./training.ply ./model.bin
./pcclassify ./dataset.ply ./model.bin ./classified.ply
```

Training classes are assumed to follow the [ASPRS 1.4 Classification](https://www.asprs.org/wp-content/uploads/2019/03/LAS_1_4_r14.pdf) and to be stored in either a `label`, `class` or `classification` property.

You can re-map classification codes by creating a `<FILE>.json` in the same directory as `<FILE>.ply`:

```
{
    "source": "https://url-to-your-data",
    "classification": {
        "0": "ground",
        "1": "building",
        "2": "low_vegetation",
        "3": "medium_vegetation",
        "4": "high_vegetation",
        "17": "ground"
    }
}
```

## Evaluation

You can automatically evaluate model accuracy by placing a `<FILE>_eval.ply` (a matching filename with `_eval` suffix). For example:

`./pctrain ./training.ply ./model.bin`

Will evaluate the model on `training_eval.ply` if it's available.

You can use PDAL to conveniently split a dataset into two (one for training, one for evaluation):

`pdal split [--capacity numpoints] input.ply input_split.ply` 

## License

AGPLv3
