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

Any format supported by [PDAL](https://pdal.io) is also accepted as input.

Training classes are assumed to follow the [ASPRS 1.4 Classification](https://www.asprs.org/wp-content/uploads/2019/03/LAS_1_4_r14.pdf) and to be stored in either a `label`, `class` or `classification` property.

You can re-map classification codes by creating a `<FILE>.json` in the same directory as `<TRAINING>.ply`:

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

## License

AGPLv3
