# OpenPointClass (OPC)

A fast, memory efficient free and open source point cloud classifier. It generates an AI model from a set of input point clouds that have been labeled and can subsequently use that model to classify new datasets.

![image](https://user-images.githubusercontent.com/1951843/222988483-2640ec66-7a4f-4bb0-a396-f1de84b46959.png)

![image](https://user-images.githubusercontent.com/1951843/222988854-afd47307-8ded-4c23-a322-f3c718ce70b8.png)

On the default parameters it can classify 15 million points in less than 2 minutes on a 4-core Intel i5, which is faster than any other freely available software we've tried.

It generalizes well to point clouds of varying density and includes local smoothing regularization methods.

It supports all point cloud formats supported by [PDAL](https://pdal.io/en/latest/stages/readers.html). When built without PDAL, it supports a subset of the PLY format only, which is optimized for speed.

## Build

Dependencies:
 * Intel TBB
 * PDAL (optional for LAS/LAZ support)

```
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

```
./pctrain ./ground_truth.ply
./pcclassify ./dataset.ply ./classified.ply [model.bin]
```

We provide access to a pre-trained model if you don't have access to labeled data. Please note the model was generated using a limited number of samples and it might not work well with all datasets.

 * [model.bin](https://github.com/uav4geo/OpenPointClass/releases/download/v1.1.1/model.zip)

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

### Supported Classes

| Class | Number |
--------|---------
| unclassified | 1 |
| ground | 2 |
| low_vegetation | 3 |
| medium_vegetation | 4 |
| high_vegetation | 5 |
| building | 6 |
| low_point | 7 |
| water | 9 |
| rail | 10 |
| road_surface | 11 |
| wire_guard | 13 |
| wire_conductor | 14 |
| transmission_tower | 15 |
| wire_structure_connector | 16 |
| bridge_deck | 17 |
| high_noise | 18 |
| overhead_structure | 19 |
| ignored_ground | 20 |
| snow | 21 |
| temporal_exclusion | 22 |
| vehicle | 64 |

### Evaluation

You can check a model accuracy by using the `--eval` argument:

`./pctrain ./ground_truth.ply --eval test.ply`

You can use [PDAL](https://pdal.io) to conveniently split a dataset into two (one for training, one for evaluation):

`pdal split [--capacity numpoints] input.ply input_split.ply`

### Color Output

You can output the results of classification as a colored point cloud by using the `--color` option:

`./pcclassify ./dataset.ply ./classified.ply --color`

### Advanced Options

See `./pctrain --help`.

### Docker

You can build a Docker image with the following command:

```bash
docker build -t uav4geo/openpointclass:latest .
```

Run the image with the following command:

```bash
docker run -it --rm -v /dataset-path:/data uav4geo/openpointclass:latest bash
```
Where `/dataset-path` is the path to the directory containing the dataset files and the `model.bin` file.

You will be presented with a bash prompt inside the container. You can then run the `pctrain` and `pcclassify` as described above.

Otherwise, you can use the commands directly with the following syntax:

```bash
docker run -it --rm -v /dataset-path:/data uav4geo/openpointclass:latest pctrain /data/ground_truth.ply
docker run -it --rm -v /dataset-path:/data uav4geo/openpointclass:latest pcclassify /data/dataset.ply /data/classified.ply /data/model.bin
```



## Known Issues

 * Gradient Boosted Trees as an alternative to Random Forest are currently broken.
 * We only support a subset of the PLY format (for performance reasons) and certain less common variations of the format might give trouble. Most importantly, the X/Y/Z coordinates must be `float` values when using binary PLY, not `double` or `float64`. We recommend to use LAS/LAZ if higher precision coordinates are needed.

## License

The software is released under the terms of the AGPLv3

[Contact us](https://uav4geo.com/contact) for other commercial licensing options.

Made with ❤️ by [UAV4GEO](https://uav4geo.com)
