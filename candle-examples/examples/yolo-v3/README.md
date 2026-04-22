# candle-yolo-v3:

Candle implementation of Yolo-V3 for object detection.

## Running an example

```bash
$ cargo run --example yolo-v3 --release -- candle-examples/examples/yolo-v8/assets/bike.jpg

> generated predictions Tensor[dims 10647, 85; f32]
> person: Bbox { xmin: 46.362198, ymin: 72.177, xmax: 135.92522, ymax: 339.8356, confidence: 0.99705493, data: () }
> person: Bbox { xmin: 137.25645, ymin: 67.58148, xmax: 216.90437, ymax: 333.80756, confidence: 0.9898516, data: () }
> person: Bbox { xmin: 245.7842, ymin: 82.76726, xmax: 316.79053, ymax: 337.21613, confidence: 0.9884322, data: () }
> person: Bbox { xmin: 207.52783, ymin: 61.815224, xmax: 266.77884, ymax: 307.92606, confidence: 0.9860648, data: () }
> person: Bbox { xmin: 11.457404, ymin: 60.335564, xmax: 34.39357, ymax: 187.7714, confidence: 0.9545012, data: () }
> person: Bbox { xmin: 251.88353, ymin: 11.235481, xmax: 286.56607, ymax: 92.54697, confidence: 0.8439807, data: () }
> person: Bbox { xmin: -0.44309902, ymin: 55.486923, xmax: 13.160354, ymax: 184.09705, confidence: 0.8266243, data: () }
> person: Bbox { xmin: 317.40826, ymin: 55.39501, xmax: 370.6704, ymax: 153.74887, confidence: 0.7327442, data: () }
> person: Bbox { xmin: 370.02835, ymin: 66.120224, xmax: 404.22824, ymax: 142.09691, confidence: 0.7265741, data: () }
> person: Bbox { xmin: 250.36511, ymin: 57.349842, xmax: 280.06335, ymax: 116.29384, confidence: 0.709422, data: () }
> person: Bbox { xmin: 32.573215, ymin: 66.66239, xmax: 50.49056, ymax: 173.42068, confidence: 0.6998766, data: () }
> person: Bbox { xmin: 131.72215, ymin: 63.946213, xmax: 166.66151, ymax: 241.52773, confidence: 0.64457536, data: () }
> person: Bbox { xmin: 407.42416, ymin: 49.106407, xmax: 415.24307, ymax: 84.7134, confidence: 0.5955802, data: () }
> person: Bbox { xmin: 51.650482, ymin: 64.4985, xmax: 67.40904, ymax: 106.952385, confidence: 0.5196007, data: () }
> bicycle: Bbox { xmin: 160.10031, ymin: 183.90837, xmax: 200.86832, ymax: 398.609, confidence: 0.9623588, data: () }
> bicycle: Bbox { xmin: 66.570915, ymin: 192.56966, xmax: 112.06765, ymax: 369.28497, confidence: 0.9174347, data: () }
> bicycle: Bbox { xmin: 258.2856, ymin: 197.04532, xmax: 298.43106, ymax: 364.8627, confidence: 0.6851388, data: () }
> bicycle: Bbox { xmin: 214.0034, ymin: 175.76498, xmax: 252.45158, ymax: 356.53818, confidence: 0.67071193, data: () }
> motorbike: Bbox { xmin: 318.23938, ymin: 95.22487, xmax: 369.9743, ymax: 213.46263, confidence: 0.96691036, data: () }
> motorbike: Bbox { xmin: 367.46417, ymin: 100.07982, xmax: 394.9981, ymax: 174.6545, confidence: 0.9185384, data: () }
> writing "candle-examples/examples/yolo-v8/assets/bike.pp.jpg"
```